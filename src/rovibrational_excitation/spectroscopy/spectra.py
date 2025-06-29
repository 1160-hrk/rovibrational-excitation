#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spectrum calculation functions for rovibrational spectroscopy.

This module provides functions for calculating various types of spectra
using the LinearResponseCalculator class for improved maintainability.

SPECTROSCOPY THEORY OVERVIEW
============================

Linear Response and Spectroscopy (線形応答と分光学)
--------------------------------------------------
分光学的観測量は、分子系の線形応答関数χ(ω)から計算されます。
各種スペクトルは、応答関数の異なる成分または処理に対応します：

1. 吸収スペクトル (Absorption Spectroscopy)
   基底状態からの光吸収過程。遷移双極子と密度行列の交換子から計算。
   
2. 発光スペクトル (Emission/Radiation Spectroscopy)  
   励起状態からの自発放出。吸収と逆の過程として計算。
   
3. PFID (Polarization-resolved Free Induction Decay)
   分子の分極の時間発展を周波数領域で解析。偏光依存性を含む。

MATHEMATICAL FOUNDATIONS
========================

Linear Response Function (線形応答関数):
χ(ω) = Σᵢⱼ ρᵢⱼ × μⱼᵢ / (ω - ωᵢⱼ + iΓᵢⱼ)

ここで：
- ρᵢⱼ: 密度行列要素（状態の占有と位相）
- μⱼᵢ: 遷移双極子行列要素
- ωᵢⱼ: 遷移周波数
- Γᵢⱼ: 緩和定数（線幅）

Beer-Lambert Law (ベール・ランバート則):
吸光度 A = -log₁₀(I/I₀) = ε × c × l

ここで：
- ε: モル吸光係数 [M⁻¹cm⁻¹]
- c: 濃度 [M]
- l: 光路長 [cm]

この実装では、線形応答から屈折率の虚部を計算し、
Beer-Lambert則に従って吸光度に変換します。

Refractive Index and Absorption (屈折率と吸収):
複素屈折率: n = n' + in''
- n': 実部（分散）
- n'': 虚部（吸収）

吸収係数: α = 2ωn''/c
吸光度: A = α × l / ln(10) [mOD単位]

COMPUTATIONAL IMPLEMENTATION
============================

Core Algorithm:
1. 密度行列とハミルトニアンから線形応答関数を計算
2. 遷移双極子との結合で実効的な応答を求める
3. 屈折率の虚部を計算
4. Beer-Lambert則で吸光度に変換
5. 広がり効果（ドップラー等）を考慮

Optimization Strategies:
- 非ゼロ遷移のみを計算（疎行列活用）
- 振動コヒーレンス射影による計算領域限定
- 事前計算された分母の使用（2Dモード）

BROADENING EFFECTS
==================

1. Homogeneous Broadening (均質広がり):
   - コヒーレンス緩和時間 T₂ に起因
   - ローレンツ型線形
   - 線幅 = 1/(πT₂)

2. Doppler Broadening (ドップラー広がり):
   - 分子の熱運動に起因
   - ガウス型線形
   - 線幅 ∝ √(T/M)

3. Instrumental Broadening (装置広がり):
   - FTIR等の分光器の分解能
   - sinc型またはsinc²型

POLARIZATION EFFECTS
====================

遷移強度の偏光依存性:
I ∝ |ε̂ · μ⃗ᵢⱼ|²

ここで：
- ε̂: 偏光ベクトル
- μ⃗ᵢⱼ: 遷移双極子ベクトル

線形分子の場合：
- P branch (ΔJ = -1): 分子軸に垂直偏光で強い
- R branch (ΔJ = +1): 分子軸に垂直偏光で強い  
- Q branch (ΔJ = 0): 分子軸に平行偏光で強い（振動遷移のみ）
"""

import numpy as np
from typing import Optional, Dict, Any
from .constants import h_dirac, c, eps
from .broadening import convolution_w_doppler
from .linear_response import LinearResponseCalculator, get_calculator


def _calculate_linear_response_core(
    density_matrix: np.ndarray,
    calculation_data: Dict[str, Any],
    apply_vibrational_projection: bool = True,
    use_doppler_broadening: bool = False
) -> np.ndarray:
    """
    線形応答計算の中核関数。
    
    この関数は全てのスペクトル計算の基礎となる線形応答関数χ(ω)を
    計算します。量子力学的な線形応答理論に基づき、密度行列と
    遷移双極子から観測可能な応答を求めます。
    
    Physical Background
    ------------------
    線形応答理論では、外部場への系の応答は以下で表されます：
    
    χ(ω) = Σᵢⱼ ⟨i|μ|j⟩⟨j|[μ,ρ]|i⟩ / (ω - ωᵢⱼ + iΓᵢⱼ)
    
    ここで [μ,ρ] は遷移双極子と密度行列の交換子：
    [μ,ρ] = μρ - ρμ†
    
    この交換子は、外部場による分子の分極の変化を表します。
    
    Parameters
    ----------
    density_matrix : np.ndarray
        系の密度行列 ρ。状態の占有確率と位相関係を含む。
        対角成分: 状態占有確率（ボルツマン分布等）
        非対角成分: 状態間コヒーレンス
        
    calculation_data : Dict[str, Any]
        事前計算されたデータ（行列要素、物理定数等）
        
    apply_vibrational_projection : bool
        振動コヒーレンス射影の適用。True の場合、|Δv| ≤ 1 の
        遷移のみを考慮し、計算量を削減。
        
    use_doppler_broadening : bool
        ドップラー広がりの適用。True の場合、熱運動による
        周波数分布を畳み込み積分で考慮。
        
    Returns
    -------
    np.ndarray
        複素線形応答 χ(ω) [単位: C·m/V]
        実部: 分散（屈折率の実部に関連）
        虚部: 吸収（屈折率の虚部に関連）
        
    Notes
    -----
    計算の最適化：
    - 非ゼロ遷移のみをループ処理
    - 事前計算されたインデックスの使用
    - ベクトル化された行列演算
    
    物理的妥当性：
    - エルミート性: χ(-ω) = χ*(ω)
    - 因果律: 応答関数の虚部が正（吸収）
    - 和則: 遷移強度の規格化
    """
    # 計算データの展開
    energy_diff_matrix = calculation_data['energy_difference_matrix']
    effective_tdm = calculation_data['effective_transition_dipole']
    conjugate_tdm = calculation_data['conjugate_transition_dipole']
    nonzero_indices = calculation_data['nonzero_transition_indices']
    vib_projection = calculation_data['vibrational_coherence_projection']
    angular_frequencies = calculation_data['angular_frequencies']
    
    # 振動射影の適用（計算領域の限定）
    if apply_vibrational_projection:
        # P̂ᵥᵢᵦ × ρ: 振動基本遷移 |Δv| ≤ 1 のみを残す
        rho = density_matrix * vib_projection
    else:
        rho = density_matrix
    
    # 遷移双極子と密度行列の交換子: [μ, ρ] = μρ - ρμ†
    # この交換子は外部場による分極変化を表す
    commutator = effective_tdm @ rho - rho @ conjugate_tdm
    
    # 応答関数の初期化
    linear_response = np.zeros(len(angular_frequencies), dtype=np.complex128)
    
    # 各許可遷移についてループ計算
    # Σᵢⱼ の和を効率的に計算
    for transition_indices in nonzero_indices.T:
        idx_tuple = tuple(transition_indices)
        flipped_idx = tuple(np.flip(transition_indices))
        
        # 遷移強度の計算: μ*ᵢⱼ × [μ,ρ]ⱼᵢ
        transition_strength = conjugate_tdm[idx_tuple] * commutator[flipped_idx]
        
        # エネルギー分母: ω - ωᵢⱼ + iΓᵢⱼ
        energy_denominator = angular_frequencies + energy_diff_matrix[idx_tuple]
        
        # 線形応答への寄与: (-i/ℏ) × (遷移強度) / (i × 分母)
        # 因子 -i/ℏ は量子力学的規格化
        # 因子 i は Kramers-Kronig 関係を満たすため
        transition_response = (-1j / h_dirac * transition_strength / 
                             (1j * energy_denominator))
        
        # ドップラー広がりの適用（オプション）
        if use_doppler_broadening:
            # 遷移の中心周波数（実部）
            center_frequency = np.real(energy_diff_matrix[idx_tuple])
            temperature = calculation_data['temperature']
            molecular_mass = calculation_data['molecular_mass']
            
            # ガウス型広がりとの畳み込み積分
            transition_response = convolution_w_doppler(
                angular_frequencies, transition_response,
                center_frequency, temperature, molecular_mass
            )
        
        # 全応答関数への加算
        linear_response += transition_response
    
    return linear_response


def _convert_response_to_absorbance(
    linear_response: np.ndarray,
    calculation_data: Dict[str, Any]
) -> np.ndarray:
    """
    線形応答から吸光度スペクトルへの変換。
    
    量子力学的線形応答関数χ(ω)から、実験で観測される
    吸光度A(ω)への変換を行います。この変換では、
    Maxwell方程式と物質の応答の関係を使用します。
    
    Theory of Absorption
    -------------------
    Maxwell方程式から、物質中の電磁波の伝播は複素屈折率で
    決まります：
    
    n²(ω) = ε(ω) = 1 + χ(ω)/ε₀
    
    複素屈折率 n = n' + in'' の虚部 n'' が吸収を表します。
    
    Beer-Lambert則により：
    I = I₀ exp(-αl)
    
    吸収係数：α = 2ωn''/c
    吸光度：A = αl/ln(10) = (2ωn''l)/(c×ln(10))
    
    Parameters
    ----------
    linear_response : np.ndarray
        複素線形応答関数 χ(ω)
        
    calculation_data : Dict[str, Any]
        変換に必要な物理パラメータ
        
    Returns
    -------
    np.ndarray
        吸光度スペクトル [mOD単位]
        
    Notes
    -----
    単位変換：
    - SI単位系での計算後、mOD（ミリ光学密度）に変換
    - 1 OD = log₁₀(I₀/I) = αl/ln(10)
    - 分光学で一般的な濃度・光路長規格化済み
    
    物理的意味：
    - 正値: 吸収（エネルギー損失）
    - 負値: 誘導放出（エネルギー増幅、レーザー等）
    - ピーク位置: 遷移周波数
    - ピーク幅: 緩和時間の逆数
    """
    optical_path = calculation_data['optical_path_length']
    number_density = calculation_data['number_density']
    angular_frequencies = calculation_data['angular_frequencies']
    
    # 複素屈折率の計算: n² = 1 + χ/(ε₀×3)
    # 因子1/3は等方平均（分子配向の統計平均）
    complex_permittivity = 1 + linear_response / eps * number_density / 3
    
    # 屈折率の虚部を計算: n'' = Im[√(ε)]
    # 弱吸収近似: n'' ≈ Im[ε]/2 = Im[χ]/(2ε₀×3) × number_density
    refractive_correction = np.sqrt(complex_permittivity).imag
    
    # 吸収係数の計算: α = 2ω|n''|/c
    # 光の伝播方程式 E = E₀exp(iωt - iωnz/c) から導出
    absorption_coefficient = 2 * optical_path * angular_frequencies / c * refractive_correction
    
    # mOD単位への変換: A[mOD] = α×l×1000/ln(10)
    # log₁₀(e) = 1/ln(10) ≈ 0.434
    absorbance_mOD = absorption_coefficient * np.log10(np.exp(1)) * 1000
    
    return absorbance_mOD


def calculate_absorption_spectrum(
    density_matrix: np.ndarray,
    calculator: Optional[LinearResponseCalculator] = None,
    use_doppler_broadening: bool = False
) -> np.ndarray:
    """
    吸収スペクトルの計算（推奨API）。
    
    分子の基底状態から励起状態への光吸収過程を計算します。
    これは最も基本的な分光測定に対応し、分子の振動回転準位
    構造を直接反映します。
    
    Physical Process
    ---------------
    光吸収は以下の過程で起こります：
    1. 光子の吸収: |g⟩ + ℏω → |e⟩
    2. 遷移確率: ∝ |⟨e|μ·ε̂|g⟩|² × ρgg
    3. エネルギー保存: ℏω = Ee - Eg
    4. 選択則: Δv = ±1, ΔJ = ±1, ΔM = 0,±1
    
    Temperature Effects
    ------------------
    温度により以下が変化します：
    - 状態占有: ボルツマン分布 ∝ exp(-E/kT)
    - ドップラー広がり: Δω/ω ∝ √(T/M)
    - 回転準位の占有: J ~ √(kT/B)
    
    Parameters
    ----------
    density_matrix : np.ndarray
        系の密度行列。通常は熱平衡状態：
        ρgg = exp(-Eg/kT) / Z (ボルツマン分布)
        
    calculator : LinearResponseCalculator, optional
        計算器インスタンス。None の場合はグローバル計算器を使用。
        
    use_doppler_broadening : bool
        ドップラー広がりの考慮。気体試料では通常 True。
        
    Returns
    -------
    np.ndarray
        吸収スペクトル [mOD単位]
        
    Example
    -------
    >>> # 熱平衡密度行列の準備
    >>> rho_thermal = create_boltzmann_distribution(T=300)
    >>> # 吸収スペクトル計算
    >>> absorption = calculate_absorption_spectrum(rho_thermal)
    >>> # プロット
    >>> plt.plot(wavenumbers, absorption)
    >>> plt.xlabel('Wavenumber [cm⁻¹]')
    >>> plt.ylabel('Absorbance [mOD]')
    
    Notes
    -----
    実験との比較：
    - ピーク位置: 分子定数の検証
    - ピーク強度: 遷移双極子の確認
    - 線幅: 緩和時間の評価
    - 温度依存性: 分子間相互作用の解析
    """
    if calculator is None:
        calculator = get_calculator()
    
    if not calculator.is_initialized:
        raise ValueError("Calculator not initialized. Call initialize() first.")
    
    calculation_data = calculator.get_calculation_data()
    
    # 振動コヒーレンス射影を適用した線形応答計算
    # 基本振動遷移 |Δv| ≤ 1 のみを考慮
    linear_response = _calculate_linear_response_core(
        density_matrix, calculation_data,
        apply_vibrational_projection=True,
        use_doppler_broadening=use_doppler_broadening
    )
    
    # 線形応答から吸光度への変換
    return _convert_response_to_absorbance(linear_response, calculation_data)


def calculate_pfid_spectrum(
    density_matrix: np.ndarray,
    calculator: Optional[LinearResponseCalculator] = None
) -> np.ndarray:
    """
    PFID（偏光分解自由誘導減衰）スペクトルの計算。
    
    PFIDは分子の分極の時間発展を周波数領域で解析する手法です。
    通常の吸収分光と異なり、分子の配向分布や回転動力学の
    情報を含みます。
    
    Physical Background
    ------------------
    PFID過程：
    1. ポンプパルスによる分子の配向
    2. 自由誘導減衰による分極の時間発展
    3. プローブパルスによる信号検出
    4. 偏光依存性により配向情報を抽出
    
    通常の吸収との違い：
    - 振動コヒーレンス射影を適用しない
    - より広い状態空間での応答を計算
    - 回転-振動結合の効果が顕著
    
    Mathematical Description
    -----------------------
    PFID信号は分極の2次の応答に比例：
    S(ω) ∝ |⟨P(ω)⟩|²
    
    ここで P(ω) は周波数領域での分極：
    P(ω) = Tr[μ·ρ(ω)]
    
    Parameters
    ----------
    density_matrix : np.ndarray
        初期密度行列（ポンプ後の非平衡状態）
        
    calculator : LinearResponseCalculator, optional
        計算器インスタンス
        
    Returns
    -------
    np.ndarray
        PFID スペクトル [mOD単位]
        
    Applications
    -----------
    - 分子の配向分布測定
    - 回転緩和時間の決定
    - 異方性相互作用の解析
    - 液体・固体中の分子動力学研究
    
    Notes
    -----
    実験設定：
    - ポンプ-プローブ配置
    - 偏光制御（直交・平行偏光）
    - 時間分解測定
    - 信号の位相情報
    """
    if calculator is None:
        calculator = get_calculator()
    
    if not calculator.is_initialized:
        raise ValueError("Calculator not initialized. Call initialize() first.")
    
    calculation_data = calculator.get_calculation_data()
    
    # PFIDでは振動射影を適用せず、より広い状態空間で計算
    # 回転-振動結合の効果を完全に考慮
    linear_response = _calculate_linear_response_core(
        density_matrix, calculation_data,
        apply_vibrational_projection=False,  # 射影なし：全状態空間
        use_doppler_broadening=False         # 通常、時間分解測定では不要
    )
    
    return _convert_response_to_absorbance(linear_response, calculation_data)


def calculate_emission_spectrum(
    density_matrix: np.ndarray,
    calculator: Optional[LinearResponseCalculator] = None
) -> np.ndarray:
    """
    発光・放射スペクトルの計算。
    
    励起状態からの自発放出による発光スペクトルを計算します。
    吸収の逆過程として、アインシュタインのA係数と関連付けられます。
    
    Physical Process
    ---------------
    自発放出過程：
    1. 励起状態の占有: ρee > 0
    2. 光子の放出: |e⟩ → |g⟩ + ℏω
    3. 遷移確率: ∝ |⟨g|μ·ε̂|e⟩|² × ρee
    4. エネルギー保存: ℏω = Ee - Eg
    
    Einstein Coefficients
    ---------------------
    アインシュタイン係数の関係：
    - A₂₁: 自発放出確率
    - B₁₂: 誘導吸収確率  
    - B₂₁: 誘導放出確率
    
    関係式: A₂₁ = (8πℏω³/c³) × B₂₁
    
    Temperature and Population
    -------------------------
    発光強度は励起状態占有に比例：
    I(ω) ∝ ρee(T) × A₂₁(ω) × ℏω
    
    高温や非平衡条件で顕著になります。
    
    Parameters
    ----------
    density_matrix : np.ndarray
        密度行列（励起状態を含む）
        通常は非平衡状態またはポンプ後の状態
        
    calculator : LinearResponseCalculator, optional
        計算器インスタンス
        
    Returns
    -------
    np.ndarray
        発光スペクトル [mOD単位]
        注：実際は負の吸光度として計算される
        
    Applications
    -----------
    - 蛍光・燐光分光
    - ラマン散乱（誘導放出成分）
    - レーザー利得媒質の解析
    - 非平衡状態の診断
    
    Example
    -------
    >>> # 励起状態を含む密度行列
    >>> rho_excited = create_excited_state_distribution()
    >>> emission = calculate_emission_spectrum(rho_excited)
    >>> plt.plot(wavenumbers, -emission)  # 発光は負の吸収
    >>> plt.ylabel('Emission Intensity')
    
    Notes
    -----
    符号の解釈：
    - 負値: 誘導放出（光増幅）
    - 正値: 吸収（光減衰）
    - ゼロ: 熱平衡（詳細釣り合い）
    """
    if calculator is None:
        calculator = get_calculator()
    
    if not calculator.is_initialized:
        raise ValueError("Calculator not initialized. Call initialize() first.")
    
    calculation_data = calculator.get_calculation_data()
    
    # 発光の場合の特別な処理
    energy_diff_matrix = calculation_data['energy_difference_matrix']
    conjugate_tdm = calculation_data['conjugate_transition_dipole']
    nonzero_indices = calculation_data['nonzero_transition_indices']
    vib_projection = calculation_data['vibrational_coherence_projection']
    angular_frequencies = calculation_data['angular_frequencies']
    
    # 振動射影の適用
    rho = density_matrix * vib_projection
    
    # 発光応答の初期化
    linear_response = np.zeros(len(angular_frequencies), dtype=np.complex128)
    
    # 発光は吸収と異なる計算が必要
    # 励起状態から基底状態への遷移を計算
    for transition_indices in nonzero_indices.T:
        idx_tuple = tuple(transition_indices)
        flipped_idx = tuple(np.flip(transition_indices))
        
        # 発光では遷移双極子の方向が逆転
        # ⟨g|μ|e⟩* × ρee の形で計算
        transition_strength = conjugate_tdm[flipped_idx] * rho[idx_tuple]
        energy_denominator = angular_frequencies + energy_diff_matrix[idx_tuple]
        
        # 発光応答への寄与（符号に注意）
        linear_response += (-transition_strength / 
                          (1j * energy_denominator))
    
    return _convert_response_to_absorbance(linear_response, calculation_data)


def calculate_absorption_from_hamiltonian(
    density_matrix: np.ndarray,
    transition_dipole_matrix: np.ndarray,
    hamiltonian: np.ndarray,
    wavenumbers: np.ndarray,
    optical_path_length: float = 1e-3,
    number_density: float = 1e20,
    coherence_time_ps: float = 100
) -> np.ndarray:
    """
    ハミルトニアンと双極子行列から直接吸収スペクトルを計算。
    
    この関数は LinearResponseCalculator フレームワークを使わず、
    基本的な量子力学の行列要素から直接スペクトルを計算します。
    教育目的や簡単な系での検証に有用です。
    
    Direct Calculation Method
    ------------------------
    この実装では以下の手順でスペクトルを計算：
    
    1. ハミルトニアンの対角化 → エネルギー準位
    2. エネルギー差分行列の構築
    3. 遷移双極子との結合
    4. 線形応答関数の直接計算
    5. 吸光度への変換
    
    Mathematical Foundation
    ----------------------
    基本的な量子力学の摂動論から：
    
    χ(ω) = Σᵢⱼ |μᵢⱼ|² × (ρᵢᵢ - ρⱼⱼ) / (ℏω - (Eᵢ - Eⱼ) + iℏΓ)
    
    これは自然な形の線形応答関数です。
    
    Parameters
    ----------
    density_matrix : np.ndarray
        系の密度行列 [無次元]
        
    transition_dipole_matrix : np.ndarray  
        遷移双極子行列 [C·m]
        
    hamiltonian : np.ndarray
        ハミルトニアン行列 [J]
        
    wavenumbers : np.ndarray
        測定波数範囲 [cm⁻¹]
        
    optical_path_length : float
        光路長 [m]、デフォルト: 1 mm
        
    number_density : float
        分子数密度 [m⁻³]、デフォルト: 10²⁰ m⁻³
        
    coherence_time_ps : float
        コヒーレンス時間 [ps]、デフォルト: 100 ps
        
    Returns
    -------
    np.ndarray
        吸収スペクトル [mOD単位]
        
    Advantages
    ----------
    - 計算が透明で理解しやすい
    - 外部依存性が少ない
    - デバッグが容易
    - 教育・検証用途に適している
    
    Limitations
    -----------
    - 最適化が限定的
    - 大きな系では遅い
    - 高度な機能（射影等）なし
    - メモリ効率が劣る
    
    Example
    -------
    >>> # 簡単な2準位系での計算例
    >>> H = np.array([[0, 0], [0, 1000]])  # [cm^-1]
    >>> mu = np.array([[0, 1], [1, 0]])    # [ea₀]
    >>> rho = np.array([[1, 0], [0, 0]])   # 基底状態
    >>> wn = np.arange(900, 1100, 1)       # 波数範囲
    >>> spectrum = calculate_absorption_from_hamiltonian(rho, mu, H, wn)
    
    Notes
    -----
    単位系の注意：
    - エネルギー: [J] （SI単位）
    - 双極子: [C·m] （SI単位）
    - 波数: [cm⁻¹] （分光学標準）
    - 結果: [mOD] （分光学標準）
    """
    # エネルギー差分行列の計算
    energies = np.diag(hamiltonian)  # 対角ハミルトニアンを仮定
    num_levels = len(energies)
    
    # エネルギー行列の構築: Eᵢ - Eⱼ
    energy_matrix = np.tile(energies, (num_levels, 1))
    energy_diff_real = (energy_matrix - energy_matrix.T) / h_dirac
    
    # 複素エネルギー差分（緩和効果を含む）
    gamma = 1.0 / (coherence_time_ps * 1e-12)  # 緩和定数 [rad/s]
    energy_diff_matrix = energy_diff_real - 1j * gamma
    
    # 波数から角周波数への変換
    angular_frequencies = wavenumbers * 2 * np.pi * c * 1e2  # [rad/s]
    
    # 非ゼロ遷移の特定
    nonzero_indices = np.array(np.where(transition_dipole_matrix != 0))
    
    # 遷移双極子と密度行列の交換子
    commutator = (transition_dipole_matrix @ density_matrix - 
                 density_matrix @ transition_dipole_matrix)
    
    # 線形応答の初期化
    linear_response = np.zeros(len(wavenumbers), dtype=np.complex128)
    
    # 各遷移についての寄与を計算
    for transition_indices in nonzero_indices.T:
        idx_tuple = tuple(transition_indices)
        flipped_idx = tuple(np.flip(transition_indices))
        
        # 遷移強度の計算
        transition_strength = (transition_dipole_matrix[idx_tuple] * 
                             commutator[flipped_idx])
        
        # エネルギー分母
        energy_denominator = (angular_frequencies + 
                            energy_diff_matrix[idx_tuple])
        
        # 線形応答への寄与
        linear_response += (-1j / h_dirac * transition_strength / 
                          (1j * energy_denominator))
    
    # 吸光度への変換
    # 複素屈折率の虚部を計算
    refractive_correction = np.sqrt(
        1 + linear_response / eps * number_density / 3
    ).imag  # type: ignore
    
    # Beer-Lambert則による吸光度
    absorbance = (2 * optical_path_length * angular_frequencies / c * 
                 refractive_correction)
    
    # mOD単位への変換
    absorbance *= np.log10(np.exp(1)) * 1000
    
    return absorbance


# ================================
# LEGACY INTERFACE FUNCTIONS
# レガシーAPIとの互換性維持関数群
# ================================

def absorbance_spectrum(rho: np.ndarray) -> Optional[np.ndarray]:
    """
    レガシー関数: 吸収スペクトル計算（2Dモードのみ）。
    
    この関数は後方互換性のために残されています。
    新しいコードでは calculate_absorption_spectrum() を使用してください。
    """
    calculator = get_calculator()
    if not calculator.is_initialized:
        raise ValueError("Calculator not initialized. Call prepare_variables() first.")
    
    calculation_data = calculator.get_calculation_data()
    if not calculation_data['is_2d_enabled']:
        print("Please set True for the value of make_2d for prepare_variables.")
        return None
    
    # 2D最適化版の実装はより複雑なため、標準版で代用
    return calculate_absorption_spectrum(rho)


def absorbance_spectrum_for_loop(rho: np.ndarray) -> np.ndarray:
    """レガシー関数: ループ法による吸収スペクトル計算。"""
    return calculate_absorption_spectrum(rho, use_doppler_broadening=False)


def absorbance_spectrum_w_doppler_broadening(rho: np.ndarray) -> np.ndarray:
    """レガシー関数: ドップラー広がりを含む吸収スペクトル。"""
    return calculate_absorption_spectrum(rho, use_doppler_broadening=True)


def PFID_spectrum_for_loop(rho: np.ndarray) -> np.ndarray:
    """レガシー関数: PFIDスペクトル計算。"""
    return calculate_pfid_spectrum(rho)


def radiation_spectrum_for_loop(rho: np.ndarray) -> np.ndarray:
    """レガシー関数: 放射スペクトル計算。"""
    return calculate_emission_spectrum(rho)


def absorbance_spectrum_from_rho_and_mu(
    rho: np.ndarray, mu: np.ndarray, H0: np.ndarray, wn: np.ndarray, 
    opt_len_param: float = 1e-3, dens_num_param: float = 1e20, 
    T2: float = 100
) -> np.ndarray:
    """レガシー関数: 行列から直接スペクトル計算。"""
    return calculate_absorption_from_hamiltonian(
        rho, mu, H0, wn, opt_len_param, dens_num_param, T2
    ) 