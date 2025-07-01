#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linear response calculations for rovibrational spectroscopy.

This module provides a class-based approach for managing linear response
theory calculations with improved maintainability and type safety.

THEORETICAL BACKGROUND
======================

Linear Response Theory (線形応答理論)
------------------------------------
線形応答理論は、弱い外部摂動に対する量子系の応答を扱う理論です。
密度行列形式では、系の時間発展は以下のリウヴィル方程式で記述されます：

    ∂ρ/∂t = -i/ℏ [H₀ + V(t), ρ] - Γ(ρ - ρ₀)

ここで：
- H₀: 無摂動ハミルトニアン
- V(t): 時間依存摂動（レーザー場との相互作用）
- Γ: 緩和演算子（フェノメノロジカル）
- ρ₀: 平衡密度行列

周波数領域での線形応答関数 χ(ω) は：
    χ(ω) = Σᵢⱼ (|μᵢⱼ|²/(ωᵢⱼ - ω - iΓᵢⱼ))

この実装では、振動回転状態間の遷移を考慮したχ(ω)を計算します。

Rovibrational Energy Levels (振動回転エネルギー準位)
--------------------------------------------------
線形分子のエネルギー準位は以下の式で表されます：

    E(v,J,M) = ωₑ(v + 1/2) - ωₑxₑ(v + 1/2)² + [Bᵥ - αₑ(v + 1/2)]J(J+1)

ここで：
- v: 振動量子数 (0, 1, 2, ...)
- J: 回転量子数 (0, 1, 2, ...)  
- M: 磁気量子数 (-J ≤ M ≤ J)
- ωₑ: 調和振動周波数
- ωₑxₑ: 非調和性定数
- Bᵥ: 回転定数
- αₑ: 振動-回転相互作用定数

Transition Dipole Moments (遷移双極子モーメント)
----------------------------------------------
電気双極子遷移の選択則：
- Δv = ±1 (基本振動遷移)
- ΔJ = ±1 (P, R branch)
- ΔM = 0, ±1 (π, σ± 偏光)

遷移双極子モーメント行列要素：
    μᵢⱼ = ⟨v',J',M'|μ|v,J,M⟩

この実装では、LinMolDipoleMatrixクラスが自動的に選択則を考慮した
遷移双極子行列を生成します。

IMPLEMENTATION DETAILS
======================

Core Algorithm Flow:
1. 基底状態の生成 (LinMolBasis)
2. 遷移双極子行列の計算 (LinMolDipoleMatrix)
3. エネルギー差分行列の構築
4. 線形応答関数の計算
5. 物理観測量への変換

Key Optimization Techniques:
- 非ゼロ遷移のみの計算（疎行列の活用）
- 振動コヒーレンス射影による計算領域の限定
- 事前計算されたエネルギー差分行列の使用
"""

import numpy as np
from typing import Optional, List, Tuple, Literal
from dataclasses import dataclass
from .constants import h_dirac, kb, c, m_CO2

# Import existing core modules
from ..core.basis import LinMolBasis
from ..dipole.linmol import LinMolDipoleMatrix


@dataclass
class SpectroscopyParameters:
    """
    分光測定に関するパラメータを管理するデータクラス。
    
    実験条件と測定設定を一元管理し、物理的に意味のある
    デフォルト値を提供します。
    
    Physical Meaning of Parameters
    ------------------------------
    coherence_relaxation_time_ps : float
        コヒーレンス緩和時間 T₂ [ps]。分子間衝突や環境との相互作用により
        量子コヒーレンスが失われる時間スケール。典型値: 100-1000 ps
        
    temperature_K : float  
        分子系の温度 [K]。ボルツマン分布による状態占有と
        ドップラー広がりに影響。室温: ~300 K
        
    pressure_Pa : float
        気体圧力 [Pa]。分子数密度の決定に使用。
        大気圧: ~10⁵ Pa, 低圧: ~10³ Pa
        
    laser_polarization, radiation_polarization : List[float]
        レーザー偏光と検出偏光の方向。[x, y, z] 成分で指定。
        直線偏光: [1,0,0], 円偏光: [1,1,0]/√2
    """
    coherence_relaxation_time_ps: float = 500.0
    molecular_mass_kg: float = m_CO2
    temperature_K: float = 300.0
    optical_path_length_m: float = 1e-3
    pressure_Pa: float = 3e4
    laser_polarization: Optional[List[float]] = None
    radiation_polarization_same: bool = True
    radiation_polarization: Optional[List[float]] = None
    wavenumbers_cm: Optional[np.ndarray] = None
    enable_2d_calculation: bool = False
    
    def __post_init__(self):
        if self.laser_polarization is None:
            self.laser_polarization = [1.0, 0.0]  # x偏光（分子軸方向）
        if self.radiation_polarization is None:
            self.radiation_polarization = [1.0, 0.0]
        if self.wavenumbers_cm is None:
            # CO₂ ν₃ mode (antisymmetric stretch) の典型的範囲
            self.wavenumbers_cm = np.arange(2100, 2400, 0.01)


@dataclass
class MolecularParameters:
    """
    分子の物理パラメータを管理するデータクラス。
    
    分子固有の定数を管理し、異なる分子系への拡張を容易にします。
    
    Physical Constants Explanation
    ------------------------------
    transition_dipole_moment : float
        基本遷移双極子モーメント [ea₀]。分子の電子分布変化による
        双極子モーメント変化。典型値: 0.1-1.0 ea₀
        
    vibrational_frequency_rad_per_fs : float  
        振動周波数 ωₑ [rad/fs]。調和振動子近似での基本周波数。
        CO₂ ν₃ mode: ~2349 cm⁻¹ ≈ 1000 rad/fs
        
    rotational_constant_rad_per_fs : float
        回転定数 Bₑ [rad/fs]。慣性モーメントに反比例。
        CO₂: ~0.39 cm⁻¹ ≈ 0.17 rad/fs
        
    vibration_rotation_coupling_rad_per_fs : float
        振動-回転相互作用定数 αₑ [rad/fs]。振動による慣性モーメント変化。
        通常 Bₑ の数%程度
        
    anharmonicity_correction_rad_per_fs : float
        非調和性定数 ωₑxₑ [rad/fs]。高次振動準位のエネルギー補正。
        通常 ωₑ の1-10%程度
    """
    transition_dipole_moment: float = 1.0
    vibrational_frequency_rad_per_fs: float = 1000.0
    rotational_constant_rad_per_fs: float = 1.0
    vibration_rotation_coupling_rad_per_fs: float = 0.0
    anharmonicity_correction_rad_per_fs: float = 0.0
    potential_type: Literal["harmonic", "morse"] = "harmonic"


class LinearResponseCalculator:
    """
    線形応答理論計算の管理クラス。
    
    このクラスは振動回転分光計算に必要な全ての状態とメソッドを
    カプセル化し、グローバル変数に依存しないクリーンな
    オブジェクト指向アプローチを提供します。
    
    CALCULATION WORKFLOW
    ===================
    
    1. 初期化 (initialize)
       - 量子数に基づく基底状態の生成
       - 分子パラメータの設定
    
    2. 基底系構築 (_create_basis_system)  
       - LinMolBasis: (v,J,M) 量子状態の管理
       - LinMolDipoleMatrix: 遷移双極子行列の計算
       
    3. システム特性計算 (_calculate_system_properties)
       - ハミルトニアン対角化によるエネルギー準位
       - エネルギー差分行列 ωᵢⱼ の構築
       - 緩和効果の組み込み
       
    4. 遷移行列生成 (_generate_transition_matrices)
       - 偏光方向に応じた実効遷移双極子行列
       - 非ゼロ遷移要素のインデックス
       - 振動コヒーレンス射影行列
       
    5. 周波数グリッド設定 (_setup_frequency_grids)
       - 測定波数範囲の角周波数変換
       - 2D計算用の周波数グリッド
    
    PHYSICS BEHIND THE IMPLEMENTATION
    ================================
    
    Energy Difference Matrix (エネルギー差分行列):
    ωᵢⱼ = (Eᵢ - Eⱼ)/ℏ - iΓᵢⱼ
    
    実部: 遷移周波数
    虚部: 緩和による線幅 (homogeneous broadening)
    
    Vibrational Coherence Projection (振動コヒーレンス射影):
    |Δv| ≤ 1 の条件を満たす状態間のみを計算対象とする射影演算子。
    計算量を大幅に削減し、物理的に意味のある遷移のみを考慮。
    
    Effective Transition Dipole (実効遷移双極子):
    μₑff = μₓ·ε̂ₓ + μᵧ·ε̂ᵧ + μᵢ·ε̂ᵢ
    
    偏光ベクトル ε̂ と遷移双極子の内積により、
    実際に観測される遷移強度を計算。
    """
    
    def __init__(self):
        # Core objects
        self._basis: Optional[LinMolBasis] = None
        self._dipole: Optional[LinMolDipoleMatrix] = None
        
        # Physical properties (calculated from molecular parameters)
        self._coherence_decay_rate: Optional[float] = None          # Γ = 1/T₂
        self._number_density: Optional[float] = None                # n = P/(kT)
        self._energy_difference_matrix: Optional[np.ndarray] = None # ωᵢⱼ matrix
        
        # Transition dipole matrices (basis-dependent)
        self._transition_dipole_x: Optional[np.ndarray] = None      # μₓ matrix  
        self._transition_dipole_y: Optional[np.ndarray] = None      # μᵧ matrix
        self._effective_transition_dipole: Optional[np.ndarray] = None      # μₑff
        self._conjugate_transition_dipole: Optional[np.ndarray] = None      # μₑff†
        
        # Calculation optimization data
        self._nonzero_transition_indices: Optional[np.ndarray] = None       # sparse indices
        self._vibrational_coherence_projection: Optional[np.ndarray] = None # |Δv|≤1 projector
        
        # Frequency domain data
        self._angular_frequencies: Optional[np.ndarray] = None              # ω array
        self._frequency_grid_2d: Optional[np.ndarray] = None                # 2D ω grid
        self._inverse_response_denominator: Optional[np.ndarray] = None     # 1/(ω-ωᵢⱼ)
        
        # Parameter storage
        self._spectroscopy_params: Optional[SpectroscopyParameters] = None
        self._molecular_params: Optional[MolecularParameters] = None
    
    def initialize(self, 
                   num_vibrational_levels: int, 
                   num_rotational_levels: int,
                   use_magnetic_quantum_numbers: bool = True,
                   spectroscopy_params: Optional[SpectroscopyParameters] = None,
                   molecular_params: Optional[MolecularParameters] = None) -> None:
        """
        分光計算器を指定されたパラメータで初期化。
        
        この関数は分子系の量子状態空間を定義し、必要な行列要素を
        事前計算します。計算の核心である線形応答関数の構築に
        必要な全ての準備を行います。
        
        Parameters
        ----------
        num_vibrational_levels : int
            含める振動準位数（v = 0, 1, ..., num_vibrational_levels-1）
            通常、基底状態+低励起状態のみで十分（2-5準位）
            
        num_rotational_levels : int  
            含める回転準位数（J = 0, 1, ..., num_rotational_levels-1）
            室温では J~10-20 程度まで有意に占有される
            
        use_magnetic_quantum_numbers : bool
            磁気量子数 M を含めるかの選択。
            True: (v,J,M) 基底 → 偏光依存性を正確に扱う
            False: (v,J) 基底 → 計算量削減、等方的平均
            
        Notes
        -----
        基底サイズの見積もり:
        - use_M=True: N = num_v × Σⱼ(2J+1) ≈ num_v × num_J²  
        - use_M=False: N = num_v × num_J
        
        メモリ使用量: O(N²) なので、大きな系では注意が必要。
        """
        self._spectroscopy_params = spectroscopy_params or SpectroscopyParameters()
        self._molecular_params = molecular_params or MolecularParameters()
        
        # Step 1: 基底状態とハミルトニアンの構築
        self._create_basis_system(
            num_vibrational_levels, 
            num_rotational_levels, 
            use_magnetic_quantum_numbers
        )
        
        # Step 2: 物理量の計算（緩和定数、数密度等）
        self._calculate_system_properties()
        
        # Step 3: 遷移双極子行列の生成と最適化
        self._generate_transition_matrices()
        
        # Step 4: 周波数領域の設定
        self._setup_frequency_grids()
    
    def _create_basis_system(self, nv: int, nj: int, use_m: bool) -> None:
        """
        量子力学的基底系と遷移双極子行列の構築。
        
        LinMolBasisクラスを使用して (v,J,M) 基底状態を生成し、
        LinMolDipoleMatrixクラスで選択則に従った遷移双極子行列を計算。
        
        Physical Background
        ------------------
        基底状態: |v,J,M⟩ = |v⟩ ⊗ |J,M⟩
        - |v⟩: 調和振動子固有状態（またはMorse振動子）
        - |J,M⟩: 剛体回転子固有状態
        
        遷移双極子行列要素の計算では、Wigner 3-j symbols や
        Clebsch-Gordan 係数が内部的に使用されます。
        """
        assert self._molecular_params is not None
        
        # LinMolBasisオブジェクトの生成
        # 内部でエネルギー準位とwave function indices を管理
        self._basis = LinMolBasis(
            V_max=nv - 1,  # Convert from number of levels to max index
            J_max=nj - 1,  # Convert from number of levels to max index
            use_M=use_m,
            omega_rad_phz=self._molecular_params.vibrational_frequency_rad_per_fs,
            delta_omega_rad_phz=self._molecular_params.anharmonicity_correction_rad_per_fs
        )
        
        # LinMolDipoleMatrixオブジェクトの生成  
        # 選択則 Δv=±1, ΔJ=±1, ΔM=0,±1 を自動的に適用
        self._dipole = LinMolDipoleMatrix(
            basis=self._basis,
            mu0=self._molecular_params.transition_dipole_moment,
            potential_type=self._molecular_params.potential_type,
            backend="numpy",
            dense=True
        )
    
    def _calculate_system_properties(self) -> None:
        """
        分子系の基本物理量を計算。
        
        ハミルトニアンの対角化から得られるエネルギー準位を用いて、
        遷移周波数とコヒーレンス緩和を含むエネルギー差分行列を構築。
        
        Physics of Energy Difference Matrix
        -----------------------------------
        ωᵢⱼ = (Eᵢ - Eⱼ)/ℏ - iΓᵢⱼ
        
        実部: 遷移角周波数 [rad/s]
          - 正値: 吸収遷移（i > j）  
          - 負値: 発光遷移（i < j）
          
        虚部: 均質広がり [rad/s]
          - コヒーレンス緩和時間 T₂ に反比例
          - ローレンツ型線形の半値幅
        
        Number Density Calculation
        --------------------------
        理想気体の状態方程式: PV = NkT
        数密度: n = N/V = P/(kT)
        
        これは Beer-Lambert則における吸収係数の計算で必要。
        """
        if not self._basis or not self._spectroscopy_params:
            raise RuntimeError("System not properly initialized")
        
        # コヒーレンス減衰定数の計算: Γ = 1/T₂
        self._coherence_decay_rate = 1.0 / (
            self._spectroscopy_params.coherence_relaxation_time_ps * 1e-12
        )
        
        # 理想気体状態方程式から数密度を計算
        self._number_density = (
            self._spectroscopy_params.pressure_Pa / 
            (kb * self._spectroscopy_params.temperature_K)
        )
        
        # ハミルトニアン行列の生成と対角化
        assert self._molecular_params is not None
        hamiltonian = self._basis.generate_H0(
            omega_rad_phz=self._molecular_params.vibrational_frequency_rad_per_fs,
            delta_omega_rad_phz=self._molecular_params.anharmonicity_correction_rad_per_fs,
            B_rad_phz=self._molecular_params.rotational_constant_rad_per_fs,
            alpha_rad_phz=self._molecular_params.vibration_rotation_coupling_rad_per_fs
        )
        
        # 対角要素（固有エネルギー）の抽出
        energies = np.diag(hamiltonian)
        num_levels = len(energies)
        
        # エネルギー差分行列の構築: (Eᵢ - Eⱼ)/ℏ
        energy_matrix = np.tile(energies, (num_levels, 1))
        energy_diff_real = (energy_matrix - energy_matrix.T) / h_dirac
        
        # 複素エネルギー差分行列: 実部=遷移周波数, 虚部=線幅
        self._energy_difference_matrix = energy_diff_real - 1j * self._coherence_decay_rate
    
    def _generate_transition_matrices(self) -> None:
        """
        偏光を考慮した実効遷移双極子行列の生成。
        
        LinMolDipoleMatrixから得られる空間固定座標系での遷移双極子
        行列要素を、実験で使用する偏光方向に投影します。
        
        Polarization Physics
        --------------------
        観測される遷移強度は偏光方向と遷移双極子の内積で決まります：
        
        I ∝ |ε̂ · μᵢⱼ|²
        
        ここで：
        - ε̂: 偏光ベクトル（レーザー・検出器）
        - μᵢⱼ: 遷移双極子ベクトル
        
        Selection Rules Implementation
        -----------------------------
        LinMolDipoleMatrixクラスが自動的に以下の選択則を適用：
        - Δv = ±1 (振動基本遷移)
        - ΔJ = ±1 (P, R branches)  
        - ΔM = 0, ±1 (π, σ± transitions)
        
        非許可遷移（forbidden transitions）は行列要素が0になります。
        """
        if not self._dipole or not self._spectroscopy_params:
            raise RuntimeError("System not properly initialized")
        
        # 空間固定座標系での遷移双極子行列（複素数化）
        self._transition_dipole_x = self._dipole.mu_x.astype(np.complex128)
        self._transition_dipole_y = self._dipole.mu_y.astype(np.complex128)
        
        # レーザー偏光方向への射影
        assert self._spectroscopy_params.laser_polarization is not None
        pol = self._spectroscopy_params.laser_polarization
        self._effective_transition_dipole = (
            self._transition_dipole_x * complex(pol[0]) +  # type: ignore
            self._transition_dipole_y * complex(pol[1])   # type: ignore
        )
        
        # 検出偏光の処理（レーザーと同じ or 異なる）
        if self._spectroscopy_params.radiation_polarization_same:
            # 同偏光配置: 吸収実験など
            self._conjugate_transition_dipole = self._effective_transition_dipole.conj()
        else:
            # 交差偏光配置: 異方性検出など
            assert self._spectroscopy_params.radiation_polarization is not None
            rad_pol = self._spectroscopy_params.radiation_polarization
            self._conjugate_transition_dipole = (
                self._transition_dipole_x * np.conjugate(rad_pol[0]) +  # type: ignore
                self._transition_dipole_y * np.conjugate(rad_pol[1])   # type: ignore
            )
        
        # 計算効率化: 非ゼロ遷移要素のインデックスを事前計算
        self._nonzero_transition_indices = np.array(
            np.where(self._effective_transition_dipole != 0)
        )
        
        # 振動コヒーレンス射影行列の構築
        self._create_vibrational_projection()
    
    def _create_vibrational_projection(self) -> None:
        """
        振動コヒーレンス射影演算子の構築。
        
        |Δv| ≤ 1 条件を満たす状態間のコヒーレンスのみを計算対象とする
        射影演算子を構築します。これにより計算量を大幅に削減し、
        物理的に意味のある振動遷移のみを考慮します。
        
        Physical Justification
        ----------------------
        基本振動遷移（Δv = ±1）が支配的で、高次遷移（|Δv| ≥ 2）は
        一般に弱く、多くの場合無視できます。この近似により：
        
        1. 計算量削減: O(N²) → O(N×有効遷移数)
        2. 物理的解釈の明確化
        3. 数値安定性の向上
        
        Projection Operator
        ------------------
        P̂ᵥᵢᵦ = Σᵥ,ᵥ' |v,J,M⟩⟨v',J',M'| where |v-v'| ≤ 1
        
        この演算子を密度行列に作用させることで、
        高次振動コヒーレンスを除去します。
        """
        if not self._basis:
            raise RuntimeError("Basis not initialized")
        
        num_levels = self._basis.size()
        self._vibrational_coherence_projection = np.zeros((num_levels, num_levels))
        
        # 各基底状態ペアについて振動量子数差をチェック
        for i in range(num_levels):
            for j in range(num_levels):
                state_i = self._basis.get_state(i)  # [v_i, J_i, M_i] or [v_i, J_i]
                state_j = self._basis.get_state(j)  # [v_j, J_j, M_j] or [v_j, J_j]
                
                v_i, v_j = state_i[0], state_j[0]
                
                # 振動量子数差の条件チェック
                if abs(v_i - v_j) <= 1:
                    self._vibrational_coherence_projection[i, j] = 1.0
    
    def _setup_frequency_grids(self) -> None:
        """
        測定周波数範囲の設定と2D計算用グリッドの構築。
        
        波数 [cm⁻¹] から角周波数 [rad/s] への変換を行い、
        高速計算のための2Dグリッドを準備します。
        
        Unit Conversion
        --------------
        波数 k [cm⁻¹] → 角周波数 ω [rad/s]:
        ω = 2πck = 2π × (2.998×10¹⁰ cm/s) × k
        
        2D Calculation Grid
        ------------------
        ωグリッド（測定周波数）と遷移周波数の全組み合わせを
        事前に計算することで、線形応答関数の計算を高速化。
        
        メモリ使用量: O(N_freq × N_transitions)
        計算量削減: O(N_freq × N_transitions) → O(N_transitions)
        """
        if not self._spectroscopy_params:
            raise RuntimeError("Parameters not initialized")
        
        # 波数から角周波数への変換
        assert self._spectroscopy_params.wavenumbers_cm is not None
        wavenumbers = self._spectroscopy_params.wavenumbers_cm
        self._angular_frequencies = 2 * np.pi * c * 1e2 * wavenumbers  # type: ignore
        
        # 2D計算モードの場合の事前計算
        if self._spectroscopy_params.enable_2d_calculation:
            # 周波数グリッドの準備: [N_freq, N_transitions]
            omega_reshaped = self._angular_frequencies.reshape((-1, 1))  # type: ignore
            assert self._nonzero_transition_indices is not None
            num_transitions = self._nonzero_transition_indices.shape[1]  # type: ignore
            self._frequency_grid_2d = omega_reshaped @ np.ones((1, num_transitions))
            
            # 線形応答分母の事前計算: 1/(ω - ωᵢⱼ + iΓᵢⱼ)
            transition_indices = tuple(self._nonzero_transition_indices)  # type: ignore
            assert self._energy_difference_matrix is not None
            self._inverse_response_denominator = 1.0 / (
                1j * (self._frequency_grid_2d + 
                      self._energy_difference_matrix[transition_indices])  # type: ignore
            )
    
    @property
    def basis(self) -> Optional[LinMolBasis]:
        """量子状態基底オブジェクトへのアクセス。"""
        return self._basis
    
    @property
    def dipole_matrix(self) -> Optional[LinMolDipoleMatrix]:
        """遷移双極子行列オブジェクトへのアクセス。"""
        return self._dipole
    
    @property
    def system_size(self) -> int:
        """量子状態空間の次元数。"""
        return self._basis.size() if self._basis else 0
    
    @property
    def is_initialized(self) -> bool:
        """計算器が正常に初期化されているかの確認。"""
        return (self._basis is not None and 
                self._dipole is not None and
                self._energy_difference_matrix is not None)
    
    def get_calculation_data(self) -> dict:
        """
        スペクトル計算に必要な全データの取得。
        
        この関数は計算済みの行列要素と物理パラメータを
        辞書形式で返し、スペクトル計算関数群で使用されます。
        
        Returns
        -------
        dict
            計算に必要な全データを含む辞書:
            - 行列要素（エネルギー差分、遷移双極子）
            - 物理パラメータ（温度、密度等）
            - 計算効率化データ（インデックス、射影行列）
        """
        if not self.is_initialized:
            raise RuntimeError("Calculator not initialized. Call initialize() first.")
        
        assert self._spectroscopy_params is not None
        return {
            # Core matrices for linear response calculation
            'energy_difference_matrix': self._energy_difference_matrix,
            'effective_transition_dipole': self._effective_transition_dipole,
            'conjugate_transition_dipole': self._conjugate_transition_dipole,
            'nonzero_transition_indices': self._nonzero_transition_indices,
            'vibrational_coherence_projection': self._vibrational_coherence_projection,
            
            # Frequency domain data
            'angular_frequencies': self._angular_frequencies,
            'wavenumbers': self._spectroscopy_params.wavenumbers_cm,
            
            # Physical parameters for spectrum conversion
            'optical_path_length': self._spectroscopy_params.optical_path_length_m,
            'number_density': self._number_density,
            'temperature': self._spectroscopy_params.temperature_K,
            'molecular_mass': self._spectroscopy_params.molecular_mass_kg,
            
            # 2D calculation optimization data
            'is_2d_enabled': self._spectroscopy_params.enable_2d_calculation,
            'frequency_grid_2d': self._frequency_grid_2d,
            'inverse_response_denominator': self._inverse_response_denominator,
        }


# Global instance for backward compatibility
# レガシーAPIとの互換性維持のためのグローバルインスタンス
_global_calculator = LinearResponseCalculator()


def prepare_variables(Nv: int, Nj: int, use_projection_number: bool = True, T2: float = 500,
                      m_mol: float = m_CO2, temp: float = 300, l: float = 1e-3, p: float = 3e4,
                      pol: Optional[List[float]] = None, pol_rad_is_same: bool = True, 
                      pol_rad: Optional[List[float]] = None,
                      Wavenumber: Optional[np.ndarray] = None,
                      make_2d: bool = False, use_wn_v: bool = False,
                      mu0: float = 1.0, omega_rad_phz: float = 1000.0,
                      B_rad_phz: float = 1.0, alpha_rad_phz: float = 0.0,
                      delta_omega_rad_phz: float = 0.0) -> None:
    """
    分光計算用変数の準備（レガシーインターフェース）。
    
    この関数は既存コードとの後方互換性を維持します。
    新しいコードではLinearResponseCalculatorクラスの直接使用を推奨。
    
    Legacy Parameter Mapping
    ------------------------
    Nv, Nj → num_vibrational_levels, num_rotational_levels
    T2 → coherence_relaxation_time_ps  
    m_mol, temp, l, p → SpectroscopyParameters
    mu0, omega_rad_phz, etc. → MolecularParameters
    
    この関数は内部的にグローバルなLinearResponseCalculatorインスタンスを
    使用し、従来のAPI呼び出しを新しいクラスベース実装に橋渡しします。
    """
    if pol is None:
        pol = [1.0, 0.0]
    if pol_rad is None:
        pol_rad = [1.0, 0.0]
    if Wavenumber is None:
        Wavenumber = np.arange(2100, 2400, 0.01)
    
    # レガシーパラメータから新データクラスへの変換
    spectroscopy_params = SpectroscopyParameters(
        coherence_relaxation_time_ps=T2,
        molecular_mass_kg=m_mol,
        temperature_K=temp,
        optical_path_length_m=l,
        pressure_Pa=p,
        laser_polarization=pol,
        radiation_polarization_same=pol_rad_is_same,
        radiation_polarization=pol_rad,
        wavenumbers_cm=Wavenumber,
        enable_2d_calculation=make_2d
    )
    
    molecular_params = MolecularParameters(
        transition_dipole_moment=mu0,
        vibrational_frequency_rad_per_fs=omega_rad_phz,
        rotational_constant_rad_per_fs=B_rad_phz,
        vibration_rotation_coupling_rad_per_fs=alpha_rad_phz,
        anharmonicity_correction_rad_per_fs=delta_omega_rad_phz
    )
    
    # グローバル計算器の初期化
    _global_calculator.initialize(
        Nv, Nj, use_projection_number,
        spectroscopy_params, molecular_params
    )


def get_calculator() -> LinearResponseCalculator:
    """グローバル計算器インスタンスの取得。"""
    return _global_calculator


def get_basis() -> Optional[LinMolBasis]:
    """現在の基底オブジェクトの取得（レガシーインターフェース）。"""
    return _global_calculator.basis


def get_dipole() -> Optional[LinMolDipoleMatrix]:
    """現在の双極子行列オブジェクトの取得（レガシーインターフェース）。"""
    return _global_calculator.dipole_matrix 