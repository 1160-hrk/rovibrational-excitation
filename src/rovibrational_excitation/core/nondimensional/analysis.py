"""
analysis.py
===========
無次元化関連の分析機能を提供するモジュール。
"""
from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

from .scales import NondimensionalizationScales
from .utils import _EV_TO_J

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np


def analyze_regime(scales: NondimensionalizationScales) -> Dict[str, Any]:
    """
    物理レジームの分析
    
    Parameters
    ----------
    scales : NondimensionalizationScales
        スケールファクター
        
    Returns
    -------
    dict
        分析結果
    """
    interaction = scales.interaction_energy
    physical_ratio = scales.physical_coupling_ratio
    if interaction == 0:
        regime = "field_free"
        description = "The interaction generator is inactive."
    elif physical_ratio is None:
        regime = "gapless_driven"
        description = "Driven system with no non-zero field-free spectral span."
    else:
        regime = "unclassified"
        description = (
            "No weak/strong label is assigned without a model-specific threshold."
        )

    return {
        "regime": regime,
        "numerical_coupling_coefficient": scales.lambda_coupling,
        "physical_coupling_ratio": physical_ratio,
        "description": description,
        "energy_scale_eV": scales.E0 / _EV_TO_J,
        "time_scale_fs": scales.t0 * 1e15,
        "reference_energy": {
            "value_J": scales.reference_energy.value,
            "source": scales.reference_energy.source,
            "method": scales.reference_energy.method,
        },
    }


def verify_nondimensional_equation(
    H0_prime: "np.ndarray",
    mu_x_prime: "np.ndarray",
    mu_y_prime: "np.ndarray",
    Efield_prime: "np.ndarray",
    scales: NondimensionalizationScales,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    無次元化された方程式の検証
    
    無次元化後の方程式が正しい形になっているかを確認:
    i d/dτ |ψ⟩ = (H₀' - λ μ' E'(τ)) |ψ⟩
    
    Parameters
    ----------
    H0_prime : np.ndarray
        無次元ハミルトニアン
    mu_x_prime, mu_y_prime : np.ndarray
        無次元双極子行列
    Efield_prime : np.ndarray
        無次元電場
    scales : NondimensionalizationScales
        スケールファクター
    verbose : bool, optional
        詳細情報を表示, デフォルト: True
        
    Returns
    -------
    dict
        検証結果
    """
    import numpy as np
    
    verification = {}
    
    # ① 無次元ハミルトニアンの検証
    if H0_prime.ndim == 2:
        H0_diag = np.diag(H0_prime)
    else:
        H0_diag = H0_prime.copy()
    
    # エネルギー差が O(1) オーダーになっているか
    energy_diffs_prime = []
    for i in range(len(H0_diag)):
        for j in range(i+1, len(H0_diag)):
            energy_diffs_prime.append(abs(H0_diag[i] - H0_diag[j]))
    
    max_energy_diff_prime = max(energy_diffs_prime) if energy_diffs_prime else 0
    verification["H0_max_diff_dimensionless"] = max_energy_diff_prime
    verification["H0_order_unity"] = 0.1 <= max_energy_diff_prime <= 10.0
    
    # ② 無次元双極子行列の検証
    all_mu_prime = []
    for mu_prime in [mu_x_prime, mu_y_prime]:
        if mu_prime.ndim == 2:
            for i in range(mu_prime.shape[0]):
                for j in range(mu_prime.shape[1]):
                    if i != j and abs(mu_prime[i,j]) > 0:
                        all_mu_prime.append(abs(mu_prime[i,j]))
        else:
            all_mu_prime.extend([abs(x) for x in mu_prime if abs(x) > 0])
    
    max_mu_prime = max(all_mu_prime) if all_mu_prime else 0
    verification["mu_max_dimensionless"] = max_mu_prime
    verification["mu_order_unity"] = 0.1 <= max_mu_prime <= 10.0
    
    # ③ 無次元電場の検証
    max_efield_prime = np.max(np.abs(Efield_prime))
    verification["Efield_max_dimensionless"] = max_efield_prime
    verification["Efield_order_unity"] = 0.1 <= max_efield_prime <= 10.0
    
    # ④ 結合強度 λ の検証
    verification["lambda_coupling"] = scales.lambda_coupling
    verification["lambda_reasonable"] = 0.001 <= scales.lambda_coupling <= 100.0
    
    # ⑤ 全体的な検証
    all_checks = [
        verification["H0_order_unity"],
        verification["mu_order_unity"], 
        verification["Efield_order_unity"],
        verification["lambda_reasonable"]
    ]
    verification["overall_valid"] = all(all_checks)
    
    if verbose:
        print("🔍 Verifying nondimensional equation form...")
        print(f"   H₀' max difference: {max_energy_diff_prime:.3f} (should be O(1))")
        print(f"   μ' max element: {max_mu_prime:.3f} (should be O(1))")
        print(f"   E' max amplitude: {max_efield_prime:.3f} (should be O(1))")
        print(f"   λ coupling strength: {scales.lambda_coupling:.3f}")
        
        if verification["overall_valid"]:
            print("✅ Nondimensional equation verified successfully!")
        else:
            print("⚠️  Warning: Some nondimensional quantities are not O(1)")
            if not verification["H0_order_unity"]:
                print("    - H₀' is not O(1), consider different energy scale")
            if not verification["mu_order_unity"]:
                print("    - μ' is not O(1), consider different dipole scale")
            if not verification["Efield_order_unity"]:
                print("    - E' is not O(1), consider different field scale")
    
    return verification


def optimize_timestep_for_coupling(
    scales: NondimensionalizationScales,
    target_accuracy: str = "standard",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    結合強度に最適化された時間ステップを提案
    
    Parameters
    ----------
    scales : NondimensionalizationScales
        無次元化スケールファクター
    target_accuracy : str, optional
        目標精度 ("fast", "standard", "high", "ultrahigh"), デフォルト: "standard"
    verbose : bool, optional
        詳細情報を表示するかどうか, デフォルト: True
        
    Returns
    -------
    dict
        最適化された時間ステップと分析結果
    """
    import numpy as np
    
    λ = scales.lambda_coupling
    
    # 精度レベルに応じた安全係数の設定
    accuracy_settings = {
        "fast": {"safety_factor": 0.5, "method": "stability", "description": "高速計算重視"},
        "standard": {"safety_factor": 0.1, "method": "adaptive", "description": "標準精度"},
        "high": {"safety_factor": 0.05, "method": "rabi", "description": "高精度"},
        "ultrahigh": {"safety_factor": 0.01, "method": "rabi", "description": "超高精度"}
    }
    
    if target_accuracy not in accuracy_settings:
        raise ValueError(f"target_accuracy must be one of {list(accuracy_settings.keys())}")
    
    settings = accuracy_settings[target_accuracy]
    
    # 推奨時間ステップの計算
    dt_dim = scales.get_recommended_timestep_dimensionless(
        safety_factor=settings["safety_factor"],
        method=settings["method"]
    )
    dt_fs = scales.get_recommended_timestep_fs(
        safety_factor=settings["safety_factor"],
        method=settings["method"]
    )
    
    # 詳細分析
    analysis = scales.analyze_timestep_requirements()
    
    # 結果のまとめ
    result = {
        "target_accuracy": target_accuracy,
        "settings": settings,
        "lambda_coupling": λ,
        "recommended_dt_fs": dt_fs,
        "recommended_dt_dimensionless": dt_dim,
        "regime": analysis["regime"],
        "rabi_period_fs": analysis.get("rabi_period_fs", np.inf),
        "computational_cost_estimate": 1.0 / dt_dim,  # 相対的計算コスト
        "all_methods": analysis["recommendations"]
    }
    
    if verbose:
        print(f"\n⚡ 結合強度最適化時間ステップ分析")
        print(f"   λ = {λ:.3f} ({analysis['regime']})")
        print(f"   目標精度: {target_accuracy} ({settings['description']})")
        print(f"   推奨時間ステップ: {dt_fs:.3f} fs ({dt_dim:.6f} 無次元)")
        print(f"   計算コスト (相対): {result['computational_cost_estimate']:.1f}x")
        
        rabi_period = result.get("rabi_period_fs", np.inf)
        if (rabi_period != np.inf and not np.isinf(rabi_period) and 
            dt_fs is not None and dt_fs > 0):
            print(f"   Rabi周期: {rabi_period:.3f} fs")
            print(f"   Rabi周期あたりステップ数: {rabi_period/dt_fs:.1f}")
        
        print(f"   アドバイス: {analysis['advice']}")
    
    return result


def calculate_nondimensionalization_scales_strict(
    H0: "np.ndarray",
    mu_x: "np.ndarray",
    mu_y: "np.ndarray",
    efield: Any,
    *,
    hbar: float = 1.054571817e-34,
    verbose: bool = True
) -> NondimensionalizationScales:
    """
    数学的に厳密な無次元化スケールファクター計算
    
    LaTeX式に基づく厳密な定義:
    - E₀ = max_{n≠m} |H₀,ₙₙ - H₀,ₘₘ|
    - t₀ = ℏ/E₀  
    - E_field,₀ = max_t |E(t)|
    - μ₀ = max_{n≠m} |μₙₘ|
    - λ = E_field,₀ * μ₀ / E₀
    
    Parameters
    ----------
    H0 : np.ndarray
        ハミルトニアン行列（対角成分）[J]
    mu_x, mu_y : np.ndarray  
        双極子モーメント行列 [C·m]
    efield : ElectricField
        電場オブジェクト [V/m]
    hbar : float, optional
        プランク定数 [J·s], デフォルト: ℏ
    verbose : bool, optional
        詳細情報を表示, デフォルト: True
        
    Returns
    -------
    NondimensionalizationScales
        数学的に厳密な無次元化スケール
    """
    import numpy as np
    
    if verbose:
        print("🔬 Calculating nondimensionalization scales with strict mathematical definitions...")
    
    # ① エネルギースケール E₀ = max_{n≠m} |H₀,ₙₙ - H₀,ₘₘ|
    if H0.ndim == 2:
        # 対角行列の場合
        diagonal_elements = np.diag(H0)
    else:
        diagonal_elements = H0.copy()
    
    # すべてのペア (n,m) with n≠m の対角成分差を計算
    n_states = len(diagonal_elements)
    energy_differences = []
    
    for n in range(n_states):
        for m in range(n_states):
            if n != m:  # n≠m の条件
                diff = abs(diagonal_elements[n] - diagonal_elements[m])
                energy_differences.append(diff)
    
    if len(energy_differences) == 0:
        # 状態が1つだけの場合
        E0 = diagonal_elements[0] if len(diagonal_elements) > 0 else _EV_TO_J
        if verbose:
            print("   ⚠️  Warning: Only one state found, using E₀ = H₀,₀₀")
    else:
        E0 = max(energy_differences)
    
    if verbose:
        print(f"   E₀ = max_{{n≠m}} |H₀,ₙₙ - H₀,ₘₘ| = {E0:.6e} J")
        print(f"      = {E0/_EV_TO_J:.3f} eV")
        print(f"      Found {len(energy_differences)} energy differences")
    
    # ② 時間スケール t₀ = ℏ/E₀
    t0 = hbar / E0
    if verbose:
        print(f"   t₀ = ℏ/E₀ = {t0:.6e} s = {t0*1e15:.3f} fs")
    
    # ③ 電場スケール E_field,₀ = max_t |E(t)|
    efield_array = efield.get_Efield()  # [V/m]
    Efield0 = np.max(np.abs(efield_array))
    
    if Efield0 == 0:
        Efield0 = 1e8  # 1 MV/cm デフォルト
        if verbose:
            print("   ⚠️  Warning: Zero electric field, using default 1 MV/cm")
    
    if verbose:
        print(f"   E_field,₀ = max_t |E(t)| = {Efield0:.6e} V/m")
        print(f"             = {Efield0/1e8:.3f} MV/cm")
    
    # ④ 双極子モーメントスケール μ₀ = max_{n≠m} |μₙₘ|
    # mu_x と mu_y を結合して全体の双極子行列要素を考える
    all_mu_elements = []
    
    for mu_matrix in [mu_x, mu_y]:
        if mu_matrix.ndim == 2:
            # 行列の場合、非対角成分のみを抽出
            for n in range(mu_matrix.shape[0]):
                for m in range(mu_matrix.shape[1]):
                    if n != m:  # n≠m の条件
                        element = abs(mu_matrix[n, m])
                        if element > 0:  # ゼロでない要素のみ
                            all_mu_elements.append(element)
        elif mu_matrix.ndim == 1:
            # 1次元配列の場合（非対角成分として扱う）
            for element in mu_matrix:
                if abs(element) > 0:
                    all_mu_elements.append(abs(element))
    
    if len(all_mu_elements) == 0:
        mu0 = 3.33564e-30  # 1 D デフォルト
        if verbose:
            print("   ⚠️  Warning: No non-zero off-diagonal dipole elements, using 1 D")
    else:
        mu0 = max(all_mu_elements)
    
    if verbose:
        print(f"   μ₀ = max_{{n≠m}} |μₙₘ| = {mu0:.6e} C·m")
        print(f"      = {mu0/3.33564e-30:.3f} D")
        print(f"      Found {len(all_mu_elements)} non-zero dipole elements")
    
    # ⑤ 結合強度パラメータ λ = E_field,₀ * μ₀ / E₀
    lambda_coupling = (Efield0 * mu0) / E0
    
    if verbose:
        print(f"   λ = E_field,₀ * μ₀ / E₀ = {lambda_coupling:.6f}")
        
        # 物理的解釈
        if lambda_coupling < 0.1:
            regime = "weak coupling (λ << 1)"
            interpretation = "摂動論的取り扱いが有効"
        elif lambda_coupling < 1.0:
            regime = "intermediate coupling (λ ~ 1)"
            interpretation = "非摂動効果が現れ始める"
        else:
            regime = "strong coupling (λ >> 1)"
            interpretation = "Rabi振動など非線形効果が顕著"
        
        print(f"   Physical regime: {regime}")
        print(f"   Interpretation: {interpretation}")
    
    # スケール情報を作成
    scales = NondimensionalizationScales(
        E0=E0,
        mu0=mu0,
        Efield0=Efield0,
        t0=t0,
        lambda_coupling=lambda_coupling,
    )
    
    if verbose:
        print("✅ Strict nondimensionalization scales calculated successfully!")
    
    return scales


class NondimensionalAnalyzer:
    """分析機能を提供するクラス"""

    @staticmethod
    def analyze_regime(scales: NondimensionalizationScales) -> Dict[str, Any]:
        """λ 値に基づく物理レジーム判定"""
        return analyze_regime(scales)

    @staticmethod
    def verify_equation(
        H0_prime: "np.ndarray",
        mu_x_prime: "np.ndarray",
        mu_y_prime: "np.ndarray",
        Efield_prime: "np.ndarray",
        scales: NondimensionalizationScales,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """無次元方程式の整合性検証"""
        return verify_nondimensional_equation(
            H0_prime,
            mu_x_prime,
            mu_y_prime,
            Efield_prime,
            scales,
            verbose=verbose,
        )

    @staticmethod
    def optimize_timestep_for_coupling(
        scales: NondimensionalizationScales, 
        target_accuracy: str = "standard", 
        verbose: bool = True
    ) -> Dict[str, Any]:
        """結合強度に最適化された時間ステップを提案"""
        return optimize_timestep_for_coupling(
            scales, target_accuracy=target_accuracy, verbose=verbose
        )

    @staticmethod
    def calculate_strict_scales(
        H0: "np.ndarray",
        mu_x: "np.ndarray",
        mu_y: "np.ndarray",
        efield: Any,
        **kwargs: Any
    ) -> NondimensionalizationScales:
        """数学的に厳密な無次元化スケールファクター計算"""
        return calculate_nondimensionalization_scales_strict(
            H0, mu_x, mu_y, efield, **kwargs
        ) 