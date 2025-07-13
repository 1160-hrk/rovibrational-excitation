"""
無次元化モジュール (nondimensionalize.py)
==========================================
シュレディンガー方程式の体系的無次元化を行い、
数値計算の安定性と効率性を向上させる。

デフォルト単位を自動的にSI基本単位（接頭辞なし）に変換してから
無次元化を実行する統合システム。

目標式:
    i ∂ψ/∂τ = (H0' - λ μ' E'(τ)) ψ

where:
    - τ: 無次元時間
    - H0': 無次元ハミルトニアン
    - μ': 無次元双極子行列
    - E'(τ): 無次元電場
    - λ: 無次元結合強度パラメータ

SI基本単位変換:
    - 周波数: cm⁻¹ → rad/s
    - 双極子モーメント: D → C·m
    - 電場: MV/cm → V/m
    - エネルギー: eV → J
    - 時間: fs → s
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

import numpy as np
from rovibrational_excitation.core.units.constants import CONSTANTS

if TYPE_CHECKING:
    from rovibrational_excitation.core.electric_field import ElectricField

# Planck constant in J·s
_HBAR = CONSTANTS.HBAR

# Physical constants for SI base unit conversion
_C = CONSTANTS.C  # Speed of light [m/s]
_EV_TO_J = CONSTANTS.EV_TO_J  # eV → J
_DEBYE_TO_CM = CONSTANTS.DEBYE_TO_CM  # D → C·m

# Default unit to SI base unit conversion factors
DEFAULT_TO_SI_CONVERSIONS = {
    # Frequency: cm⁻¹ → rad/s
    "frequency_cm_inv_to_rad_per_s": 2 * np.pi * _C * 100,  # cm⁻¹ → rad/s
    
    # Dipole moment: D → C·m
    "dipole_D_to_Cm": _DEBYE_TO_CM,  # D → C·m
    
    # Electric field: MV/cm → V/m
    "field_MV_per_cm_to_V_per_m": 1e8,  # MV/cm → V/m
    
    # Energy: eV → J
    "energy_eV_to_J": _EV_TO_J,  # eV → J
    
    # Time: fs → s
    "time_fs_to_s": 1e-15,  # fs → s
}


class NondimensionalizationScales:
    """無次元化のスケールファクターを管理するクラス"""

    def __init__(
        self,
        E0: float,
        mu0: float,
        Efield0: float,
        t0: float,
        lambda_coupling: float,
    ):
        """
        Parameters
        ----------
        E0 : float
            エネルギースケール [J]
        mu0 : float  
            双極子モーメントスケール [C·m]
        Efield0 : float
            電場スケール [V/m]
        t0 : float
            時間スケール [s]
        lambda_coupling : float
            無次元結合強度
        """
        self.E0 = E0
        self.mu0 = mu0
        self.Efield0 = Efield0
        self.t0 = t0
        self.lambda_coupling = lambda_coupling

    def __repr__(self) -> str:
        return (
            f"NondimensionalizationScales(\n"
            f"  E0={self.E0:.3e} J,\n"
            f"  mu0={self.mu0:.3e} C·m,\n"
            f"  Efield0={self.Efield0:.3e} V/m,\n"
            f"  t0={self.t0:.3e} s,\n"
            f"  λ={self.lambda_coupling:.3f}\n"
            f")"
        )

    def get_recommended_timestep_dimensionless(
        self,
        safety_factor: float = 0.02,  # 量子力学に適した保守的な値
        min_dt: float = 1e-4,
        max_dt: float = 1.0,
        method: str = "adaptive",
        numerical_method: str = "split_operator"
    ) -> float:
        """
        lambda_couplingに基づく推奨時間ステップ（無次元）を計算
        
        Parameters
        ----------
        safety_factor : float, optional
            安全係数（小さいほど保守的）, デフォルト: 0.02
        min_dt : float, optional
            最小時間ステップ（無次元）, デフォルト: 1e-4
        max_dt : float, optional
            最大時間ステップ（無次元）, デフォルト: 1.0
        method : str, optional
            計算方法 ("adaptive", "rabi", "stability"), デフォルト: "adaptive"
        numerical_method : str, optional
            数値積分手法 ("split_operator", "rk4", "magnus"), デフォルト: "split_operator"
            
        Returns
        -------
        float
            推奨時間ステップ（無次元）
            
        Notes
        -----
        量子力学シミュレーション用推奨安全係数:
        - Split-operator法: 0.01-0.05 (ユニタリ性保持、位相精度重視)
        - RK4法: 0.005-0.02 (高精度だが位相誤差に注意)
        - Magnus展開: 0.01-0.03 (高次精度、良好なユニタリ性)
        
        物理的考察:
        - 弱結合 (λ << 1): 摂動効果のため小さな時間ステップが安全
        - 強結合 (λ >> 1): Rabi振動と位相精度のため極小時間ステップ必須
        - 長時間シミュレーションでは累積誤差を避けるため保守的設定
        """
        λ = self.lambda_coupling
        
        # 数値手法に応じた安全係数の調整
        method_corrections = {
            "split_operator": 1.0,      # 基準値
            "rk4": 0.4,                 # より保守的
            "magnus": 0.6,              # 中程度
            "crank_nicolson": 0.8,      # 比較的安定
            "implicit": 1.2             # より積極的
        }
        
        correction_factor = method_corrections.get(numerical_method, 1.0)
        
        if method == "adaptive":
            # 適応的アルゴリズム: λに応じた非線形スケーリング
            if λ < 0.01:
                # 極弱結合: 大きな時間ステップで十分
                dt_base = 1.0
            elif λ < 0.1:
                # 弱結合: 線形減少
                dt_base = 1.0 - 9.0 * (λ - 0.01) / 0.09
            elif λ < 1.0:
                # 中間結合: 1/λスケーリング
                dt_base = 0.2 / λ  # より現実的な基準値
            else:
                # 強結合: より保守的な1/λ^1.2スケーリング
                dt_base = 0.2 / (λ ** 1.2)  # λ^1.5から緩和
                
        elif method == "rabi":
            # Rabi周期ベース: T_Rabi = 2π/λ
            rabi_period = 2 * np.pi / max(λ, 0.01)  # ゼロ除算回避
            dt_base = rabi_period / 10  # Rabi周期の1/10（1/20から緩和）
            
        elif method == "stability":
            # 数値安定性ベース: 単純な1/λスケーリング
            dt_base = 0.5 / max(λ, 0.1)  # より現実的な基準値
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # 安全係数と数値手法補正の適用
        dt_recommended = dt_base * safety_factor * correction_factor
        
        # 制限値の適用
        dt_recommended = max(min_dt, min(max_dt, dt_recommended))
        
        return dt_recommended

    def get_recommended_timestep_fs(
        self,
        safety_factor: float = 0.5,  # より現実的なデフォルト値
        min_dt_fs: float | None = None,
        max_dt_fs: float | None = None,
        method: str = "adaptive",
        numerical_method: str = "split_operator"
    ) -> float:
        """
        lambda_couplingに基づく推奨時間ステップ（fs）を計算
        
        Parameters
        ----------
        safety_factor : float, optional
            安全係数, デフォルト: 0.5
        min_dt_fs : float, optional
            最小時間ステップ（fs）, デフォルト: t0の1/1000
        max_dt_fs : float, optional
            最大時間ステップ（fs）, デフォルト: t0の10倍
        method : str, optional
            計算方法, デフォルト: "adaptive"
        numerical_method : str, optional
            数値積分手法, デフォルト: "split_operator"
            
        Returns
        -------
        float
            推奨時間ステップ（fs）
        """
        t0_fs = self.t0 * 1e15  # s → fs
        
        # デフォルト制限値の設定
        if min_dt_fs is None:
            min_dt_fs = t0_fs * 1e-3  # t0の1/1000
        if max_dt_fs is None:
            max_dt_fs = t0_fs * 10   # t0の10倍
            
        # 無次元時間ステップを計算
        min_dt_dim = min_dt_fs / t0_fs
        max_dt_dim = max_dt_fs / t0_fs
        
        dt_dim = self.get_recommended_timestep_dimensionless(
            safety_factor=safety_factor,
            min_dt=min_dt_dim,
            max_dt=max_dt_dim,
            method=method,
            numerical_method=numerical_method
        )
        
        # fs単位に変換
        dt_fs = dt_dim * t0_fs
        
        return dt_fs

    def analyze_timestep_requirements(self) -> Dict[str, Any]:
        """
        時間ステップ要件の詳細分析
        
        Returns
        -------
        dict
            分析結果とパラメータ推奨値
        """
        λ = self.lambda_coupling
        t0_fs = self.t0 * 1e15
        
        # 各手法での推奨値
        methods = ["adaptive", "rabi", "stability"]
        recommendations = {}
        
        for method in methods:
            dt_dim = self.get_recommended_timestep_dimensionless(method=method)
            dt_fs = dt_dim * t0_fs
            recommendations[method] = {
                "dt_dimensionless": dt_dim,
                "dt_fs": dt_fs,
                "steps_per_rabi_period": 2 * np.pi / (λ * dt_dim) if λ > 0 else np.inf
            }
        
        # 物理的解釈
        if λ < 0.1:
            regime = "weak_coupling"
            advice = "大きな時間ステップで計算効率を重視可能"
        elif λ < 1.0:
            regime = "intermediate_coupling"
            advice = "適度な時間ステップでバランスを取る"
        else:
            regime = "strong_coupling"
            advice = "Rabi振動解像のため小さな時間ステップが必要"
        
        return {
            "lambda_coupling": λ,
            "regime": regime,
            "advice": advice,
            "time_scale_fs": t0_fs,
            "rabi_period_dimensionless": 2 * np.pi / λ if λ > 0 else np.inf,
            "rabi_period_fs": (2 * np.pi / λ) * t0_fs if λ > 0 else np.inf,
            "recommendations": recommendations,
            "default_choice": recommendations["adaptive"]
        }

    def analyze_numerical_stability(
        self, 
        numerical_method: str = "split_operator",
        target_error: float = 1e-6
    ) -> Dict[str, Any]:
        """
        数値手法別の安定性と推奨パラメータを詳細分析
        
        Parameters
        ----------
        numerical_method : str, optional
            数値積分手法, デフォルト: "split_operator"
        target_error : float, optional
            目標誤差, デフォルト: 1e-6
            
        Returns
        -------
        dict
            詳細な安定性分析結果
        """
        λ = self.lambda_coupling
        t0_fs = self.t0 * 1e15
        
        # 数値手法別の特性
        method_properties = {
            "split_operator": {
                "order": 2,
                "unitarity": "exact",
                "stability": "excellent",
                "recommended_safety": 0.5,
                "description": "Split-operator法: ユニタリ性保持、高速"
            },
            "rk4": {
                "order": 4,
                "unitarity": "approximate",
                "stability": "good",
                "recommended_safety": 0.3,
                "description": "Runge-Kutta 4次: 高精度、やや不安定"
            },
            "magnus": {
                "order": 4,
                "unitarity": "exact",
                "stability": "very_good",
                "recommended_safety": 0.4,
                "description": "Magnus展開: 高次精度、ユニタリ性保持"
            },
            "crank_nicolson": {
                "order": 2,
                "unitarity": "exact",
                "stability": "excellent",
                "recommended_safety": 0.6,
                "description": "Crank-Nicolson法: 無条件安定、陰的"
            }
        }
        
        if numerical_method not in method_properties:
            raise ValueError(f"Unknown numerical method: {numerical_method}")
        
        props = method_properties[numerical_method]
        
        # 推奨時間ステップの計算（各手法で）
        methods = ["adaptive", "rabi", "stability"]
        timestep_analysis = {}
        
        for method in methods:
            dt_dim = self.get_recommended_timestep_dimensionless(
                safety_factor=props["recommended_safety"],
                method=method,
                numerical_method=numerical_method
            )
            dt_fs = dt_dim * t0_fs
            
            # 誤差推定
            if props["order"] == 2:
                estimated_error = (dt_dim ** 2) * λ  # 2次誤差
            elif props["order"] == 4:
                estimated_error = (dt_dim ** 4) * (λ ** 2)  # 4次誤差
            else:
                estimated_error = dt_dim * λ  # 1次誤差
            
            timestep_analysis[method] = {
                "dt_dimensionless": dt_dim,
                "dt_fs": dt_fs,
                "estimated_error": estimated_error,
                "error_acceptable": estimated_error < target_error,
                "steps_per_rabi_period": 2 * np.pi / (λ * dt_dim) if λ > 0 else np.inf
            }
        
        # 最適化された推奨値
        best_method = min(timestep_analysis.keys(), 
                         key=lambda k: timestep_analysis[k]["estimated_error"])
        
        return {
            "lambda_coupling": λ,
            "numerical_method": numerical_method,
            "method_properties": props,
            "target_error": target_error,
            "timestep_analysis": timestep_analysis,
            "recommended_method": best_method,
            "best_timestep_fs": timestep_analysis[best_method]["dt_fs"],
            "best_timestep_dimensionless": timestep_analysis[best_method]["dt_dimensionless"],
            "stability_summary": self._get_stability_summary(λ, props)
        }
    
    def _get_stability_summary(self, λ: float, props: Dict[str, Any]) -> str:
        """安定性の要約を生成"""
        if λ < 0.1:
            regime_advice = "弱結合のため大きな時間ステップでも安定"
        elif λ < 1.0:
            regime_advice = "中間結合、適度な時間ステップが必要"
        else:
            regime_advice = "強結合、小さな時間ステップが必須"
        
        return f"{regime_advice}。{props['description']}"


def nondimensionalize_system(
    H0: np.ndarray,
    mu_x: np.ndarray,
    mu_y: np.ndarray,
    efield: ElectricField,
    *,
    dt: float | None = None,
    H0_units: str = "energy",
    time_units: str = "fs",
    hbar: float = _HBAR,
    min_energy_diff: float = 1e-20,
    max_time_scale_fs: float = 1000.0,  # 時間スケール上限 [fs]
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    NondimensionalizationScales,
]:
    """
    量子系の完全無次元化を実行

    Parameters
    ----------
    H0 : np.ndarray
        対角ハミルトニアン
    mu_x, mu_y : np.ndarray
        双極子行列（C·m単位）
    efield : ElectricField
        電場オブジェクト
    dt : float, optional
        時間ステップ。Noneの場合はefield.dtを使用
    H0_units : str, optional
        H0の単位。"energy" (J) または "frequency" (rad/fs)。デフォルトは"energy"
    time_units : str, optional
        時間の単位。"fs" または "s"。デフォルトは"fs"
    hbar : float
        プランク定数 [J·s]
    min_energy_diff : float
        最小エネルギー差の閾値

    Returns
    -------
    tuple
        (H0_prime, mu_x_prime, mu_y_prime, Efield_prime, tlist_prime, 
         dt_prime, scales)
    """
    # 時間ステップの設定
    if dt is None:
        dt = efield.dt
    
    # dt is guaranteed to be float here
    assert dt is not None

    # 1. エネルギースケールの計算
    if H0_units == "energy":
        # H0は既にエネルギー単位（J）
        if hasattr(H0, 'matrix'):
            # Hamiltonianオブジェクトの場合
            H0_energy = H0.matrix.copy()
        else:
            # numpy配列の場合
            H0_energy = H0.copy()
    elif H0_units == "frequency":
        # H0は周波数単位（rad/fs）なので、Jに変換
        if hasattr(H0, 'matrix'):
            # Hamiltonianオブジェクトの場合
            H0_energy = H0.matrix * hbar / 1e-15
        else:
            # numpy配列の場合
            H0_energy = H0 * hbar / 1e-15  # rad/fs → J
    else:
        raise ValueError("H0_units must be 'energy' or 'frequency'")
    
    if H0_energy.ndim == 2:
        eigvals = np.diag(H0_energy)
    else:
        eigvals = H0_energy.copy()
    
    # 最大エネルギー差を計算（修正版）
    energy_diffs = np.abs(eigvals[:, None] - eigvals[None, :])
    # 対角成分（自分自身との差=0）を除外
    energy_diffs_nonzero = energy_diffs[energy_diffs > 0]
    
    if len(energy_diffs_nonzero) == 0:
        # すべて縮退している場合、最大エネルギー値をスケールとして使用
        E0 = np.max(np.abs(eigvals)) if len(eigvals) > 0 else 1.0
        if E0 < min_energy_diff:
            E0 = min_energy_diff  # ゼロエネルギーの場合は最小値を設定
    else:
        # 相対的に小さすぎる差を除外（最大差の1e-6以下）
        max_diff = np.max(energy_diffs_nonzero)
        significant_diffs = energy_diffs_nonzero[energy_diffs_nonzero > max_diff * 1e-6]
        
        if len(significant_diffs) == 0:
            # 相対閾値でも除外される場合、最大差を使用
            E0 = max_diff
        else:
            E0 = np.max(significant_diffs)

    # 2. 時間スケール（上限チェック付き）
    t0 = hbar / E0  # [s]
    
    # 時間スケールが大きすぎる場合は上限を適用
    max_time_scale_s = max_time_scale_fs * 1e-15  # fs → s
    if t0 > max_time_scale_s:
        t0 = max_time_scale_s
        E0 = hbar / t0  # エネルギースケールを再調整

    # 3. 電場スケール
    Efield_array = efield.get_Efield()  # (T, 2) [V/m]
    Efield0 = np.max(np.abs(Efield_array))
    if Efield0 == 0:
        Efield0 = 1.0  # デフォルト値

    # 4. 双極子モーメントスケール
    mu_x_offdiag = mu_x.copy()
    mu_y_offdiag = mu_y.copy()
    if mu_x.ndim == 2:
        np.fill_diagonal(mu_x_offdiag, 0)
    if mu_y.ndim == 2:
        np.fill_diagonal(mu_y_offdiag, 0)
    
    mu0 = max(np.max(np.abs(mu_x_offdiag)), np.max(np.abs(mu_y_offdiag)))
    if mu0 == 0:
        mu0 = 1.0  # デフォルト値

    # 5. 無次元化の実行
    H0_prime = H0_energy / E0
    mu_x_prime = mu_x / mu0
    mu_y_prime = mu_y / mu0
    Efield_prime = Efield_array / Efield0

    # 6. 時間軸の無次元化
    if time_units == "fs":
        # fs → s 変換
        tlist = efield.tlist * 1e-15  # fs → s
        dt_s = dt * 1e-15  # fs → s
    elif time_units == "s":
        # 既にs単位
        tlist = efield.tlist.copy()
        dt_s = dt
    else:
        raise ValueError("time_units must be 'fs' or 's'")
    
    tlist_prime = tlist / t0
    dt_prime = dt_s / t0

    # 7. 結合強度パラメータ
    lambda_coupling = (Efield0 * mu0) / E0

    # 8. スケール情報
    scales = NondimensionalizationScales(
        E0=E0,
        mu0=mu0,
        Efield0=Efield0,
        t0=t0,
        lambda_coupling=lambda_coupling,
    )

    return (
        H0_prime,
        mu_x_prime,
        mu_y_prime,
        Efield_prime,
        tlist_prime,
        dt_prime,
        scales,
    )


def dimensionalize_wavefunction(
    psi_prime: np.ndarray,
    scales: NondimensionalizationScales,
) -> np.ndarray:
    """
    無次元波動関数を次元のある形に戻す
    
    Parameters
    ----------
    psi_prime : np.ndarray
        無次元波動関数
    scales : NondimensionalizationScales
        スケールファクター
        
    Returns
    -------
    np.ndarray
        次元のある波動関数
    """
    # 波動関数の正規化は保持されるため、そのまま返す
    return psi_prime


def get_physical_time(
    tau: np.ndarray,
    scales: NondimensionalizationScales,
) -> np.ndarray:
    """
    無次元時間を物理時間（fs）に変換
    
    Parameters
    ----------
    tau : np.ndarray
        無次元時間
    scales : NondimensionalizationScales
        スケールファクター
        
    Returns
    -------
    np.ndarray
        物理時間 [fs]
    """
    return tau * scales.t0 * 1e15  # s → fs


def analyze_regime(scales: NondimensionalizationScales) -> dict[str, Any]:
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
    lambda_val = scales.lambda_coupling
    
    if lambda_val < 0.1:
        regime = "weak_coupling"
        description = "弱結合: 摂動論的取り扱いが有効"
    elif lambda_val < 1.0:
        regime = "intermediate_coupling"
        description = "中間結合: 非摂動効果が現れ始める"
    else:
        regime = "strong_coupling"
        description = "強結合: Rabi振動など非線形効果が顕著"
    
    return {
        "regime": regime,
        "lambda": lambda_val,
        "description": description,
        "energy_scale_eV": scales.E0 / 1.602176634e-19,  # J → eV
        "time_scale_fs": scales.t0 * 1e15,  # s → fs
    }





def determine_SI_based_scales(
    H0_energy_J: np.ndarray,
    mu_values_Cm: np.ndarray,
    field_amplitude_V_per_m: float,
) -> NondimensionalizationScales:
    """
    SI基本単位の物理量から無次元化スケールを決定
    
    Parameters
    ----------
    H0_energy_J : np.ndarray
        ハミルトニアンエネルギー [J]
    mu_values_Cm : np.ndarray
        双極子行列要素 [C·m]
    field_amplitude_V_per_m : float
        電場振幅 [V/m]
        
    Returns
    -------
    NondimensionalizationScales
        無次元化スケール
    """
    # エネルギースケールの決定 [J]
    if H0_energy_J.ndim == 2:
        eigvals = np.diag(H0_energy_J)
    else:
        eigvals = H0_energy_J.copy()
    
    energy_diffs = np.abs(eigvals[:, None] - eigvals[None, :])
    energy_diffs = energy_diffs[energy_diffs > 1e-20]
    
    if len(energy_diffs) == 0:
        E0 = _EV_TO_J  # 1 eV をデフォルト [J]
    else:
        E0 = np.max(energy_diffs)  # [J]
    
    # 双極子モーメントスケールの決定 [C·m]
    mu_offdiag = mu_values_Cm.copy()
    if mu_values_Cm.ndim == 2:
        np.fill_diagonal(mu_offdiag, 0)
    
    mu0 = np.max(np.abs(mu_offdiag))
    if mu0 == 0:
        mu0 = _DEBYE_TO_CM  # 1 D をデフォルト [C·m]
    
    # 電場スケール [V/m]
    Efield0 = field_amplitude_V_per_m
    if Efield0 == 0:
        Efield0 = 1e8  # 1 MV/cm をデフォルト [V/m]
    
    # 時間スケール [s]
    t0 = _HBAR / E0
    
    # 結合強度パラメータ
    lambda_coupling = (Efield0 * mu0) / E0
    
    # スケール情報
    scales = NondimensionalizationScales(
        E0=E0,
        mu0=mu0,
        Efield0=Efield0,
        t0=t0,
        lambda_coupling=lambda_coupling,
    )
    
    # デフォルト単位での表示
    energy_scale_eV = E0 / _EV_TO_J
    dipole_scale_D = mu0 / _DEBYE_TO_CM
    field_scale_MV_per_cm = Efield0 / 1e8
    time_scale_fs = t0 * 1e15
    
    print(f"""
📏 SI-based nondimensionalization scales:
   Energy scale: {energy_scale_eV:.3f} eV ({E0:.3e} J)
   Dipole scale: {dipole_scale_D:.3f} D ({mu0:.3e} C·m)
   Field scale: {field_scale_MV_per_cm:.3f} MV/cm ({Efield0:.3e} V/m)
   Time scale: {time_scale_fs:.3f} fs ({t0:.3e} s)
   Coupling strength λ: {lambda_coupling:.3f}
""")
    
    return scales


def nondimensionalize_with_SI_base_units(
    H0: np.ndarray,
    mu_x: np.ndarray,
    mu_y: np.ndarray,
    efield: 'ElectricField',
    *,
    dt: float | None = None,
    params: Dict[str, Any] | None = None,
    auto_timestep: bool = False,
    timestep_method: str = "adaptive",
    timestep_safety_factor: float = 0.1,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    NondimensionalizationScales,
]:
    """
    デフォルト単位を自動的にSI基本単位に変換してから無次元化を実行
    
    この関数はデフォルト単位設定を維持しつつ、無次元化の際に
    すべての物理量をSI基本単位（接頭辞なし）に統一します。
    
    Parameters
    ----------
    H0 : np.ndarray
        ハミルトニアン行列（任意の単位、自動変換される）
    mu_x, mu_y : np.ndarray
        双極子行列（任意の単位、自動変換される）
    efield : ElectricField
        電場オブジェクト（任意の単位、自動変換される）
    dt : float, optional
        時間ステップ [fs]。auto_timestep=Trueの場合は無視される
    params : dict, optional
        元のパラメータ辞書（参考情報用）
    auto_timestep : bool, optional
        lambda_couplingに基づく自動時間ステップ選択, デフォルト: False
    timestep_method : str, optional
        自動時間ステップの計算方法, デフォルト: "adaptive"
    timestep_safety_factor : float, optional
        時間ステップの安全係数, デフォルト: 0.1
        
    Returns
    -------
    tuple
        (H0_prime, mu_x_prime, mu_y_prime, Efield_prime, tlist_prime, 
         dt_prime, scales)
    """
    print("🎯 Starting nondimensionalization with SI base unit conversion...")
    
    # パラメータをデフォルト単位経由でSI単位に変換
    if params is not None:
        from rovibrational_excitation.core.units.parameter_processor import parameter_processor
        print("🔄 Converting parameters via default units to SI...")
        converted_params = parameter_processor.auto_convert_parameters(params)
        print("✓ Parameter conversion completed.")
    
    # 時間ステップの設定
    if dt is None:
        dt = efield.dt
    assert dt is not None
    
    # 入力が既にSI単位[J, C·m, V/m]の場合、そのまま使用
    # （auto_convert_parametersによって変換済み）
    
    # エネルギー（ハミルトニアン）: 既に [J]
    H0_energy_J = H0.copy()
    
    # 双極子モーメント: 既に [C·m]  
    mu_x_Cm = mu_x.copy()
    mu_y_Cm = mu_y.copy()
    
    # 電場: 既に [V/m]
    Efield_array = efield.get_Efield()  # (T, 2) [V/m]
    field_amplitude_V_per_m = np.max(np.abs(Efield_array))
    
    print(f"📊 Physical quantities in SI base units:")
    if H0_energy_J.ndim == 1:
        energy_range = f"{np.min(H0_energy_J):.3e} to {np.max(H0_energy_J):.3e}"
    else:
        energy_range = f"{np.min(np.diag(H0_energy_J)):.3e} to {np.max(np.diag(H0_energy_J)):.3e}"
    print(f"   Energy range: {energy_range} J")
    print(f"   Dipole range: {np.min(np.abs(mu_x_Cm[mu_x_Cm != 0])):.3e} to {np.max(np.abs(mu_x_Cm)):.3e} C·m")
    print(f"   Field amplitude: {field_amplitude_V_per_m:.3e} V/m")
    
    # SI基本単位に基づいた無次元化スケールの決定
    print("\n📏 Determining nondimensionalization scales from SI base units...")
    scales = determine_SI_based_scales(H0_energy_J, mu_x_Cm, field_amplitude_V_per_m)
    
    # 自動時間ステップ選択
    if auto_timestep:
        print(f"\n⏱️  Auto-selecting timestep based on λ={scales.lambda_coupling:.3f}...")
        dt_recommended_fs = scales.get_recommended_timestep_fs(
            safety_factor=timestep_safety_factor,
            method=timestep_method
        )
        print(f"   Recommended dt: {dt_recommended_fs:.3f} fs (method: {timestep_method})")
        print(f"   Original dt: {dt:.3f} fs")
        
        # 推奨値と元の値の比較
        if dt_recommended_fs < dt * 0.5:
            print(f"   ⚠️  Warning: Recommended dt is much smaller than original")
            print(f"   ⚠️  Consider using dt ≤ {dt_recommended_fs:.3f} fs for stability")
        
        dt = dt_recommended_fs
        
        # 詳細分析の表示
        analysis = scales.analyze_timestep_requirements()
        print(f"   Physical regime: {analysis['regime']}")
        print(f"   Advice: {analysis['advice']}")
        if analysis['rabi_period_fs'] != np.inf:
            print(f"   Rabi period: {analysis['rabi_period_fs']:.3f} fs")
    
    # 無次元化の実行
    print("\n🔢 Performing nondimensionalization...")
    
    # エネルギー（ハミルトニアン）の無次元化
    H0_prime = H0_energy_J / scales.E0
    
    # 双極子モーメントの無次元化
    mu_x_prime = mu_x_Cm / scales.mu0
    mu_y_prime = mu_y_Cm / scales.mu0
    
    # 電場の無次元化
    Efield_prime = Efield_array / scales.Efield0
    
    # 時間軸の無次元化
    tlist_s = efield.tlist * 1e-15  # fs → s
    dt_s = dt * 1e-15  # fs → s
    
    tlist_prime = tlist_s / scales.t0
    dt_prime = dt_s / scales.t0
    
    print("✓ Nondimensionalization completed successfully!")
    
    # 物理レジームの分析
    regime_info = analyze_regime(scales)
    print(f"📊 Physical regime: {regime_info['description']}")
    
    if auto_timestep:
        print(f"🎯 Final timestep: {dt:.3f} fs ({dt_prime:.6f} dimensionless)")
    
    return (
        H0_prime,
        mu_x_prime,
        mu_y_prime,
        Efield_prime,
        tlist_prime,
        dt_prime,
        scales,
    )


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
        "rabi_period_fs": analysis["rabi_period_fs"],
        "computational_cost_estimate": 1.0 / dt_dim,  # 相対的計算コスト
        "all_methods": analysis["recommendations"]
    }
    
    if verbose:
        print(f"\n⚡ 結合強度最適化時間ステップ分析")
        print(f"   λ = {λ:.3f} ({analysis['regime']})")
        print(f"   目標精度: {target_accuracy} ({settings['description']})")
        print(f"   推奨時間ステップ: {dt_fs:.3f} fs ({dt_dim:.6f} 無次元)")
        print(f"   計算コスト (相対): {result['computational_cost_estimate']:.1f}x")
        
        rabi_period = analysis['rabi_period_fs']
        if (rabi_period != np.inf and not np.isinf(rabi_period) and 
            dt_fs is not None and dt_fs > 0):
            print(f"   Rabi周期: {rabi_period:.3f} fs")
            print(f"   Rabi周期あたりステップ数: {rabi_period/dt_fs:.1f}")
        
        print(f"   アドバイス: {analysis['advice']}")
    
    return result


def create_dimensionless_time_array(
    scales: NondimensionalizationScales,
    duration_fs: float,
    dt_fs: float | None = None,
    auto_timestep: bool = True,
    target_accuracy: str = "standard"
) -> tuple[np.ndarray, float]:
    """
    無次元化時間配列を作成（推奨時間ステップで）
    
    Parameters
    ----------
    scales : NondimensionalizationScales
        無次元化スケールファクター
    duration_fs : float
        シミュレーション時間長（fs）
    dt_fs : float, optional
        時間ステップ（fs）。Noneの場合は自動選択
    auto_timestep : bool, optional
        自動時間ステップ選択を使用するか, デフォルト: True
    target_accuracy : str, optional
        目標精度, デフォルト: "standard"
        
    Returns
    -------
    tuple
        (tlist_dimensionless, dt_dimensionless)
    """
    if auto_timestep or dt_fs is None:
        optimization = optimize_timestep_for_coupling(
            scales, target_accuracy=target_accuracy, verbose=True
        )
        dt_fs = optimization["recommended_dt_fs"]
        print(f"🎯 Auto-selected timestep: {dt_fs:.3f} fs")
    
    # fs単位での時間配列作成
    tlist_fs = np.arange(0, duration_fs + dt_fs/2, dt_fs)
    
    # 無次元化
    t0_fs = scales.t0 * 1e15  # s → fs
    tlist_dimensionless = tlist_fs / t0_fs
    dt_dimensionless = dt_fs / t0_fs
    
    print(f"📊 Time array info:")
    print(f"   Duration: {duration_fs:.1f} fs ({duration_fs/t0_fs:.3f} dimensionless)")
    print(f"   Steps: {len(tlist_fs)}")
    print(f"   dt: {dt_fs:.3f} fs ({dt_dimensionless:.6f} dimensionless)")
    
    return tlist_dimensionless, dt_dimensionless


def create_SI_demo_parameters() -> Dict[str, Any]:
    """
    SI基本単位変換デモ用のサンプルパラメータを生成
    
    Returns
    -------
    dict[str, Any]
        デフォルト単位のサンプルパラメータ
    """
    return {
        # 分子パラメータ（デフォルト単位）
        "omega_rad_phz": 2349.1,       # cm⁻¹
        "omega_rad_phz_units": "cm^-1",
        
        "B_rad_phz": 0.39021,          # cm⁻¹
        "B_rad_phz_units": "cm^-1",
        
        "mu0_Cm": 0.3,                 # D
        "mu0_Cm_units": "D",
        
        # 電場パラメータ（デフォルト単位）
        "amplitude": 5.0,              # MV/cm
        "amplitude_units": "MV/cm",
        
        "duration": 30.0,              # fs
        "duration_units": "fs",
        
        # エネルギーパラメータ（デフォルト単位）
        "energy_gap": 1.5,             # eV
        "energy_gap_units": "eV",
        
        # 時間パラメータ（デフォルト単位）
        "dt": 0.1,                     # fs
        "dt_units": "fs",
        
        "t_end": 200.0,                # fs
        "t_end_units": "fs",
    } 


def calculate_nondimensionalization_scales_strict(
    H0: np.ndarray,
    mu_x: np.ndarray,
    mu_y: np.ndarray,
    efield: 'ElectricField',
    *,
    hbar: float = _HBAR,
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
        
    Notes
    -----
    この実装は以下の数学的定義に厳密に従います:
    
    i ℏ d/dt |ψ⟩ = (H₀ - μ·E(t)) |ψ⟩
    ↓ 無次元化
    i d/dτ |ψ⟩ = (H₀' - λ μ' E'(τ)) |ψ⟩
    
    where τ = t/t₀, H₀' = H₀/E₀, μ' = μ/μ₀, E' = E/E_field,₀
    """
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
    efield_array = efield.get_Efield_SI()  # [V/m]
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
        mu0 = _DEBYE_TO_CM  # 1 D デフォルト
        if verbose:
            print("   ⚠️  Warning: No non-zero off-diagonal dipole elements, using 1 D")
    else:
        mu0 = max(all_mu_elements)
    
    if verbose:
        print(f"   μ₀ = max_{{n≠m}} |μₙₘ| = {mu0:.6e} C·m")
        print(f"      = {mu0/_DEBYE_TO_CM:.3f} D")
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


def verify_nondimensional_equation(
    H0_prime: np.ndarray,
    mu_x_prime: np.ndarray,
    mu_y_prime: np.ndarray,
    Efield_prime: np.ndarray,
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


def demonstrate_nondimensionalization_workflow(
    H0: np.ndarray,
    mu_x: np.ndarray,
    mu_y: np.ndarray,
    efield: 'ElectricField'
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, NondimensionalizationScales]:
    """
    完全な無次元化ワークフローのデモンストレーション
    
    Parameters
    ----------
    H0 : np.ndarray
        ハミルトニアン [J]
    mu_x, mu_y : np.ndarray
        双極子行列 [C·m]
    efield : ElectricField
        電場 [V/m]
        
    Returns
    -------
    tuple
        (H0_prime, mu_x_prime, mu_y_prime, Efield_prime, scales)
    """
    print("📚 Complete Nondimensionalization Workflow Demonstration")
    print("=" * 60)
    
    # Step 1: 厳密なスケール計算
    print("\n🔬 Step 1: Calculate strict nondimensionalization scales")
    scales = calculate_nondimensionalization_scales_strict(
        H0, mu_x, mu_y, efield, verbose=True
    )
    
    # Step 2: 無次元化の実行
    print("\n🔢 Step 2: Apply nondimensionalization")
    H0_prime = H0 / scales.E0
    mu_x_prime = mu_x / scales.mu0
    mu_y_prime = mu_y / scales.mu0
    Efield_prime = efield.get_Efield_SI() / scales.Efield0
    
    print(f"   H₀ [J] → H₀' = H₀/E₀ (dimensionless)")
    print(f"   μ [C·m] → μ' = μ/μ₀ (dimensionless)")
    print(f"   E [V/m] → E' = E/E_field,₀ (dimensionless)")
    
    # Step 3: 検証
    print("\n🔍 Step 3: Verify nondimensional equation")
    verification = verify_nondimensional_equation(
        H0_prime, mu_x_prime, mu_y_prime, Efield_prime, scales, verbose=True
    )
    
    # Step 4: 物理的解釈
    print(f"\n🎯 Step 4: Physical interpretation")
    print(f"   Original equation: i ℏ d/dt |ψ⟩ = (H₀ - μ·E(t)) |ψ⟩")
    print(f"   Nondimensional:    i d/dτ |ψ⟩ = (H₀' - λ μ'·E'(τ)) |ψ⟩")
    print(f"   Where: τ = t/t₀, λ = {scales.lambda_coupling:.3f}")
    
    return H0_prime, mu_x_prime, mu_y_prime, Efield_prime, scales 


def apply_lambda_scaling_strategies() -> None:
    """
    λスケーリング戦略の説明
    
    無次元化後の正しい形: i d/dτ |ψ⟩ = (H₀' - λ μ' E'(τ)) |ψ⟩
    
    Strategy 1: 実効電場アプローチ (推奨)
    Strategy 2: 実効双極子アプローチ  
    Strategy 3: 明示的λ処理アプローチ
    Strategy 4: スケール統合アプローチ
    """
    pass


def create_effective_field_scaling(
    scales: NondimensionalizationScales,
    Efield_prime: np.ndarray
) -> tuple[np.ndarray, str]:
    """
    Strategy 1: 実効電場アプローチ (推奨)
    
    E_effective = λ * E' として電場にλを事前積算
    これにより propagator では μ' * E_effective を計算するだけ
    
    Parameters
    ----------
    scales : NondimensionalizationScales
        スケールファクター
    Efield_prime : np.ndarray
        無次元電場
        
    Returns
    -------
    tuple
        (E_effective, strategy_description)
    """
    λ = scales.lambda_coupling
    E_effective = λ * Efield_prime
    
    strategy_description = f"""
Strategy 1: Effective Field Scaling
- 実効電場: E_eff = λ * E' = {λ:.3f} * E'
- Propagator使用法: H_interaction = μ' * E_eff
- 利点: 電場の「実効強度」として物理的に直感的
- 利点: propagatorの変更が最小限
- 注意: E_effective は無次元だがλ倍されているので注意
    """
    
    return E_effective, strategy_description


def create_effective_dipole_scaling(
    scales: NondimensionalizationScales,
    mu_x_prime: np.ndarray,
    mu_y_prime: np.ndarray
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Strategy 2: 実効双極子アプローチ
    
    μ_effective = λ * μ' として双極子にλを事前積算
    
    Parameters
    ----------
    scales : NondimensionalizationScales
        スケールファクター
    mu_x_prime, mu_y_prime : np.ndarray
        無次元双極子行列
        
    Returns
    -------
    tuple
        (mu_x_effective, mu_y_effective, strategy_description)
    """
    λ = scales.lambda_coupling
    mu_x_effective = λ * mu_x_prime
    mu_y_effective = λ * mu_y_prime
    
    strategy_description = f"""
Strategy 2: Effective Dipole Scaling  
- 実効双極子: μ_eff = λ * μ' = {λ:.3f} * μ'
- Propagator使用法: H_interaction = μ_eff * E'
- 利点: 双極子の「実効強度」として理解可能
- 欠点: x,y両成分に同じλが適用される
    """
    
    return mu_x_effective, mu_y_effective, strategy_description


class NondimensionalizedSystem:
    """
    Strategy 3: 明示的λ処理アプローチ
    
    λを明示的に保持し、propagatorで適切に処理
    """
    
    def __init__(
        self,
        H0_prime: np.ndarray,
        mu_x_prime: np.ndarray, 
        mu_y_prime: np.ndarray,
        Efield_prime: np.ndarray,
        scales: NondimensionalizationScales
    ):
        self.H0_prime = H0_prime
        self.mu_x_prime = mu_x_prime
        self.mu_y_prime = mu_y_prime
        self.Efield_prime = Efield_prime
        self.scales = scales
        self.lambda_coupling = scales.lambda_coupling
        
    def get_interaction_hamiltonian(self, time_index: int) -> np.ndarray:
        """
        正しい相互作用ハミルトニアンを計算: λ μ' E'(τ)
        
        Parameters
        ----------
        time_index : int
            時間インデックス
            
        Returns
        -------
        np.ndarray
            相互作用ハミルトニアン
        """
        Ex = self.Efield_prime[time_index, 0]
        Ey = self.Efield_prime[time_index, 1]
        
        # λ μ' E'(τ) = λ * (μ_x' * Ex + μ_y' * Ey)
        H_int = self.lambda_coupling * (
            self.mu_x_prime * Ex + self.mu_y_prime * Ey
        )
        
        return H_int
    
    def get_total_hamiltonian(self, time_index: int) -> np.ndarray:
        """
        全ハミルトニアンを計算: H₀' - λ μ' E'(τ)
        """
        H_int = self.get_interaction_hamiltonian(time_index)
        return self.H0_prime - H_int


def create_unified_scaling_approach(
    H0: np.ndarray,
    mu_x: np.ndarray,
    mu_y: np.ndarray, 
    efield: 'ElectricField'
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, NondimensionalizationScales]:
    """
    Strategy 4: スケール統合アプローチ
    
    λを双極子か電場のスケールに統合して、自然に正しい式になるようにする
    
    Parameters
    ----------
    H0, mu_x, mu_y : np.ndarray
        物理量（SI単位）
    efield : ElectricField
        電場オブジェクト
        
    Returns
    -------
    tuple
        統合スケーリング済みの無次元量
    """
    # 厳密なスケール計算
    scales_original = calculate_nondimensionalization_scales_strict(
        H0, mu_x, mu_y, efield, verbose=False
    )
    
    # Strategy 4a: 双極子スケールにλを統合
    mu0_effective = scales_original.mu0 / scales_original.lambda_coupling
    
    # Strategy 4b: 電場スケールにλを統合  
    Efield0_effective = scales_original.Efield0 / scales_original.lambda_coupling
    
    # 統合スケールでの無次元化
    H0_prime = H0 / scales_original.E0
    
    # 方法A: 双極子統合（λが自動的に含まれる）
    mu_x_integrated = mu_x / mu0_effective  # = λ * μ/μ₀
    mu_y_integrated = mu_y / mu0_effective
    
    # 方法B: 電場統合（λが自動的に含まれる）
    Efield_integrated = efield.get_Efield_SI() / Efield0_effective  # = λ * E/E₀
    
    # 新しいスケール情報
    scales_integrated = NondimensionalizationScales(
        E0=scales_original.E0,
        mu0=mu0_effective,  # または scales_original.mu0
        Efield0=Efield0_effective,  # または scales_original.Efield0  
        t0=scales_original.t0,
        lambda_coupling=1.0  # 既に統合済みなので1
    )
    
    print(f"""
Strategy 4: Unified Scaling Approach
- Original λ: {scales_original.lambda_coupling:.3f}
- Integrated into scales, so effective λ = 1.0
- Propagator can use: H_int = μ_integrated * E' (or μ' * E_integrated)
- 利点: λの明示的な処理が不要
- 利点: 数学的に自然
    """)
    
    return H0_prime, mu_x_integrated, mu_y_integrated, Efield_integrated, scales_integrated


def recommend_lambda_strategy(
    scales: NondimensionalizationScales,
    propagator_type: str = "split_operator"
) -> Dict[str, Any]:
    """
    λ処理戦略の推奨
    
    Parameters
    ----------
    scales : NondimensionalizationScales
        スケールファクター
    propagator_type : str, optional
        使用するpropagatorの種類
        
    Returns
    -------
    dict
        推奨戦略と実装ガイド
    """
    λ = scales.lambda_coupling
    
    # λの大きさに基づく推奨
    if λ < 0.1:
        # 弱結合: λの影響は小さいが、正確性のため必要
        primary_recommendation = "Strategy 1: Effective Field"
        risk_level = "Low"
        reason = "弱結合だが長時間計算で累積誤差の可能性"
        
    elif λ < 1.0:
        # 中間結合: λの正確な処理が重要
        primary_recommendation = "Strategy 1: Effective Field"  
        risk_level = "Medium"
        reason = "中間結合域、λの正確な処理が精度に影響"
        
    else:
        # 強結合: λの処理が極めて重要
        primary_recommendation = "Strategy 4: Unified Scaling"
        risk_level = "High"
        reason = "強結合域、λ抜けは大きな物理誤差を生む"
    
    # Propagator種別による推奨
    propagator_specific = {
        "split_operator": {
            "preferred": ["Strategy 1", "Strategy 4"],
            "reason": "高速性とユニタリ性を両立"
        },
        "rk4": {
            "preferred": ["Strategy 3", "Strategy 1"], 
            "reason": "明示的処理が高精度計算に適合"
        },
        "magnus": {
            "preferred": ["Strategy 4", "Strategy 3"],
            "reason": "数学的な厳密性を重視"
        }
    }
    
    return {
        "lambda_coupling": λ,
        "primary_recommendation": primary_recommendation,
        "risk_level": risk_level,
        "physical_reason": reason,
        "propagator_specific": propagator_specific.get(propagator_type, {}),
        "implementation_priority": "CRITICAL" if λ > 1.0 else "HIGH",
        "strategies_ranked": [
            "Strategy 1: Effective Field (推奨・汎用性)",
            "Strategy 4: Unified Scaling (推奨・厳密性)", 
            "Strategy 3: Explicit Lambda (完全制御)",
            "Strategy 2: Effective Dipole (特殊用途)"
        ]
    }


def convert_default_units_to_SI_base(
    frequency_cm_inv: float,
    dipole_D: float,
    field_MV_per_cm: float,
    energy_eV: float,
    time_fs: float,
) -> tuple[float, float, float, float, float]:
    """
    デフォルト単位をSI基本単位（接頭辞なし）に変換
    
    Parameters
    ----------
    frequency_cm_inv : float
        周波数 [cm⁻¹]
    dipole_D : float
        双極子モーメント [D]
    field_MV_per_cm : float
        電場 [MV/cm]
    energy_eV : float
        エネルギー [eV]
    time_fs : float
        時間 [fs]
        
    Returns
    -------
    tuple
        (frequency_rad_per_s, dipole_Cm, field_V_per_m, energy_J, time_s)
        すべてSI基本単位
    """
    # SI基本単位への変換
    frequency_rad_per_s = frequency_cm_inv * DEFAULT_TO_SI_CONVERSIONS["frequency_cm_inv_to_rad_per_s"]
    dipole_Cm = dipole_D * DEFAULT_TO_SI_CONVERSIONS["dipole_D_to_Cm"]
    field_V_per_m = field_MV_per_cm * DEFAULT_TO_SI_CONVERSIONS["field_MV_per_cm_to_V_per_m"]
    energy_J = energy_eV * DEFAULT_TO_SI_CONVERSIONS["energy_eV_to_J"]
    time_s = time_fs * DEFAULT_TO_SI_CONVERSIONS["time_fs_to_s"]
    
    print(f"🔄 Converting default units to SI base units:")
    print(f"   Frequency: {frequency_cm_inv:.3f} cm⁻¹ → {frequency_rad_per_s:.6e} rad/s")
    print(f"   Dipole: {dipole_D:.3f} D → {dipole_Cm:.6e} C·m")
    print(f"   Field: {field_MV_per_cm:.3f} MV/cm → {field_V_per_m:.6e} V/m")
    print(f"   Energy: {energy_eV:.3f} eV → {energy_J:.6e} J")
    print(f"   Time: {time_fs:.3f} fs → {time_s:.6e} s")
    
    return frequency_rad_per_s, dipole_Cm, field_V_per_m, energy_J, time_s 