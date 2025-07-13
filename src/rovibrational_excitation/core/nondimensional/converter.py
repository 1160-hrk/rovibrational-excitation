"""
converter.py
=============
無次元化の変換機能を提供するモジュール。

このモジュールは物理量の無次元化変換を行う実装を含む。
"""
from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING
import numpy as np

# スケールクラス
from .scales import NondimensionalizationScales
from .utils import (
    get_energy_scale_from_hamiltonian,
    get_dipole_scale_from_matrices,
    get_electric_field_scale,
    DEFAULT_TO_SI_CONVERSIONS,
    _HBAR,
    _EV_TO_J,
)

# 型ヒント用 (循環参照を避けるため文字列で書く)
if TYPE_CHECKING:  # pragma: no cover
    from rovibrational_excitation.core.electric_field import ElectricField
    from rovibrational_excitation.core.basis.hamiltonian import Hamiltonian
    from rovibrational_excitation.dipole.base import DipoleMatrixBase


def nondimensionalize_system(
    H0: np.ndarray,
    mu_x: np.ndarray,
    mu_y: np.ndarray,
    efield: "ElectricField",
    *,
    dt: float | None = None,
    H0_units: str = "energy",
    time_units: str = "fs",
    hbar: float = _HBAR,
    max_time_scale_fs: float = 1000.0,
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
    max_time_scale_fs : float
        時間スケール上限 [fs]

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
        H0_energy = H0.copy()
    elif H0_units == "frequency":
        # H0は周波数単位（rad/fs）なので、Jに変換
        H0_energy = H0 * hbar / 1e-15  # rad/fs → J
    else:
        raise ValueError("H0_units must be 'energy' or 'frequency'")
    
    E0 = get_energy_scale_from_hamiltonian(
        H0_energy, max_time_scale_fs, hbar
    )

    # 2. 時間スケール
    t0 = hbar / E0  # [s]
    
    # 3. 電場スケール
    Efield0 = get_electric_field_scale(efield)

    # 4. 双極子モーメントスケール
    mu0 = get_dipole_scale_from_matrices(mu_x, mu_y)

    # 5. 無次元化の実行
    H0_prime = H0_energy / E0
    mu_x_prime = mu_x / mu0
    mu_y_prime = mu_y / mu0
    Efield_prime = efield.get_Efield() / Efield0

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
    E0 = get_energy_scale_from_hamiltonian(H0_energy_J)
    
    # 双極子モーメントスケールの決定 [C·m]
    mu0 = get_dipole_scale_from_matrices(mu_values_Cm, mu_values_Cm)
    
    # 電場スケール [V/m]
    Efield0 = field_amplitude_V_per_m if field_amplitude_V_per_m > 0 else 1e8
    
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
    dipole_scale_D = mu0 / DEFAULT_TO_SI_CONVERSIONS["dipole_D_to_Cm"]
    field_scale_MV_per_cm = Efield0 / DEFAULT_TO_SI_CONVERSIONS["field_MV_per_cm_to_V_per_m"]
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
    efield: np.ndarray,
    tlist: np.ndarray,
    *,
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
    
    Parameters
    ----------
    H0 : np.ndarray
        ハミルトニアン行列（J）
    mu_x, mu_y : np.ndarray
        双極子行列（C·m）
    efield : np.ndarray
        電場（V/m）
    tlist : np.ndarray
        時間軸（s）
    params : dict,  optional
        パラメータ辞書（参考情報用）
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
    
    # 入力が既にSI単位[J, C·m, V/m]の場合、そのまま使用
    H0_energy_J = H0.copy()
    mu_x_Cm = mu_x.copy()
    mu_y_Cm = mu_y.copy()
    
    # 電場: 既に [V/m]
    field_amplitude_V_per_m = np.max(np.abs(efield))
    
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
    dt_final = (tlist[1] - tlist[0])  # Default dt in seconds
    if auto_timestep:
        print(f"\n⏱️  Auto-selecting timestep based on λ={scales.lambda_coupling:.3f}...")
        dt_recommended_fs = scales.get_recommended_timestep_fs(
            safety_factor=timestep_safety_factor,
            method=timestep_method
        )
        dt_recommended_s = dt_recommended_fs * 1e-15
        print(f"   Recommended dt: {dt_recommended_fs:.3f} fs (method: {timestep_method})")
        print(f"   Original dt: {dt_final * 1e15:.3f} fs")
        
        # 推奨値と元の値の比較
        if dt_recommended_s < dt_final * 0.5:
            print(f"   ⚠️  Warning: Recommended dt is much smaller than original")
            print(f"   ⚠️  Consider using dt ≤ {dt_recommended_fs:.3f} fs for stability")
        
        dt_final = dt_recommended_s
    
    # 無次元化の実行
    print("\n🔢 Performing nondimensionalization...")
    
    # エネルギー（ハミルトニアン）の無次元化
    H0_prime = H0_energy_J / scales.E0
    
    # 双極子モーメントの無次元化
    mu_x_prime = mu_x_Cm / scales.mu0
    mu_y_prime = mu_y_Cm / scales.mu0
    
    # 電場の無次元化
    Efield_prime = efield / scales.Efield0

    # 時間軸の無次元化
    tlist_s = tlist * 1e-15
    
    tlist_prime = tlist_s / scales.t0
    dt_prime = dt_final / scales.t0
    
    print("✓ Nondimensionalization completed successfully!")
    
    return (
        H0_prime,
        mu_x_prime,
        mu_y_prime,
        Efield_prime,
        tlist_prime,
        dt_prime,
        scales,
    )


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
        # 分析機能を使用（循環インポートを避けるため遅延インポート）
        from .analysis import NondimensionalAnalyzer
        
        optimization = NondimensionalAnalyzer.optimize_timestep_for_coupling(
            scales, target_accuracy=target_accuracy, verbose=True
        )
        dt_fs = optimization["recommended_dt_fs"]
        print(f"🎯 Auto-selected timestep: {dt_fs:.3f} fs")
    
    # dt_fs がまだ None の場合のフォールバック
    if dt_fs is None:
        raise ValueError("dt_fs must be provided or auto_timestep must be True")
    
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


def nondimensionalize_from_objects(
    hamiltonian: "Hamiltonian",
    dipole_matrix: "DipoleMatrixBase",
    efield: "ElectricField",
    *,
    auto_timestep: bool = False,
    timestep_method: str = "adaptive",
    timestep_safety_factor: float = 0.1,
    verbose: bool = True,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    NondimensionalizationScales,
]:
    """
    HamiltonianとDipoleMatrixBaseクラスから自動的にSI単位系に変換して無次元化を実行
    
    Parameters
    ----------
    hamiltonian : Hamiltonian
        ハミルトニアンオブジェクト（内部単位管理）
    dipole_matrix : DipoleMatrixBase
        双極子行列オブジェクト（内部単位管理）
    efield : ElectricField
        電場オブジェクト
    dt : float, optional
        時間ステップ [fs]。auto_timestep=Trueの場合は無視される
    time_units : str, optional
        時間の単位。"fs" または "s"。デフォルトは"fs"
    auto_timestep : bool, optional
        lambda_couplingに基づく自動時間ステップ選択, デフォルト: False
    timestep_method : str, optional
        自動時間ステップの計算方法, デフォルト: "adaptive"
    timestep_safety_factor : float, optional
        時間ステップの安全係数, デフォルト: 0.1
    verbose : bool, optional
        詳細出力の有無, デフォルト: True
        
    Returns
    -------
    tuple
        (H0_prime, mu_x_prime, mu_y_prime, mu_z_prime, Efield_prime, tlist_prime, 
         dt_prime, scales)
    """
    if verbose:
        print("🎯 Nondimensionalization from Hamiltonian and DipoleMatrix objects...")
    
    # 1. HamiltonianクラスからSI単位系（J）でハミルトニアン行列を取得
    H0_energy_J = hamiltonian.get_matrix(units="J")
    
    if verbose:
        print(f"📊 Hamiltonian matrix: {hamiltonian.shape} in J units")
        if hamiltonian.is_diagonal():
            eigenvals = hamiltonian.get_eigenvalues(units="J")
            print(f"   Eigenvalues: {eigenvals[0]:.3e} to {eigenvals[-1]:.3e} J")
    
    # 2. DipoleMatrixBaseクラスからSI単位系（C·m）で双極子行列を取得
    mu_x_Cm = dipole_matrix.get_mu_x_SI(dense=True)
    mu_y_Cm = dipole_matrix.get_mu_y_SI(dense=True)
    mu_z_Cm = dipole_matrix.get_mu_z_SI(dense=True)
    
    if verbose:
        print(f"📊 Dipole matrices: {mu_x_Cm.shape} in C·m units")
        
        mu_x_nonzero = np.abs(mu_x_Cm[mu_x_Cm != 0])
        if mu_x_nonzero.size > 0:
            print(f"   mu_x range: {np.min(mu_x_nonzero):.3e} to {np.max(mu_x_nonzero):.3e} C·m")
        else:
            print("   mu_x range: All elements are zero.")
            
        mu_y_nonzero = np.abs(mu_y_Cm[mu_y_Cm != 0])
        if mu_y_nonzero.size > 0:
            print(f"   mu_y range: {np.min(mu_y_nonzero):.3e} to {np.max(mu_y_nonzero):.3e} C·m")
        else:
            print("   mu_y range: All elements are zero.")

        mu_z_nonzero = np.abs(mu_z_Cm[mu_z_Cm != 0])
        if mu_z_nonzero.size > 0:
            print(f"   mu_z range: {np.min(mu_z_nonzero):.3e} to {np.max(mu_z_nonzero):.3e} C·m")
        else:
            print("   mu_z range: All elements are zero.")

    # 3. 電場はそのまま使用（既にV/mの想定）
    Efield_array = efield.get_Efield()
    field_amplitude_V_per_m = np.max(np.abs(Efield_array))
    
    if verbose:
        print(f"📊 Electric field amplitude: {field_amplitude_V_per_m:.3e} V/m")
    
    # 4. 時間ステップの設定
    tlist = efield.tlist
    dt = efield.dt
    
    # 5. SI基本単位に基づいた無次元化スケールの決定
    if verbose:
        print("\n📏 Determining nondimensionalization scales from SI base units...")
    
    scales = determine_SI_based_scales(H0_energy_J, mu_x_Cm, field_amplitude_V_per_m)
    
    # 6. 自動時間ステップ選択
    if auto_timestep:
        if verbose:
            print(f"\n⏱️  Auto-selecting timestep based on λ={scales.lambda_coupling:.3f}...")
        dt_recommended_fs = scales.get_recommended_timestep_fs(
            safety_factor=timestep_safety_factor,
            method=timestep_method
        )
        if verbose:
            print(f"   Recommended dt: {dt_recommended_fs:.3f} fs (method: {timestep_method})")
            print(f"   Original dt: {dt:.3f} fs")
        
        if dt_recommended_fs < dt * 0.5:
            if verbose:
                print(f"   ⚠️  Warning: Recommended dt is much smaller than original")
                print(f"   ⚠️  Consider using dt ≤ {dt_recommended_fs:.3f} fs for stability")
        stride_recommended = int(np.ceil(dt_recommended_fs / dt))
        dt *= stride_recommended
        Efield_array = Efield_array[::stride_recommended]
        tlist = tlist[::stride_recommended]
    
    # 7. 無次元化の実行
    if verbose:
        print("\n🔢 Performing nondimensionalization...")
    
    # エネルギー（ハミルトニアン）の無次元化
    H0_prime = H0_energy_J / scales.E0
    
    # 双極子モーメントの無次元化
    mu_x_prime = mu_x_Cm / scales.mu0
    mu_y_prime = mu_y_Cm / scales.mu0
    mu_z_prime = mu_z_Cm / scales.mu0
    
    # 電場の無次元化
    Efield_prime = Efield_array / scales.Efield0
    try:
        Efield_prime_scalar = efield.get_scalar_and_pol()[0] / scales.Efield0
    except ValueError:
        Efield_prime_scalar = np.zeros_like(Efield_prime)[:, 0]
        
    # 8. 時間軸の無次元化
    tlist_s = tlist * 1e-15  # fs → s
    dt_s = dt * 1e-15  # fs → s
    
    tlist_prime = tlist_s / scales.t0
    dt_prime = dt_s / scales.t0
    
    if verbose:
        print("✓ Nondimensionalization completed successfully!")
        print(f"\n📈 Results:")
        print(f"   λ (coupling strength): {scales.lambda_coupling:.3f}")
        print(f"   dt (dimensionless): {dt_prime:.6f}")
        print(f"   Time points: {len(tlist_prime)}")
    
    return (
        H0_prime,
        mu_x_prime,
        mu_y_prime,
        mu_z_prime,
        Efield_prime,
        Efield_prime_scalar,
        tlist_prime,
        dt_prime,
        scales,
    )


def auto_nondimensionalize(
    hamiltonian: "Hamiltonian",
    dipole_matrix: "DipoleMatrixBase",
    efield: "ElectricField",
    *,
    target_accuracy: str = "standard",
    verbose: bool = True,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    NondimensionalizationScales,
]:
    """
    完全自動無次元化：最適な時間ステップを自動選択
    
    Parameters
    ----------
    hamiltonian : Hamiltonian
        ハミルトニアンオブジェクト
    dipole_matrix : DipoleMatrixBase
        双極子行列オブジェクト
    efield : ElectricField
        電場オブジェクト
    target_accuracy : str, optional
        目標精度 ("high", "standard", "fast"), デフォルト: "standard"
    verbose : bool, optional
        詳細出力の有無, デフォルト: True
        
    Returns
    -------
    tuple
        (H0_prime, mu_x_prime, mu_y_prime, mu_z_prime, Efield_prime, tlist_prime, 
         dt_prime, scales)
    """
    accuracy_settings = {
        "high": {"safety_factor": 0.02, "method": "adaptive"},
        "standard": {"safety_factor": 0.1, "method": "adaptive"},
        "fast": {"safety_factor": 0.3, "method": "stability"},
    }
    
    if target_accuracy not in accuracy_settings:
        raise ValueError(f"target_accuracy must be one of {list(accuracy_settings.keys())}")
    
    settings = accuracy_settings[target_accuracy]
    
    if verbose:
        print(f"🚀 Auto-nondimensionalization (target: {target_accuracy})")
    
    return nondimensionalize_from_objects(
        hamiltonian,
        dipole_matrix,
        efield,
        auto_timestep=True,
        timestep_method=settings["method"],
        timestep_safety_factor=settings["safety_factor"],
        verbose=verbose,
    )


class NondimensionalConverter:
    """高レベル無次元化インターフェース"""

    @staticmethod
    def nondimensionalize_system(
        H0: np.ndarray,
        mu_x: np.ndarray,
        mu_y: np.ndarray,
        efield: "ElectricField",
        **kwargs: Any,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
        NondimensionalizationScales,
    ]:
        """基本的な無次元化を実行"""
        return nondimensionalize_system(H0, mu_x, mu_y, efield, **kwargs)

    @staticmethod
    def nondimensionalize_with_SI_base_units(
        H0: np.ndarray,
        mu_x: np.ndarray,
        mu_y: np.ndarray,
        efield: np.ndarray,
        tlist: np.ndarray,
        **kwargs: Any,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
        NondimensionalizationScales,
    ]:
        """SI基本単位での無次元化を実行"""
        return nondimensionalize_with_SI_base_units(H0, mu_x, mu_y, efield, tlist, **kwargs)

    @staticmethod
    def nondimensionalize_from_objects(
        hamiltonian: "Hamiltonian",
        dipole_matrix: "DipoleMatrixBase",
        efield: "ElectricField",
        **kwargs: Any,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
        NondimensionalizationScales,
    ]:
        """HamiltonianとDipoleMatrixBaseクラスから自動的にSI単位系に変換して無次元化を実行"""
        return nondimensionalize_from_objects(hamiltonian, dipole_matrix, efield, **kwargs)

    @staticmethod
    def auto_nondimensionalize(
        hamiltonian: "Hamiltonian",
        dipole_matrix: "DipoleMatrixBase",
        efield: "ElectricField",
        **kwargs: Any,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
        NondimensionalizationScales,
    ]:
        """完全自動無次元化：最適な時間ステップを自動選択"""
        return auto_nondimensionalize(hamiltonian, dipole_matrix, efield, **kwargs)

    @staticmethod
    def create_dimensionless_time_array(
        scales: NondimensionalizationScales,
        duration_fs: float,
        **kwargs: Any,
    ) -> tuple[np.ndarray, float]:
        """無次元化済みの時間配列を生成"""
        return create_dimensionless_time_array(scales, duration_fs, **kwargs) 