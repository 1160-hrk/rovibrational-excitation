"""
utils.py
========
無次元化に関する共通ユーティリティ関数と定数を提供する。

このモジュールは他のモジュールから共通して使用される基本的な機能を
提供し、依存関係を最小限に抑える。
"""

from __future__ import annotations

from typing import Dict, Any, TYPE_CHECKING
import numpy as np

from rovibrational_excitation.core.units.constants import CONSTANTS

if TYPE_CHECKING:
    from rovibrational_excitation.core.electric_field import ElectricField

# 物理定数
_HBAR = CONSTANTS.HBAR
_C = CONSTANTS.C  # Speed of light [m/s]
_EV_TO_J = CONSTANTS.EV_TO_J  # eV → J
_DEBYE_TO_CM = CONSTANTS.DEBYE_TO_CM  # D → C·m

# デフォルト単位からSI基本単位への変換係数
DEFAULT_TO_SI_CONVERSIONS: Dict[str, float] = {
    # Frequency: cm⁻¹ → rad/s
    "frequency_cm_inv_to_rad_per_s": 2 * np.pi * _C * 100,
    # Dipole moment: D → C·m
    "dipole_D_to_Cm": _DEBYE_TO_CM,
    # Electric field: MV/cm → V/m
    "field_MV_per_cm_to_V_per_m": 1e8,
    # Energy: eV → J
    "energy_eV_to_J": _EV_TO_J,
    # Time: fs → s
    "time_fs_to_s": 1e-15,
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


def get_energy_scale_from_hamiltonian(
    H0: np.ndarray,
    max_time_scale_fs: float = 1000.0,
    hbar: float = _HBAR,
) -> float:
    """
    ハミルトニアンからエネルギースケールを計算
    
    Parameters
    ----------
    H0 : np.ndarray
        ハミルトニアン行列（J）
    max_time_scale_fs : float, optional
        時間スケール上限 [fs]
    hbar : float, optional
        プランク定数 [J·s]
        
    Returns
    -------
    float
        エネルギースケール [J]
    """
    if H0.ndim == 2:
        eigvals = np.diag(H0)
    else:
        eigvals = H0.copy()
    
    # 最大エネルギー差を計算
    energy_diffs = np.abs(eigvals[:, None] - eigvals[None, :])
    # 対角成分（自分自身との差=0）を除外
    energy_diffs_nonzero = energy_diffs[energy_diffs > 0]
    
    if len(energy_diffs_nonzero) == 0:
        # すべて縮退している場合、最大エネルギー値をスケールとして使用
        E0 = np.max(np.abs(eigvals))
        if E0 == 0:
            E0 = hbar / 1e-15  # 最終的なフォールバック
    else:
        E0 = np.max(energy_diffs_nonzero)
    
    # 時間スケールが大きすぎる場合は上限を適用
    t0 = hbar / E0
    max_time_scale_s = max_time_scale_fs * 1e-15
    if t0 > max_time_scale_s:
        t0 = max_time_scale_s
        E0 = hbar / t0  # エネルギースケールを再調整
    
    return E0


def get_dipole_scale_from_matrices(
    mu_x: np.ndarray,
    mu_y: np.ndarray,
    default_scale: float = _DEBYE_TO_CM,
) -> float:
    """
    双極子行列からスケールを計算
    
    Parameters
    ----------
    mu_x, mu_y : np.ndarray
        双極子行列
    default_scale : float, optional
        デフォルトスケール [C·m]
        
    Returns
    -------
    float
        双極子スケール [C·m]
    """
    mu_x_offdiag = mu_x.copy()
    mu_y_offdiag = mu_y.copy()
    
    if mu_x.ndim == 2:
        np.fill_diagonal(mu_x_offdiag, 0)
    if mu_y.ndim == 2:
        np.fill_diagonal(mu_y_offdiag, 0)
    
    mu0 = max(np.max(np.abs(mu_x_offdiag)), np.max(np.abs(mu_y_offdiag)))
    
    if mu0 == 0:
        mu0 = default_scale
    
    return mu0


def get_electric_field_scale(
    efield: "ElectricField",
    default_scale: float = 1e8,
) -> float:
    """
    電場オブジェクトからスケールを計算
    
    Parameters
    ----------
    efield : ElectricField
        電場オブジェクト
    default_scale : float, optional
        デフォルトスケール [V/m]
        
    Returns
    -------
    float
        電場スケール [V/m]
    """
    Efield_array = efield.get_Efield()  # (T, 2) [V/m]
    field_magnitudes = np.sqrt(Efield_array[:, 0]**2 + Efield_array[:, 1]**2)
    Efield0 = np.max(field_magnitudes)
    
    if Efield0 == 0:
        Efield0 = default_scale
    
    return Efield0


def dimensionalize_wavefunction(
    psi_prime: np.ndarray,
    scales: Any,
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
    scales: Any,
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