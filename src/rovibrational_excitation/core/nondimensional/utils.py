"""
utils.py
========
無次元化に関する共通ユーティリティ関数と定数を提供する。

このモジュールは他のモジュールから共通して使用される基本的な機能を
提供し、依存関係を最小限に抑える。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
DEFAULT_TO_SI_CONVERSIONS: dict[str, float] = {
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

    print("🔄 Converting default units to SI base units:")
    print(f"   Frequency: {frequency_cm_inv:.3f} cm⁻¹ → {frequency_rad_per_s:.6e} rad/s")
    print(f"   Dipole: {dipole_D:.3f} D → {dipole_Cm:.6e} C·m")
    print(f"   Field: {field_MV_per_cm:.3f} MV/cm → {field_V_per_m:.6e} V/m")
    print(f"   Energy: {energy_eV:.3f} eV → {energy_J:.6e} J")
    print(f"   Time: {time_fs:.3f} fs → {time_s:.6e} s")

    return frequency_rad_per_s, dipole_Cm, field_V_per_m, energy_J, time_s


def get_energy_scale_from_hamiltonian(
    H0: np.ndarray,
    hbar: float = _HBAR,
) -> float:
    """
    ハミルトニアンからエネルギースケールを計算
    
    Parameters
    ----------
    H0 : np.ndarray
        ハミルトニアン行列（J）
    hbar : float, optional
        プランク定数 [J·s]
        
    Returns
    -------
    float
        エネルギースケール [J]
    """
    del hbar
    matrix = np.asarray(H0)
    if matrix.ndim == 2:
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("H0 must be square")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("H0 must contain only finite values")
        if not np.array_equal(matrix, matrix.conj().T):
            raise ValueError("H0 must be exactly Hermitian at the scaling boundary")
        eigvals = np.linalg.eigvalsh(matrix)
    elif matrix.ndim == 1:
        eigvals = matrix
    else:
        raise ValueError("H0 must be a square matrix or eigenvalue vector")
    span = float(np.max(eigvals) - np.min(eigvals))
    if span <= 0.0:
        raise ValueError("H0 has no non-zero spectral span")
    return span


def get_dipole_scale_from_matrices(
    mu_x: np.ndarray,
    mu_y: np.ndarray,
) -> float:
    """
    双極子行列からスケールを計算
    
    Parameters
    ----------
    mu_x, mu_y : np.ndarray
        双極子行列
        
    Returns
    -------
    float
        双極子スケール [C·m]
    """
    matrices = [np.asarray(mu_x), np.asarray(mu_y)]
    if any(not np.all(np.isfinite(matrix)) for matrix in matrices):
        raise ValueError("dipole matrices must contain only finite values")
    return float(max(np.linalg.norm(matrix, ord=2) for matrix in matrices))


def get_electric_field_scale(
    efield: ElectricField,
) -> float:
    """
    電場オブジェクトからスケールを計算
    
    Parameters
    ----------
    efield : ElectricField
        電場オブジェクト
        
    Returns
    -------
    float
        電場スケール [V/m]
    """
    Efield_array = np.asarray(efield.get_Efield())  # (T, 2) [V/m]
    if not np.all(np.isfinite(Efield_array)):
        raise ValueError("electric field must contain only finite values")
    field_magnitudes = np.sqrt(Efield_array[:, 0] ** 2 + Efield_array[:, 1] ** 2)
    return float(np.max(field_magnitudes))


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


def create_SI_demo_parameters() -> dict[str, Any]:
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
