"""
converter.py
=============
無次元化の変換機能を提供するモジュール。

このモジュールは物理量の無次元化変換を行う実装を含む。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

# スケールクラス
from ..electric_field import ZeroField
from .scales import NondimensionalizationScales
from .utils import _HBAR

# 型ヒント用 (循環参照を避けるため文字列で書く)
if TYPE_CHECKING:  # pragma: no cover
    from rovibrational_excitation.core.basis.hamiltonian import Hamiltonian
    from rovibrational_excitation.core.electric_field import ElectricField
    from rovibrational_excitation.dipole.base import DipoleMatrixBase


def _as_numpy(value: Any) -> np.ndarray:
    """Move an optional backend array to host memory for scale analysis."""
    get = getattr(value, "get", None)
    if callable(get):
        value = get()
    return np.asarray(value)


def _validated_hermitian(matrix: np.ndarray, *, name: str) -> np.ndarray:
    """Validate a finite Hermitian matrix without modifying it."""
    value = np.asarray(matrix)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values")
    scale = float(np.linalg.norm(value, ord=2))
    roundoff = np.finfo(float).eps * max(1, value.shape[0]) * scale
    if not np.allclose(value, value.conj().T, rtol=0.0, atol=roundoff):
        raise ValueError(f"{name} must be Hermitian")
    return value


def _derive_scales(
    H0_energy_J: np.ndarray,
    coupling_dipoles_Cm: np.ndarray,
    field_amplitude_V_per_m: float,
    *,
    explicit_zero_field: bool,
    hbar: float = _HBAR,
    energy_scale_J: float | None = None,
) -> tuple[np.ndarray, NondimensionalizationScales]:
    """Center the free Hamiltonian and derive scales from the full generator."""
    h0 = _validated_hermitian(H0_energy_J, name="H0")
    dipoles = np.asarray(coupling_dipoles_Cm)
    if dipoles.ndim == 2:
        dipoles = dipoles[np.newaxis, ...]
    if dipoles.ndim != 3 or dipoles.shape[1:] != h0.shape:
        raise ValueError("coupling dipoles must be square matrices matching H0")
    if not np.all(np.isfinite(dipoles)):
        raise ValueError("coupling dipoles must contain only finite values")
    for index, matrix in enumerate(dipoles):
        _validated_hermitian(matrix, name=f"coupling dipole {index}")
    if not np.isfinite(field_amplitude_V_per_m) or field_amplitude_V_per_m < 0:
        raise ValueError("field amplitude must be finite and non-negative")

    eigenvalues = np.linalg.eigvalsh(h0)
    energy_offset = float(eigenvalues[0])
    free_energy_span = float(eigenvalues[-1] - eigenvalues[0])
    centered_h0 = h0 - energy_offset * np.eye(h0.shape[0], dtype=h0.dtype)

    dipole_norm = float(
        max((np.linalg.norm(matrix, ord=2) for matrix in dipoles), default=0.0)
    )
    mu0 = dipole_norm if dipole_norm > 0 else None

    if explicit_zero_field:
        if field_amplitude_V_per_m != 0:
            raise ValueError("ZeroField must be identically zero")
        Efield0 = None
        interaction_energy = 0.0
    else:
        if field_amplitude_V_per_m == 0:
            raise ValueError(
                "an identically zero ElectricField is ambiguous; use ZeroField "
                "to request field-free evolution"
            )
        if mu0 is None:
            raise ValueError("a driven problem requires a non-zero coupling dipole")
        Efield0 = float(field_amplitude_V_per_m)
        interaction_energy = Efield0 * mu0

    if energy_scale_J is None:
        E0 = max(free_energy_span, interaction_energy)
        energy_source = "derived"
        energy_method = "max(spectral_span,interaction_operator_norm)"
        if E0 == 0 and energy_offset != 0:
            E0 = abs(energy_offset)
            energy_method = "absolute_identity_offset"
    else:
        if not np.isfinite(energy_scale_J) or energy_scale_J <= 0:
            raise ValueError("energy_scale_J must be finite and positive")
        E0 = float(energy_scale_J)
        energy_source = "explicit"
        energy_method = "caller_supplied"
    if E0 == 0:
        raise ValueError(
            "the generator has no characteristic energy; handle trivial "
            "evolution explicitly at the high-level propagation boundary"
        )

    physical_ratio = (
        interaction_energy / free_energy_span
        if free_energy_span > 0
        else (None if interaction_energy > 0 else 0.0)
    )
    scales = NondimensionalizationScales(
        E0=E0,
        mu0=mu0,
        Efield0=Efield0,
        t0=hbar / E0,
        lambda_coupling=interaction_energy / E0,
        energy_offset=energy_offset,
        free_energy_span=free_energy_span,
        interaction_energy=interaction_energy,
        physical_coupling_ratio=physical_ratio,
        energy_source=energy_source,
        energy_method=energy_method,
    )
    return centered_h0, scales


def nondimensionalize_system(
    H0: np.ndarray,
    mu_x: np.ndarray,
    mu_y: np.ndarray,
    efield: ElectricField,
    *,
    H0_units: str,
    time_units: str,
    dt: float | None = None,
    hbar: float = _HBAR,
    energy_scale_J: float | None = None,
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
    H0_units : str
        H0の単位。"energy" (J) または "frequency" (rad/fs)。
    time_units : str
        時間の単位。"fs" または "s"。
    hbar : float
        プランク定数 [J·s]
    energy_scale_J : float, optional
        Explicit positive reference energy. By default it is derived.

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

    field_array = np.asarray(efield.get_Efield())
    field_amplitude = float(np.max(np.linalg.norm(field_array, axis=1)))
    centered_h0, scales = _derive_scales(
        H0_energy,
        np.stack([mu_x, mu_y]),
        field_amplitude,
        explicit_zero_field=isinstance(efield, ZeroField),
        hbar=hbar,
        energy_scale_J=energy_scale_J,
    )

    H0_prime = centered_h0 / scales.E0
    if scales.mu0 is None:
        mu_x_prime = np.zeros_like(mu_x)
        mu_y_prime = np.zeros_like(mu_y)
    else:
        mu_x_prime = mu_x / scales.mu0
        mu_y_prime = mu_y / scales.mu0
    Efield_prime = (
        np.zeros_like(field_array)
        if scales.Efield0 is None
        else field_array / scales.Efield0
    )

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

    tlist_prime = tlist / scales.t0
    dt_prime = dt_s / scales.t0

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
    *,
    explicit_zero_field: bool = False,
    hbar: float = _HBAR,
    energy_scale_J: float | None = None,
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
    _, scales = _derive_scales(
        H0_energy_J,
        mu_values_Cm,
        field_amplitude_V_per_m,
        explicit_zero_field=explicit_zero_field,
        hbar=hbar,
        energy_scale_J=energy_scale_J,
    )
    return scales


def nondimensionalize_with_SI_base_units(
    H0: np.ndarray,
    mu_x: np.ndarray,
    mu_y: np.ndarray,
    efield: np.ndarray,
    tlist: np.ndarray,
    *,
    explicit_zero_field: bool = False,
    energy_scale_J: float | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    NondimensionalizationScales,
]:
    """Nondimensionalize arrays already expressed in SI base units.

    tlist is in seconds. Its uniform step is preserved exactly.
    """
    time_s = np.asarray(tlist, dtype=np.float64)
    if time_s.ndim != 1 or time_s.size < 2 or not np.all(np.isfinite(time_s)):
        raise ValueError("tlist must be a finite one-dimensional SI-second grid")
    intervals = np.diff(time_s)
    time_roundoff = (
        np.finfo(float).eps
        * max(1, time_s.size)
        * max(float(np.max(np.abs(time_s))), float(np.max(np.abs(intervals))))
    )
    if intervals[0] <= 0 or not np.allclose(
        intervals, intervals[0], rtol=0.0, atol=time_roundoff
    ):
        raise ValueError("tlist must be strictly increasing and uniformly spaced")

    field = np.asarray(efield)
    if field.ndim == 1:
        field_amplitude = float(np.max(np.abs(field)))
    elif field.ndim == 2:
        field_amplitude = float(np.max(np.linalg.norm(field, axis=1)))
    else:
        raise ValueError("efield must be a scalar waveform or component matrix")
    if field.shape[0] != time_s.size:
        raise ValueError("efield and tlist must have the same number of samples")

    centered_h0, scales = _derive_scales(
        np.asarray(H0),
        np.stack([mu_x, mu_y]),
        field_amplitude,
        explicit_zero_field=explicit_zero_field,
        energy_scale_J=energy_scale_J,
    )
    H0_prime = centered_h0 / scales.E0
    if scales.mu0 is None:
        mu_x_prime = np.zeros_like(mu_x)
        mu_y_prime = np.zeros_like(mu_y)
    else:
        mu_x_prime = np.asarray(mu_x) / scales.mu0
        mu_y_prime = np.asarray(mu_y) / scales.mu0
    Efield_prime = (
        np.zeros_like(field) if scales.Efield0 is None else field / scales.Efield0
    )
    return (
        H0_prime,
        mu_x_prime,
        mu_y_prime,
        Efield_prime,
        time_s / scales.t0,
        float(intervals[0] / scales.t0),
        scales,
    )


def create_dimensionless_time_array(
    scales: NondimensionalizationScales,
    duration_fs: float,
    dt_fs: float,
) -> tuple[np.ndarray, float]:
    """Create a dimensionless grid from an explicit physical step.

    The duration must contain an integer number of intervals. The function
    never changes dt_fs or extends the requested endpoint.
    """
    if not np.isfinite(duration_fs) or duration_fs <= 0:
        raise ValueError("duration_fs must be finite and positive")
    if not np.isfinite(dt_fs) or dt_fs <= 0:
        raise ValueError("dt_fs must be finite and positive")

    interval_ratio = duration_fs / dt_fs
    intervals = int(round(interval_ratio))
    roundoff = np.finfo(float).eps * max(1.0, abs(interval_ratio))
    if not np.isclose(interval_ratio, intervals, rtol=0.0, atol=roundoff):
        raise ValueError("duration_fs must be an integer multiple of dt_fs")

    tlist_fs = np.linspace(0.0, duration_fs, intervals + 1)
    t0_fs = scales.t0 * 1e15
    return tlist_fs / t0_fs, dt_fs / t0_fs


def nondimensionalize_from_objects(
    hamiltonian: Hamiltonian,
    dipole_matrix: DipoleMatrixBase,
    efield: ElectricField,
    *,
    coupling_axes: tuple[str, ...],
    scalar_coupling: bool,
    verbose: bool = True,
    energy_scale_J: float | None = None,
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
    coupling_axes : tuple[str, ...]
        Active Cartesian dipole components.
    scalar_coupling : bool
        Whether the field is interpreted as one scalar waveform.
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
    H0_energy_J = _as_numpy(hamiltonian.get_matrix(units="J"))

    if verbose:
        print(f"📊 Hamiltonian matrix: {hamiltonian.shape} in J units")
        if hamiltonian.is_diagonal():
            eigenvals = hamiltonian.get_eigenvalues(units="J")
            print(f"   Eigenvalues: {eigenvals[0]:.3e} to {eigenvals[-1]:.3e} J")

    # 2. DipoleMatrixBaseクラスからSI単位系（C·m）で双極子行列を取得
    mu_x_Cm = _as_numpy(dipole_matrix.get_mu_x_SI(dense=True))
    mu_y_Cm = _as_numpy(dipole_matrix.get_mu_y_SI(dense=True))
    mu_z_Cm = _as_numpy(dipole_matrix.get_mu_z_SI(dense=True))
    dipoles_by_axis = {"x": mu_x_Cm, "y": mu_y_Cm, "z": mu_z_Cm}
    if not coupling_axes or any(axis not in dipoles_by_axis for axis in coupling_axes):
        raise ValueError("coupling_axes must contain only 'x', 'y', or 'z'")
    coupling_dipoles = np.stack(
        [np.asarray(dipoles_by_axis[axis]) for axis in coupling_axes]
    )

    if verbose:
        print(f"📊 Dipole matrices: {mu_x_Cm.shape} in C·m units")

        mu_x_nonzero = np.abs(mu_x_Cm[mu_x_Cm != 0])
        if mu_x_nonzero.size > 0:
            print(
                f"   mu_x range: {np.min(mu_x_nonzero):.3e} to {np.max(mu_x_nonzero):.3e} C·m"
            )
        else:
            print("   mu_x range: All elements are zero.")

        mu_y_nonzero = np.abs(mu_y_Cm[mu_y_Cm != 0])
        if mu_y_nonzero.size > 0:
            print(
                f"   mu_y range: {np.min(mu_y_nonzero):.3e} to {np.max(mu_y_nonzero):.3e} C·m"
            )
        else:
            print("   mu_y range: All elements are zero.")

        mu_z_nonzero = np.abs(mu_z_Cm[mu_z_Cm != 0])
        if mu_z_nonzero.size > 0:
            print(
                f"   mu_z range: {np.min(mu_z_nonzero):.3e} to {np.max(mu_z_nonzero):.3e} C·m"
            )
        else:
            print("   mu_z range: All elements are zero.")

    Efield_array = np.asarray(efield.get_Efield())
    try:
        scalar_field = np.asarray(efield.get_scalar_and_pol()[0])
    except ValueError as exc:
        if scalar_coupling:
            raise ValueError(
                "scalar_coupling requires an ElectricField with an explicit "
                "constant polarization"
            ) from exc
        scalar_field = None

    if scalar_coupling:
        assert scalar_field is not None
        field_amplitude_V_per_m = float(np.max(np.abs(scalar_field)))
    else:
        field_amplitude_V_per_m = float(np.max(np.linalg.norm(Efield_array, axis=1)))

    if verbose:
        print(f"📊 Electric field amplitude: {field_amplitude_V_per_m:.3e} V/m")

    tlist = efield.tlist
    dt = efield.dt
    centered_h0, scales = _derive_scales(
        H0_energy_J,
        coupling_dipoles,
        field_amplitude_V_per_m,
        explicit_zero_field=isinstance(efield, ZeroField),
        energy_scale_J=energy_scale_J,
    )

    if verbose:
        print("\n🔢 Performing nondimensionalization...")

    H0_prime = centered_h0 / scales.E0
    if scales.mu0 is None:
        mu_x_prime = np.zeros_like(mu_x_Cm)
        mu_y_prime = np.zeros_like(mu_y_Cm)
        mu_z_prime = np.zeros_like(mu_z_Cm)
    else:
        mu_x_prime = mu_x_Cm / scales.mu0
        mu_y_prime = mu_y_Cm / scales.mu0
        mu_z_prime = mu_z_Cm / scales.mu0

    if scales.Efield0 is None:
        Efield_prime = np.zeros_like(Efield_array)
        Efield_prime_scalar = (
            None if scalar_field is None else np.zeros_like(scalar_field)
        )
    else:
        Efield_prime = Efield_array / scales.Efield0
        Efield_prime_scalar = (
            None if scalar_field is None else scalar_field / scales.Efield0
        )

    # 8. 時間軸の無次元化
    tlist_s = tlist * 1e-15  # fs → s
    dt_s = dt * 1e-15  # fs → s

    tlist_prime = tlist_s / scales.t0
    dt_prime = dt_s / scales.t0

    if verbose:
        print("✓ Nondimensionalization completed successfully!")
        print("\n📈 Results:")
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
