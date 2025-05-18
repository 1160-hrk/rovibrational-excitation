"""
rovibrational_excitation/core/propagator.py
------------------------------------------
* axes="xy"  → Ex ↔ μ_x,  Ey ↔ μ_y   (デフォルト)
* axes="zx"  → Ex ↔ μ_z,  Ey ↔ μ_x
"""

from __future__ import annotations
from typing import Iterable, Tuple, Union, TYPE_CHECKING

import numpy as np

# ---------------------------------------------------------------------
# optional CuPy
try:
    import cupy as _cp                   # noqa: N811
except ImportError:
    _cp = None                           # type: ignore[assignment]

# ---------------------------------------------------------------------
# type-hints
if TYPE_CHECKING:
    Array = Union[np.ndarray, "_cp.ndarray"]
    from .electric_field import ElectricField
    from linmol_dipole.cache import LinMolDipoleMatrix
else:
    Array = np.ndarray  # runtime dummy

# ---------------------------------------------------------------------
# constants / helpers
_DIRAC_HBAR = 6.62607015e-019 / (2*np.pi)  # J fs

def _cm_to_rad_phz(mu: Array) -> Array:
    """μ (C·m) → rad / (PHz/(V·m⁻¹))."""
    return mu / (_DIRAC_HBAR)      # divide once; xp 対応は呼び出し側で

def _backend(name: str):
    if name == "cupy":
        if _cp is None:
            raise RuntimeError("CuPy backend requested but CuPy not installed")
        return _cp
    return np

def _pick_mu(dip, axis: str) -> Array:
    attr = f"mu_{axis}"
    if not hasattr(dip, attr):
        raise AttributeError(f"{type(dip).__name__} has no attribute '{attr}'")
    return getattr(dip, attr)

# ---------------------------------------------------------------------
def _prepare_args(
    H0: Array,
    E: "ElectricField",
    dip: "LinMolDipoleMatrix",
    *,
    axes: str = "xy",
    dt: float | None = None,
) -> Tuple[Array, Array, Array, Array, Array, float, int]:
    """
    共通前処理

    Returns (順序は旧バージョンと互換):
        H0, μ_a, μ_b, Ex, Ey, dt, steps
        └─ μ_a: Ex に対応 / μ_b: Ey に対応
    """
    axes = axes.lower()
    if len(axes) != 2 or any(a not in "xyz" for a in axes):
        raise ValueError("axes must be like 'xy', 'zx', ...")

    ax0, ax1 = axes
    xp = _cp if _cp is not None else np

    dt_half = E.dt if dt is None else dt / 2
    steps   = E.steps_state

    Ex, Ey  = E.Efield[:, 0], E.Efield[:, 1]

    mu_a = xp.asarray(_cm_to_rad_phz(_pick_mu(dip, ax0)))
    mu_b = xp.asarray(_cm_to_rad_phz(_pick_mu(dip, ax1)))

    return xp.asarray(H0), mu_a, mu_b, xp.asarray(Ex), xp.asarray(Ey), dt_half * 2, steps

# ---------------------------------------------------------------------
# RK4 kernels
from ._rk4_lvne        import rk4_lvne_traj, rk4_lvne
from ._rk4_schrodinger import rk4_schrodinger_traj, rk4_schrodinger

# ---------------------------------------------------------------------
def schrodinger_propagation(
    H0: Array,
    Efield: "ElectricField",
    dipole_matrix: "LinMolDipoleMatrix",
    psi0: Array,
    *,
    axes: str = "xy",
    return_traj: bool = True,
    return_time_psi: bool = True,
    sample_stride: int = 1,
    backend: str = "numpy",
) -> Array:
    xp = _backend(backend)
    H0_, mu_a, mu_b, Ex, Ey, dt, steps = _prepare_args(
        H0, Efield, dipole_matrix, axes=axes
    )

    rk4_args = (H0_, mu_a, mu_b, Ex, Ey, xp.asarray(psi0), dt, steps)
    rk4 = rk4_schrodinger_traj if return_traj else rk4_schrodinger
    if return_traj:
        if return_time_psi:
            dt_psi = dt * 2 * sample_stride
            time_psi = xp.arange(0, (steps+1)*dt_psi, dt_psi)
            return time_psi, rk4(*rk4_args, sample_stride)
        else:
            return rk4(*rk4_args, sample_stride)
    else:
        return rk4(*rk4_args)

# ---------------------------------------------------------------------
def mixed_state_propagation(
    H0: Array,
    Efield: "ElectricField",
    psi0_array: Iterable[Array],
    dipole_matrix: "LinMolDipoleMatrix",
    *,
    axes: str = "xy",
    return_traj: bool = True,
    return_time_rho: bool = True,
    sample_stride: int = 1,
    backend: str = "numpy",
) -> Array:
    xp = _backend(backend)
    dim = psi0_array[0].shape[0]
    steps_out = (len(Efield.tlist) // 2) // sample_stride + 1
    rho_out = xp.zeros((steps_out, dim, dim)) if return_traj else xp.zeros((dim, dim))

    for psi0 in psi0_array:
        psi_t = schrodinger_propagation(
            H0, Efield, dipole_matrix, psi0,
            axes=axes, return_traj=return_traj,
            sample_stride=sample_stride, backend=backend
        )
        if return_traj:
            rho_out += xp.einsum("ti, tj -> tij", psi_t, psi_t.conj())
        else:
            rho_out += psi_t @ psi_t.conj().T
    if return_traj:
        if return_time_rho:
            dt_rho = Efield.dt_state * sample_stride
            steps = Efield.steps_state
            time_psi = xp.arange(0, (steps+1)*dt_rho, dt_rho)
            return time_psi, rho_out
        else:
            return rho_out
    else:
        return rho_out

# ---------------------------------------------------------------------
def liouville_propagation(
    H0: Array,
    Efield: "ElectricField",
    dipole_matrix: "LinMolDipoleMatrix",
    rho0: Array,
    *,
    axes: str = "xy",
    return_traj: bool = True,
    sample_stride: int = 1,
    backend: str = "numpy",
) -> Array:
    xp = _backend(backend)
    H0_, mu_a, mu_b, Ex, Ey, dt, steps = _prepare_args(
        H0, Efield, dipole_matrix, axes=axes
    )

    rk4_args = (H0_, mu_a, mu_b, Ex, Ey, xp.asarray(rho0), dt, steps)
    rk4 = rk4_lvne_traj if return_traj else rk4_lvne
    return rk4(*rk4_args, sample_stride) if return_traj else rk4(*rk4_args)
