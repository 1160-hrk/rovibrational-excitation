# シュレディンガー・リウヴィル時間発展
# propagator.py
import numpy as np
from typing import Iterable, Tuple, Union
try:
    import cupy as _cp
except ImportError:
    _cp = None

Array = Union[np.ndarray, "cupy.ndarray"]

def _backend(xp: str):
    if xp == "cupy":
        if _cp is None:
            raise RuntimeError("CuPy not installed")
        return _cp
    return np

from ._rk4_lvne import rk4_lvne_traj, rk4_lvne
from ._rk4_schrodinger import rk4_schrodinger_traj, rk4_schrodinger
from .electric_field import ElectricField
from .dipole_matrix import LinMolDipoleMatrix

def _prepare_args(
    H0: Array, E: ElectricField, dip: LinMolDipoleMatrix, dt: float | None = None
) -> Tuple[Array, Array, Array, Array, Array, float, int]:
    """共通前処理：電場を x, y に分割し dt, steps を返す"""
    use_cupy = (_cp is not None) and isinstance(H0, _cp.ndarray)
    xp = _cp if use_cupy else np
    dt_half = E.dt if dt is None else dt / 2
    steps = E.steps_state
    Ex, Ey = E.Efield[:, 0], E.Efield[:, 1]
    return H0, dip.mu_x, dip.mu_y, xp.asarray(Ex), xp.asarray(Ey), dt_half * 2, steps

def schrodinger_propagation(
    H0: Array,
    Efield: ElectricField,
    dipole_matrix: LinMolDipoleMatrix,
    psi0: Array,
    *,
    return_traj: bool = True,
    sample_stride: int = 1,
    backend: str = "numpy",
) -> Array:
    xp = _backend(backend)
    H0_, mux_, muy_, Ex, Ey, dt, steps = _prepare_args(H0, Efield, dipole_matrix)
    rk4_args = (
        H0_, mux_, muy_,
        Ex, Ey,
        xp.asarray(psi0),
        dt, steps,
    )
    rk4 = rk4_schrodinger_traj if return_traj else rk4_schrodinger
    return rk4(*rk4_args, sample_stride) if return_traj else rk4(*rk4_args)


def mixed_state_propagation(
    H0:np.ndarray,
    Efield:ElectricField,
    psi0_array:list,
    dipole_matrix:LinMolDipoleMatrix,
    return_traj:bool=True,
    sample_stride:int=1
    ) -> np.ndarray:
    dim = psi0_array[0].shape[0]
    steps = (len(Efield.tlist) //2) // sample_stride + 1
    if return_traj:
        result_rho = np.zeros((steps, dim, dim))
    else:
        result_rho = np.zeros((dim, dim))
    for psi0 in psi0_array:
        result_psi = schrodinger_propagation(H0, Efield, dipole_matrix, psi0, return_traj=return_traj, sample_stride=sample_stride)
        if return_traj:
            result_rho += np.matmul(result_psi, np.conj(np.transpose(result_psi, axes=(0, 2, 1))))
        else:
            result_rho += result_psi @ result_psi.conj().T
    return result_rho


def liouville_propagation(
    H0: Array,
    Efield: ElectricField,
    dipole_matrix: LinMolDipoleMatrix,
    rho0: Array,
    *,
    return_traj: bool = True,
    sample_stride: int = 1,
    backend: str = "numpy",
) -> Array:
    xp = _backend(backend)
    H0_, mux_, muy_, Ex, Ey, dt, steps = _prepare_args(H0, Efield, dipole_matrix)
    rk4_args = (
        H0_, mux_, muy_,
        Ex, Ey,
        xp.asarray(rho0),
        dt, steps,
    )
    rk4 = rk4_lvne_traj if return_traj else rk4_lvne
    return rk4(*rk4_args, sample_stride) if return_traj else rk4(*rk4_args)