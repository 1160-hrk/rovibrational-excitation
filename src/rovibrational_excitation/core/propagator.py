"""
propagator.py  ―  軸マッピング機能付き
-------------------------------------
* `axes` キーワードを追加  
    - 例 `"xy"`  →  Ex ↔ μ_x,  Ey ↔ μ_y  (従来どおり)  
    - 例 `"zx"`  →  Ex ↔ μ_z,  Ey ↔ μ_x
* LinMolDipoleMatrix に **mu_z** が定義されていることを前提
  （無い場合は AttributeError を投げて知らせる）
"""

from __future__ import annotations
import numpy as np
from typing import Iterable, Tuple, Union
try:
    import cupy as _cp
except ImportError:
    _cp = None

Array = Union[np.ndarray, "cupy.ndarray"]


# ----------------------------------------------------------------------
# 内部ユーティリティ
# ----------------------------------------------------------------------

def _backend(xp: str):
    if xp == "cupy":
        if _cp is None:
            raise RuntimeError("CuPy not installed")
        return _cp
    return np


def _pick_mu(dip, axis: str) -> Array:
    """'x','y','z' の文字から双極子行列を取得"""
    attr = f"mu_{axis}"
    if not hasattr(dip, attr):
        raise AttributeError(f"{type(dip).__name__} has no attribute '{attr}'")
    return getattr(dip, attr)


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
    Ex/Ey と双極子行列 μa/μb の **対応** を axes で指定する。
    """
    if len(axes) != 2 or any(a not in "xyz" for a in axes.lower()):
        raise ValueError("axes must be like 'xy', 'zx', etc.")
    ax0, ax1 = axes.lower()

    use_cupy = (_cp is not None) and isinstance(H0, _cp.ndarray)
    xp = _cp if use_cupy else np

    dt_half = E.dt if dt is None else dt / 2
    steps   = E.steps_state

    Ex, Ey = E.Efield[:, 0], E.Efield[:, 1]
    mu_a   = _pick_mu(dip, ax0)
    mu_b   = _pick_mu(dip, ax1)

    return (
        H0,
        mu_a,           # ← Ex に対応する μ
        mu_b,           # ← Ey に対応する μ
        xp.asarray(Ex),
        xp.asarray(Ey),
        dt_half * 2,
        steps,
    )


# ----------------------------------------------------------------------
# 伝搬ラッパ
# ----------------------------------------------------------------------

from ._rk4_lvne import rk4_lvne_traj, rk4_lvne
from ._rk4_schrodinger import rk4_schrodinger_traj, rk4_schrodinger
from .electric_field import ElectricField
from .dipole_matrix import LinMolDipoleMatrix


def schrodinger_propagation(
    H0: Array,
    Efield: ElectricField,
    dipole_matrix: LinMolDipoleMatrix,
    psi0: Array,
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

    rk4_args = (
        H0_, mu_a, mu_b,
        Ex, Ey,
        xp.asarray(psi0),
        dt, steps,
    )

    rk4 = rk4_schrodinger_traj if return_traj else rk4_schrodinger
    return rk4(*rk4_args, sample_stride) if return_traj else rk4(*rk4_args)


def mixed_state_propagation(
    H0: Array,
    Efield: ElectricField,
    psi0_array: Iterable[Array],
    dipole_matrix: LinMolDipoleMatrix,
    *,
    axes: str = "xy",
    return_traj: bool = True,
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
            axes=axes,
            return_traj=return_traj,
            sample_stride=sample_stride,
            backend=backend,
        )
        if return_traj:
            rho_out += xp.einsum("t i, t j -> t i j", psi_t, psi_t.conj())
        else:
            rho_out += psi_t @ psi_t.conj().T
    return rho_out


def liouville_propagation(
    H0: Array,
    Efield: ElectricField,
    dipole_matrix: LinMolDipoleMatrix,
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

    rk4_args = (
        H0_, mu_a, mu_b,
        Ex, Ey,
        xp.asarray(rho0),
        dt, steps,
    )

    rk4 = rk4_lvne_traj if return_traj else rk4_lvne
    return rk4(*rk4_args, sample_stride) if return_traj else rk4(*rk4_args)
