"""Validated public wrappers for dense NumPy Liouville RK4 propagation."""

from __future__ import annotations

import numpy as np

from ....core.validation import validate_density_matrix_problem
from .liouville_numpy import rk4_liouville_numpy_dense


# ------------------------------------------------------------
# 公開 API
# ------------------------------------------------------------
def rk4_lvne_traj(
    H0,
    mu_x,
    mu_y,
    Efield_x,
    Efield_y,
    rho0,
    dt: float,
    steps: int,
    sample_stride: int = 1,
) -> np.ndarray:
    """
    軌跡を返す版  ―  shape = (steps//sample_stride+1, dim, dim)
    """
    validate_density_matrix_problem(
        H0,
        (mu_x, mu_y),
        (Efield_x, Efield_y),
        rho0,
        dt=dt,
        stride=sample_stride,
        backend="numpy",
        require_odd_field=True,
    )
    expected_steps = (len(Efield_x) - 1) // 2
    if steps != expected_steps:
        raise ValueError(f"steps must be {expected_steps} for the supplied field grid")
    return rk4_liouville_numpy_dense(
        np.ascontiguousarray(H0, dtype=np.complex128),
        np.ascontiguousarray(mu_x, dtype=np.complex128),
        np.ascontiguousarray(mu_y, dtype=np.complex128),
        np.asarray(Efield_x, dtype=np.float64),
        np.asarray(Efield_y, dtype=np.float64),
        np.ascontiguousarray(rho0, dtype=np.complex128),
        float(dt),
        int(steps),
        int(sample_stride),
        True,  # record_traj
    )


def rk4_lvne(
    H0,
    mu_x,
    mu_y,
    Efield_x,
    Efield_y,
    rho0,
    dt: float,
    steps: int,
) -> np.ndarray:
    """
    最終密度行列だけ返す軽量版  ―  shape = (dim, dim)
    """
    validate_density_matrix_problem(
        H0,
        (mu_x, mu_y),
        (Efield_x, Efield_y),
        rho0,
        dt=dt,
        stride=1,
        backend="numpy",
        require_odd_field=True,
    )
    expected_steps = (len(Efield_x) - 1) // 2
    if steps != expected_steps:
        raise ValueError(f"steps must be {expected_steps} for the supplied field grid")
    traj = rk4_liouville_numpy_dense(
        np.ascontiguousarray(H0, dtype=np.complex128),
        np.ascontiguousarray(mu_x, dtype=np.complex128),
        np.ascontiguousarray(mu_y, dtype=np.complex128),
        np.asarray(Efield_x, dtype=np.float64),
        np.asarray(Efield_y, dtype=np.float64),
        np.ascontiguousarray(rho0, dtype=np.complex128),
        float(dt),
        int(steps),
        1,  # stride (dummy)
        False,  # record_traj
    )
    return traj[0]  # (1, dim, dim) → (dim, dim)
