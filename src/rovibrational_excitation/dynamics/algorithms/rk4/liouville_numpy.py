"""Prevalidated dense NumPy/Numba kernel for Liouville RK4 propagation."""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(
    "c16[:, :, :](c16[:, :], c16[:, :], c16[:, :],"
    "f8[:], f8[:],"
    "c16[:, :], f8, i8, i8, b1)",
    cache=True,
    fastmath=True,
)  # type: ignore[untyped-decorator]
def rk4_liouville_numpy_dense(
    H0: np.ndarray,
    mu_x: np.ndarray,
    mu_y: np.ndarray,
    Ex: np.ndarray,
    Ey: np.ndarray,
    rho0: np.ndarray,
    dt: float,
    steps: int,
    stride: int,
    record_traj: bool,
) -> np.ndarray:
    """Run the unchanged complex128 dense kernel on prepared numeric arrays."""
    dim = rho0.shape[0]
    n_out = steps // stride + 1 if record_traj else 1
    traj = np.empty((n_out, dim, dim), np.complex128)

    rho = rho0.copy()
    traj[0] = rho

    buf = np.empty_like(rho)
    out_idx = 1

    for s in range(steps):
        idx = 2 * s
        ex1 = Ex[idx]
        ex2 = Ex[idx + 1]
        ex4 = Ex[idx + 2]

        ey1 = Ey[idx]
        ey2 = Ey[idx + 1]
        ey4 = Ey[idx + 2]

        H1 = H0 - mu_x * ex1 - mu_y * ey1
        H2 = H0 - mu_x * ex2 - mu_y * ey2
        H4 = H0 - mu_x * ex4 - mu_y * ey4

        k1 = -1j * (H1 @ rho - rho @ H1)

        buf[:, :] = rho + 0.5 * dt * k1
        k2 = -1j * (H2 @ buf - buf @ H2)

        buf[:, :] = rho + 0.5 * dt * k2
        k3 = -1j * (H2 @ buf - buf @ H2)

        buf[:, :] = rho + dt * k3
        k4 = -1j * (H4 @ buf - buf @ H4)

        rho += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        if record_traj and ((s + 1) % stride == 0):
            traj[out_idx] = rho
            out_idx += 1

    if not record_traj:
        traj[0] = rho

    return traj


__all__ = ["rk4_liouville_numpy_dense"]
