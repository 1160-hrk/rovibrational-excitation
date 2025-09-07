from __future__ import annotations

from typing import Literal, Tuple
import numpy as np

try:
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover
    cp = None

from ..units.constants import CONSTANTS


def _fft_forward(a, xp):
    if xp is np:
        return np.fft.fftn(a, axes=tuple(range(a.ndim)))
    else:
        return cp.fft.fftn(a, axes=tuple(range(a.ndim)))  # type: ignore


def _fft_inverse(A, xp):
    if xp is np:
        return np.fft.ifftn(A, axes=tuple(range(A.ndim)))
    else:
        return cp.fft.ifftn(A, axes=tuple(range(A.ndim)))  # type: ignore


def _exp(i_arg, xp):
    return xp.exp(i_arg)


def _asxp(a, xp):
    return a if xp is np else cp.asarray(a)  # type: ignore


def splitop_realspace(
    psi0: np.ndarray,
    grid_axes: Tuple[np.ndarray, ...],
    m: float,
    dt: float,
    steps: int,
    *,
    V: np.ndarray | None = None,
    mu_x: np.ndarray | None = None,
    E_t: np.ndarray | None = None,
    backend: Literal["numpy", "cupy"] = "numpy",
    return_traj: bool = True,
    sample_stride: int = 1,
    hbar: float | None = None,
) -> np.ndarray:
    """
    Real-space split-operator propagation for H = T + V(r) - E(t) mu_x(r).

    Uses second-order splitting per step:
        psi <- e^{-i V_eff dt/2} F^{-1} [ e^{-i T dt} F[ e^{-i V_eff dt/2} psi ] ]
    with V_eff(r, t) = V(r) - E(t_mid) mu_x(r).

    Parameters
    ----------
    psi0 : ndarray (Nx[,Ny[,Nz]]) complex
        Initial wavefunction on Cartesian grid.
    grid_axes : tuple of arrays
        (x,), (x,y), or (x,y,z) defining uniform grids in each axis.
    m : float
        Mass for kinetic operator.
    dt : float
        Time step.
    steps : int
        Number of split steps.
    V : ndarray, optional
        Static potential on the same grid (broadcastable).
    mu_x : ndarray, optional
        Real-space dipole along x on the same grid (broadcastable).
    E_t : ndarray, optional
        Scalar electric field samples with length 2*steps+1 or steps.
        If length is 2*steps+1, midpoint sampling is used.
    backend : {"numpy", "cupy"}
        Computational backend. CuPy requires cupy installed.
    return_traj : bool
        Return sampled trajectory if True, else only final state.
    sample_stride : int
        Sampling stride for trajectory.
    hbar : float, optional
        Planck constant over 2π; defaults to SI value.
    """
    if backend == "cupy" and cp is None:
        raise RuntimeError("backend='cupy' requested but CuPy is not installed")

    xp = np if backend == "numpy" else cp  # type: ignore
    hbar = CONSTANTS.HBAR if hbar is None else float(hbar)

    dim = len(grid_axes)
    if dim < 1 or dim > 3:
        raise ValueError("grid_axes must be a tuple of 1, 2, or 3 arrays")

    psi = _asxp(psi0, xp).astype(xp.complex128)

    # Construct k-space grids and kinetic phase
    if dim == 1:
        x = grid_axes[0]
        Nx = x.size
        dx = float(x[1] - x[0])
        kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
        k2 = _asxp(kx**2, xp)
        k2 = k2.reshape((Nx,))
    elif dim == 2:
        x, y = grid_axes  # type: ignore
        Nx, Ny = x.size, y.size
        dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
        kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
        ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        k2 = _asxp(KX**2 + KY**2, xp)
    else:
        x, y, z = grid_axes  # type: ignore
        Nx, Ny, Nz = x.size, y.size, z.size
        dx, dy, dz = float(x[1] - x[0]), float(y[1] - y[0]), float(z[1] - z[0])
        kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
        ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
        kz = 2 * np.pi * np.fft.fftfreq(Nz, d=dz)
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
        k2 = _asxp(KX**2 + KY**2 + KZ**2, xp)

    T_phase = _exp(-1j * dt * (hbar**2) * k2 / (2.0 * m), xp)

    # Prepare static parts
    V_grid = _asxp(0.0, xp)
    if V is not None:
        V_grid = _asxp(V, xp)
    mu_grid = _asxp(0.0, xp)
    if mu_x is not None:
        mu_grid = _asxp(mu_x, xp)

    # Handle field sampling (midpoints if available)
    if E_t is None:
        E_mid = _asxp(np.zeros((steps,), dtype=np.float64), xp)
    else:
        E_t = np.asarray(E_t)
        if E_t.size == steps:
            E_mid = _asxp(E_t, xp)
        elif E_t.size == 2 * steps + 1:
            E_mid = _asxp(E_t[1 : 2 * steps + 1 : 2], xp)
        else:
            raise ValueError("E_t length must be steps or 2*steps+1 for midpoint rule")

    n_samples = steps // sample_stride + 1
    if return_traj:
        traj = xp.empty((n_samples,) + psi.shape, dtype=xp.complex128)
        traj[0] = psi
    s_idx = 1

    for k in range(steps):
        V_eff = V_grid - E_mid[k] * mu_grid
        expV_half = _exp(-1j * V_eff * (dt / 2.0) / hbar, xp)

        psi = expV_half * psi
        psi_k = _fft_forward(psi, xp)
        psi_k = T_phase * psi_k
        psi = _fft_inverse(psi_k, xp)
        psi = expV_half * psi

        if return_traj and ((k + 1) % sample_stride == 0):
            traj[s_idx] = psi
            s_idx += 1

    if return_traj:
        return traj.get() if xp is not np else traj  # type: ignore
    else:
        return psi.get() if xp is not np else psi  # type: ignore

