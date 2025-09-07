"""_splitop_schrodinger.py
=================================
Split‑Operator time propagator that mirrors the API of ``rk4_schrodinger.py``
but allows two execution back‑ends:

* **CuPy**  – for GPU acceleration (if ``cupy`` is available and the user passes
  ``backend='cupy'``).
* **NumPy + Numba** – CPU execution with an inner loop compiled by ``@njit`` when
  CuPy is not selected (or not installed).

Only *real* electric‑field envelopes are considered, and Hermiticity of the
interaction Hamiltonian is enforced via
:math:`A = (M + M^\\dagger)/2` with
:math:`M = p_x\\,\\mu_x + p_y\\,\\mu_y`.

The returned trajectory has exactly the same shape as the one produced by
``rk4_schrodinger_traj`` so the two integrators can be swapped freely in user
code.
"""

from __future__ import annotations

from typing import Literal, Union

import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix

# ---------------------------------------------------------------------------
# Optional back‑ends ----------------------------------------------------------
# ---------------------------------------------------------------------------
try:
    import scipy.sparse as sp
except ImportError:
    sp = None

try:
    import cupy as cp  # type: ignore
except ImportError:  # CuPy が無い環境でも読み込めるように動作
    cp = None  # noqa: N816

try:
    from numba import njit  # type: ignore

    _HAS_NUMBA = True
except ImportError:  # NumPy fallback（遅くなるが動く）

    def njit(**_kwargs):  # type: ignore
        """Dummy decorator when numba is absent."""

        def _decorator(func):
            return func

        return _decorator

    _HAS_NUMBA = False

__all__ = ["splitop_schrodinger"]

# ---------------------------------------------------------------------------
# Helper (CPU, Numba) --------------------------------------------------------
# ---------------------------------------------------------------------------


@njit(
    "c16[:, :](c16[:, :], c16[:, :], f8[:], c16[:], c16[:], f8[:], c16, b1, i8)", # type: ignore
    cache=True,
)
def _propagate_numpy(
    U: np.ndarray,  # (dim, dim)  unitary eigenvector matrix
    U_H: np.ndarray,  # U.conj().T  – Hermitian adjoint
    eigvals: np.ndarray,  # (dim,)   eigenvalues of A (real)
    psi0: np.ndarray,  # (dim,)
    exp_half: np.ndarray,  # (dim,)   element‑wise ½‑step phase from H0
    e_mid: np.ndarray,  # (steps,)   midpoint values of E(t)
    phase_coeff: complex,  # −i·2·dt/hbar   (scalar complex)
    return_traj: bool,
    stride: int,
) -> np.ndarray:
    """Numba‑accelerated inner loop (CPU, NumPy)."""

    dim = psi0.shape[0]
    steps = e_mid.size
    n_samples = steps // stride + 1
    traj = np.empty((n_samples, dim), dtype=np.complex128)

    psi = psi0.copy()
    traj[0] = psi
    s_idx = 1

    for k in range(steps):
        # H0 – 前半
        psi *= exp_half

        # Interaction part   exp[ phase_coeff * E * eigvals ]
        phase = np.exp(phase_coeff * e_mid[k] * eigvals)
        psi = U @ (phase * (U_H @ psi))

        # H0 – 後半
        psi *= exp_half

        if return_traj and (k + 1) % stride == 0:
            traj[s_idx] = psi
            s_idx += 1

    if return_traj:
        return traj
    else:
        return psi.reshape((1, dim))


def _propagate_numpy_sparse(
    U: Union[np.ndarray, csr_matrix],  # (dim, dim)  unitary eigenvector matrix
    U_H: Union[np.ndarray, csr_matrix],  # U.conj().T  – Hermitian adjoint
    eigvals: np.ndarray,  # (dim,)   eigenvalues of A (real)
    psi0: np.ndarray,  # (dim,)
    exp_half: np.ndarray,  # (dim,)   element‑wise ½‑step phase from H0
    e_mid: np.ndarray,  # (steps,)   midpoint values of E(t)
    phase_coeff: complex,  # −i·2·dt/hbar   (scalar complex)
    return_traj: bool,
    stride: int,
    renorm: bool = False,
) -> np.ndarray:
    """inner loop (CPU, NumPy, sparse)."""
    if not isinstance(U, csr_matrix):
        U = csr_matrix(U)  # type: ignore
    if not isinstance(U_H, csr_matrix):
        U_H = csr_matrix(U_H)  # type: ignore
    pattern = ((U != 0) + (U_H != 0))
    # Ensure pattern is a sparse matrix with complex dtype
    if not scipy.sparse.issparse(pattern):
        pattern = scipy.sparse.csr_matrix(pattern, dtype=np.complex128)
    else:
        pattern = pattern.astype(np.complex128)  # type: ignore
    pattern.data[:] = 1.0 + 0j
    pattern = pattern.tocsr()  # type: ignore

    # 2️⃣ パターンに合わせてデータを展開
    def expand_to_pattern(matrix: csr_matrix, pattern: csr_matrix) -> np.ndarray:
        result_data = np.zeros_like(pattern.data, dtype=np.complex128)
        m_csr = matrix.tocsr()
        pi, pj = pattern.nonzero()
        m_dict = {(i, j): v for i, j, v in zip(*m_csr.nonzero(), m_csr.data)}
        for idx_, (i, j) in enumerate(zip(pi, pj)):
            result_data[idx_] = m_dict.get((i, j), 0.0 + 0j)
        return result_data

    U_data = expand_to_pattern(U, pattern)  # type: ignore
    U_H_data = expand_to_pattern(U_H, pattern)  # type: ignore
    dim = psi0.shape[0]
    steps = e_mid.size
    n_samples = steps // stride + 1
    traj = np.empty((n_samples, dim), dtype=np.complex128)

    psi = psi0.copy()
    traj[0] = psi
    s_idx = 1

    for k in range(steps):
        # H0 – 前半
        psi *= exp_half

        # Interaction part   exp[ phase_coeff * E * eigvals ]
        phase = np.exp(phase_coeff * e_mid[k] * eigvals)
        psi = U_data @ (phase * (U_H_data @ psi))

        # H0 – 後半
        psi *= exp_half
        if renorm:
            norm = np.sqrt((psi.conj() @ psi).real)
            if norm > 1e-12:
                psi *= 1.0 / norm
            else:
                continue
        if return_traj and (k + 1) % stride == 0:
            traj[s_idx] = psi
            s_idx += 1

    if return_traj:
        return traj
    else:
        return psi.reshape((1, dim))

# ---------------------------------------------------------------------------
# Public API -----------------------------------------------------------------
# ---------------------------------------------------------------------------


def splitop_schrodinger(
    H0: np.ndarray,
    mu_x: np.ndarray,
    mu_y: np.ndarray,
    pol: np.ndarray,
    Efield: np.ndarray,
    psi: np.ndarray,
    dt: float,
    return_traj: bool = True,
    sample_stride: int = 1,
    *,
    backend: Literal["numpy", "cupy"] = "numpy",
    sparse: bool = False,
    renorm: bool = False,
) -> np.ndarray:
    """Split‑Operator propagator with interchangeable back‑ends.

    Parameters
    ----------
    backend : {"numpy", "cupy"}
        Select ``"cupy"`` to run on the GPU (requires CuPy).  Defaults to
        ``"numpy"`` which uses NumPy/Numba.
    """

    steps = (len(Efield) - 1) // 2
    E_mid = Efield[1 : 2 * steps + 1 : 2]
    
    if backend == "cupy":
        if cp is None:
            raise RuntimeError(
                "backend='cupy' was requested but CuPy is not installed."
            )
        return _splitop_cupy(
            H0, mu_x, mu_y, pol, E_mid, psi, dt, return_traj, sample_stride
        )

    # ---------------- CPU / NumPy (+Numba) path ---------------------------
    H0 = np.asarray(H0, dtype=np.float64)

    # スパース行列の場合は適切に処理
    if sp is not None:
        if sp.issparse(mu_x):
            pass  # スパース行列の場合はそのまま使用
        else:
            mu_x = np.asarray(mu_x, dtype=np.complex128)
        if sp.issparse(mu_y):
            pass  # スパース行列の場合はそのまま使用
        else:
            mu_y = np.asarray(mu_y, dtype=np.complex128)
    else:
        mu_x = np.asarray(mu_x, dtype=np.complex128)
        mu_y = np.asarray(mu_y, dtype=np.complex128)

    pol = np.asarray(pol, dtype=np.float64)
    Efield = np.asarray(Efield, dtype=np.float64)
    psi = np.asarray(psi, dtype=np.complex128).flatten()  # 1次元に変換

    # ½‑step phase from diagonal H0
    diag_H0 = np.diag(H0) if H0.ndim == 2 else H0
    exp_half = np.exp(-1j * diag_H0 * dt / (2.0))

    M_raw = np.triu(-pol[0] * mu_x - pol[1] * mu_y, k=1)
    A = (M_raw + M_raw.conj().T)
    eigvals, U = np.linalg.eigh(A)
    U_H = U.conj().T
    # midpoint electric field samples (len = steps)

    phase_coeff = -1j * dt

    if sparse:
        return _propagate_numpy_sparse(
            U, U_H, eigvals, psi, exp_half, E_mid, phase_coeff, return_traj, sample_stride, renorm
        )
    else:
        return _propagate_numpy(
            U, U_H, eigvals, psi, exp_half, E_mid, phase_coeff, return_traj, sample_stride, renorm
        )


# ---------------------------------------------------------------------------
# CuPy back‑end --------------------------------------------------------------
# ---------------------------------------------------------------------------


def _splitop_cupy(
    H0: np.ndarray,
    mu_x: np.ndarray,
    mu_y: np.ndarray,
    pol: np.ndarray,
    E_mid: np.ndarray,
    psi: np.ndarray,
    dt: float,
    return_traj: bool,
    sample_stride: int,
    renorm: bool = False,
):
    """GPU implementation (requires CuPy)."""
    assert cp is not None, "CuPy backend requested but CuPy is not installed."

    # Convert to CuPy arrays once
    H0_cp = cp.asarray(np.diag(H0) if H0.ndim == 2 else H0, dtype=cp.float64)
    mu_x_cp = cp.asarray(mu_x)
    mu_y_cp = cp.asarray(mu_y)
    pol_cp = cp.asarray(pol)
    E_mid_cp = cp.asarray(E_mid)
    psi_cp = cp.asarray(psi)

    exp_half = cp.exp(-1j * H0_cp * dt / (2.0))

    M_raw = -pol_cp[0] * mu_x_cp - pol_cp[1] * mu_y_cp
    A_cp = 0.5 * (M_raw + M_raw.conj().T)

    eigvals, U = cp.linalg.eigh(A_cp)
    U_H = U.conj().T

    # midpoint field samples on GPU
    steps = E_mid_cp.size

    n_samples = steps // sample_stride + 1
    traj = cp.empty((n_samples, psi_cp.size), dtype=cp.complex128)
    traj[0] = psi_cp

    phase_coeff = -1j * dt

    s_idx = 1
    for k in range(steps):
        psi_cp *= exp_half
        phase = cp.exp(phase_coeff * E_mid_cp[k] * eigvals)
        psi_cp = U @ (phase * (U_H @ psi_cp))
        psi_cp *= exp_half
        if renorm:
            norm = np.sqrt((psi_cp.conj() @ psi_cp).real)
            if norm > 1e-12:
                psi_cp *= 1.0 / norm
            else:
                continue
        if return_traj and (k + 1) % sample_stride == 0:
            traj[s_idx] = psi_cp
            s_idx += 1

    if return_traj:
        return cp.asnumpy(traj)
    else:
        return cp.asnumpy(psi_cp)