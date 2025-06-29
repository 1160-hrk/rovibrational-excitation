from __future__ import annotations
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

from typing import Literal

import numpy as np

# ---------------------------------------------------------------------------
# Optional back‑ends ----------------------------------------------------------
# ---------------------------------------------------------------------------
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

@njit(fastmath=True, cache=True)
def _propagate_numpy(
    U: np.ndarray,  # (dim, dim)  unitary eigenvector matrix
    U_H: np.ndarray,  # U.conj().T  – Hermitian adjoint
    eigvals: np.ndarray,  # (dim,)   eigenvalues of A (real)
    psi0: np.ndarray,  # (dim,)
    exp_half: np.ndarray,  # (dim,)   element‑wise ½‑step phase from H0
    e_mid: np.ndarray,  # (steps,)   midpoint values of E(t)
    phase_coeff: complex,  # −i·2·dt/hbar   (scalar complex)
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

        if (k + 1) % stride == 0:
            traj[s_idx] = psi
            s_idx += 1

    return traj


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
    steps: int,
    sample_stride: int = 1,
    hbar: float = 1.0,
    *,
    backend: Literal["numpy", "cupy"] = "numpy",
) -> np.ndarray:
    """Split‑Operator propagator with interchangeable back‑ends.

    Parameters
    ----------
    backend : {"numpy", "cupy"}
        Select ``"cupy"`` to run on the GPU (requires CuPy).  Defaults to
        ``"numpy"`` which uses NumPy/Numba.
    """

    if backend == "cupy":
        if cp is None:
            raise RuntimeError("backend='cupy' was requested but CuPy is not installed.")
        return _splitop_cupy(
            H0, mu_x, mu_y, pol, Efield, psi, dt, steps, sample_stride, hbar
        )

    # ---------------- CPU / NumPy (+Numba) path ---------------------------
    H0 = np.asarray(H0, dtype=np.float64)
    
    # スパース行列の場合は適切に処理
    try:
        import scipy.sparse as sp
        if sp.issparse(mu_x):
            pass  # スパース行列の場合はそのまま使用
        else:
            mu_x = np.asarray(mu_x, dtype=np.complex128)
        if sp.issparse(mu_y):
            pass  # スパース行列の場合はそのまま使用
        else:
            mu_y = np.asarray(mu_y, dtype=np.complex128)
    except ImportError:
        mu_x = np.asarray(mu_x, dtype=np.complex128)
        mu_y = np.asarray(mu_y, dtype=np.complex128)
    
    pol = np.asarray(pol, dtype=np.complex128)
    Efield = np.asarray(Efield, dtype=np.float64)
    psi = np.asarray(psi, dtype=np.complex128).flatten()  # 1次元に変換

    # ½‑step phase from diagonal H0
    diag_H0 = np.diag(H0) if H0.ndim == 2 else H0
    exp_half = np.exp(-1j * diag_H0 * dt / (2.0 * hbar))

    # Hermitian A = (M+M†)/2  with  M = p·μ
    # スパース行列の場合は適切に処理
    try:
        import scipy.sparse as sp
        if sp.issparse(mu_x) or sp.issparse(mu_y):
            # スパース行列の演算
            M_raw = pol[0] * mu_x + pol[1] * mu_y
            if sp.issparse(M_raw):
                A = 0.5 * (M_raw + M_raw.getH())  # getH() は共役転置
                # 小サイズの場合はdenseに変換して固有値分解（メモリ効率が良い）
                if A.shape[0] <= 100:
                    A_dense = A.toarray()
                    eigvals, U = np.linalg.eigh(A_dense)
                else:
                    # 大サイズの場合はdenseでの固有値分解にフォールバック
                    A_dense = A.toarray()
                    eigvals, U = np.linalg.eigh(A_dense)
            else:
                A = 0.5 * (M_raw + M_raw.conj().T)
                eigvals, U = np.linalg.eigh(A)
        else:
            # Dense行列の場合（従来の処理）
            M_raw = pol[0] * mu_x + pol[1] * mu_y
            A = 0.5 * (M_raw + M_raw.conj().T)
            eigvals, U = np.linalg.eigh(A)  # Hermitian, so eigh is fine
    except ImportError:
        # scipy が無い場合は従来の処理
        M_raw = pol[0] * mu_x + pol[1] * mu_y
        A = 0.5 * (M_raw + M_raw.conj().T)
        eigvals, U = np.linalg.eigh(A)  # Hermitian, so eigh is fine
    U_H = U.conj().T

    # midpoint electric field samples (len = steps)
    E_mid = Efield[1 : 2 * steps + 1 : 2]

    # phase_coeff = -1j * 2.0 * dt / hbar
    phase_coeff = -1j * dt / hbar # 2のファクターは必要ない。cosineを複素電場で表現するときにつく1/2で相殺される。

    traj = _propagate_numpy(U, U_H, eigvals, psi, exp_half, E_mid, phase_coeff, sample_stride)

    # reshape to (n_samples, dim, 1) to match rk4
    return traj.reshape(traj.shape[0], traj.shape[1], 1)


# ---------------------------------------------------------------------------
# CuPy back‑end --------------------------------------------------------------
# ---------------------------------------------------------------------------

def _splitop_cupy(
    H0: np.ndarray,
    mu_x: np.ndarray,
    mu_y: np.ndarray,
    pol: np.ndarray,
    Efield: np.ndarray,
    psi: np.ndarray,
    dt: float,
    steps: int,
    sample_stride: int,
    hbar: float,
):
    """GPU implementation (requires CuPy)."""

    assert cp is not None, "CuPy backend requested but CuPy is not installed."

    # Convert to CuPy arrays once
    H0_cp = cp.asarray(np.diag(H0) if H0.ndim == 2 else H0, dtype=cp.float64)
    mu_x_cp = cp.asarray(mu_x)
    mu_y_cp = cp.asarray(mu_y)
    pol_cp = cp.asarray(pol)
    E_cp = cp.asarray(Efield)
    psi_cp = cp.asarray(psi)

    exp_half = cp.exp(-1j * H0_cp * dt / (2.0 * hbar))

    M_raw = pol_cp[0] * mu_x_cp + pol_cp[1] * mu_y_cp
    A_cp = 0.5 * (M_raw + M_raw.conj().T)

    eigvals, U = cp.linalg.eigh(A_cp)
    U_H = U.conj().T

    # midpoint field samples on GPU
    E_mid = E_cp[1 : 2 * steps + 1 : 2]

    n_samples = steps // sample_stride + 1
    traj = cp.empty((n_samples, psi_cp.size), dtype=cp.complex128)
    traj[0] = psi_cp

    phase_coeff = -1j * 2.0 * dt / hbar

    s_idx = 1
    for k in range(steps):
        psi_cp *= exp_half
        phase = cp.exp(phase_coeff * E_mid[k] * eigvals)
        psi_cp = U @ (phase * (U_H @ psi_cp))
        psi_cp *= exp_half
        if (k + 1) % sample_stride == 0:
            traj[s_idx] = psi_cp
            s_idx += 1

    traj_np = cp.asnumpy(traj)
    return traj_np.reshape(traj_np.shape[0], traj_np.shape[1], 1)
