# _rk4_schrodinger.py  ----------------------------------------------
"""
4-th order Runge–Kutta propagator
=================================
* backend="numpy"  →  CPU  (NumPy / Numba)
* backend="cupy"   →  GPU  (CuPy RawKernel)

電場配列は奇数・偶数どちらの長さでも OK。
"""

from __future__ import annotations

from typing import Literal, Union

import numpy as np
from scipy.sparse import csr_matrix



# ------------------------------------------------------------------ #
# 0.  電場ヘルパ：3-tuple 配列 & step 数を返す                       #
# ------------------------------------------------------------------ #
def _field_to_triplets(field: np.ndarray) -> np.ndarray:
    """
    奇数長 → そのまま
    偶数長 → 末尾 1 点をバッサリ捨てる
    """
    ex1 = field[0:-2:2]
    ex2 = field[1:-1:2]
    ex4 = field[2::2]
    return np.column_stack((ex1, ex2, ex4)).astype(np.float64, copy=False)


# ================================================================== #
# 1.  CPU (NumPy / Numba)                                            #
# ================================================================== #
try:
    from numba import njit
except ImportError:  # numba 不在でも動くダミー

    def njit(*args, **kwargs):  # type: ignore
        def deco(f):
            return f

        return deco


@njit(
    "c16[:, :](c16[:, :], c16[:, :], c16[:, :],"
    "f8[:], f8[:],"
    "c16[:], f8, b1, i8, b1)",
    fastmath=True,
    cache=True,
)
def _rk4_cpu(H0, mux, muy, Ex, Ey, psi0, dt, return_traj, stride, renorm):

    steps = (Ex.size - 1) // 2  # 必ず整数
    Ex3, Ey3 = np.zeros((steps, 3), dtype=np.float64), np.zeros((steps, 3), dtype=np.float64)
    Ex3[:, 0], Ey3[:, 0] = Ex[0:-2:2], Ey[0:-2:2]
    Ex3[:, 1], Ey3[:, 1] = Ex[1:-1:2], Ey[1:-1:2]
    Ex3[:, 2], Ey3[:, 2] = Ex[2::2], Ey[2::2]
    psi = psi0.copy()
    dim = psi.size
    n_out = steps // stride + 1
    out = np.empty((n_out, dim), np.complex128)
    out[0] = psi
    idx = 1
    buf = np.empty_like(psi)
    k1 = np.empty_like(psi)
    k2 = np.empty_like(psi)
    k3 = np.empty_like(psi)
    k4 = np.empty_like(psi)
    for s in range(steps):
        ex1, ex2, ex4 = Ex3[s]
        ey1, ey2, ey4 = Ey3[s]
        H1 = H0 - mux * ex1 - muy * ey1
        H2 = H0 - mux * ex2 - muy * ey2  # =H3
        H4 = H0 - mux * ex4 - muy * ey4
        k1[:] = -1j * (H1 @ psi)
        buf[:] = psi + 0.5 * dt * k1
        k2[:] = -1j * (H2 @ buf)
        buf[:] = psi + 0.5 * dt * k2
        k3[:] = -1j * (H2 @ buf)
        buf[:] = psi + dt * k3
        k4[:] = -1j * (H4 @ buf)
        psi += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if renorm:
            # より高精度な正規化
            norm = np.sqrt((psi.conj() @ psi).real)
            if norm > 1e-12:  # 数値的にゼロでない場合のみ正規化
                psi *= 1.0 / norm
            else:
                continue
        if return_traj and (s + 1) % stride == 0:
            out[idx] = psi
            idx += 1
    if return_traj:
        return out
    else:
        return psi.reshape((1, dim))


def _rk4_cpu_sparse(
    H0: Union[csr_matrix, np.ndarray],
    mux: Union[csr_matrix, np.ndarray],
    muy: Union[csr_matrix, np.ndarray],
    Ex: np.ndarray,
    Ey: np.ndarray,
    psi0: np.ndarray,
    dt: float,
    return_traj: bool,
    stride: int,
    renorm: bool = False,
) -> np.ndarray:
    """
    4th-order Runge-Kutta propagator with sparse matrices using precomputed pattern.

    Parameters
    ----------
    H0, mux, muy : csr_matrix
        Hamiltonian and dipole operators (sparse)
    Ex, Ey : (2*steps+1) ndarray
        Electric field triplets
    psi0 : (dim,) ndarray
        Initial wavefunction
    dt : float
        Time step
    steps : int
        Number of time steps
    stride : int
        Output stride
    renorm : bool
        Normalize wavefunction at each step

    Returns
    -------
    out : (n_out, dim) ndarray
        Time evolution
    """
    steps = (Ex.size - 1) // 2  # 必ず整数
    Ex3 = _field_to_triplets(Ex)
    Ey3 = _field_to_triplets(Ey)
    
    if not isinstance(H0, csr_matrix):
        H0 = csr_matrix(H0)
    if not isinstance(mux, csr_matrix):
        mux = csr_matrix(mux)
    if not isinstance(muy, csr_matrix):
        muy = csr_matrix(muy)

    psi = psi0.copy()
    dim = psi.size
    n_out = steps // stride + 1
    out = np.empty((n_out, dim), dtype=np.complex128)
    out[0] = psi
    idx = 1

    # 1️⃣ 共通パターン（構造のみ）を作成
    pattern = ((H0 != 0) + (mux != 0) + (muy != 0))
    pattern = pattern.astype(np.complex128)  # 確実に複素数
    pattern.data[:] = 1.0 + 0j
    pattern = pattern.tocsr()

    # 2️⃣ パターンに合わせてデータを展開
    def expand_to_pattern(matrix: csr_matrix, pattern: csr_matrix) -> np.ndarray:
        result_data = np.zeros_like(pattern.data, dtype=np.complex128)
        m_csr = matrix.tocsr()
        pi, pj = pattern.nonzero()
        m_dict = {(i, j): v for i, j, v in zip(*m_csr.nonzero(), m_csr.data)}
        for idx_, (i, j) in enumerate(zip(pi, pj)):
            result_data[idx_] = m_dict.get((i, j), 0.0 + 0j)
        return result_data

    H0_data = expand_to_pattern(H0, pattern)
    mux_data = expand_to_pattern(mux, pattern)
    muy_data = expand_to_pattern(muy, pattern)

    # 3️⃣ 計算用行列
    H = pattern.copy()

    # 4️⃣ 作業バッファ
    buf = np.empty_like(psi)
    k1 = np.empty_like(psi)
    k2 = np.empty_like(psi)
    k3 = np.empty_like(psi)
    k4 = np.empty_like(psi)

    for s in range(steps):
        ex1, ex2, ex4 = Ex3[s]
        ey1, ey2, ey4 = Ey3[s]

        # H1
        H.data[:] = H0_data -mux_data * ex1 -muy_data * ey1
        k1[:] = -1j * H.dot(psi)
        buf[:] = psi + 0.5 * dt * k1

        # H2
        H.data[:] = H0_data -mux_data * ex2 -muy_data * ey2
        k2[:] = -1j * H.dot(buf)
        buf[:] = psi + 0.5 * dt * k2

        # H3
        H.data[:] = H0_data -mux_data * ex2 -muy_data * ey2
        k3[:] = -1j * H.dot(buf)
        buf[:] = psi + dt * k3

        # H4
        H.data[:] = H0_data -mux_data * ex4 -muy_data * ey4
        k4[:] = -1j * H.dot(buf)
        psi += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        
        if renorm:
            # より高精度な正規化
            norm = np.sqrt((psi.conj() @ psi).real)
            if norm > 1e-12:  # 数値的にゼロでない場合のみ正規化
                psi *= 1.0 / norm
            else:
                continue
        
        if return_traj and (s + 1) % stride == 0:
            out[idx] = psi
            idx += 1

    if return_traj:
        return out
    else:
        return psi



# ================================================================== #
# 2.  GPU (CuPy RawKernel)                                           #
# ================================================================== #
try:
    import cupy as cp
except ImportError:
    cp = None

_KERNEL_SRC_TEMPLATE = r"""
extern "C" __global__
void rk4_loop(const cuDoubleComplex* __restrict__ H0,
              const cuDoubleComplex* __restrict__ mux,
              const cuDoubleComplex* __restrict__ muy,
              const double*  __restrict__ Ex3,
              const double*  __restrict__ Ey3,
              cuDoubleComplex* __restrict__ psi)
{{
    const int DIM   = {dim};
    const int STEPS = {steps};
    const double dt = {dt};

    extern __shared__ cuDoubleComplex sh[];
    cuDoubleComplex* k1  = sh;
    cuDoubleComplex* k2  = k1  + DIM;
    cuDoubleComplex* k3  = k2  + DIM;
    cuDoubleComplex* k4  = k3  + DIM;
    cuDoubleComplex* buf = k4  + DIM;

    const int row = threadIdx.x;
    if (row < DIM) buf[row] = psi[row];
    __syncthreads();

#define MATVEC(Hmat, ex, ey, dst)                                   \
    if (row < DIM) {{                                               \
        cuDoubleComplex acc = make_cuDoubleComplex(0.0, 0.0);       \
        for (int col = 0; col < DIM; ++col) {{                      \
            cuDoubleComplex hij = Hmat[row*DIM+col];                \
            cuDoubleComplex mx  = mux[row*DIM+col];                 \
            cuDoubleComplex my  = muy[row*DIM+col];                 \
            hij = cuCadd(hij,                                       \
                  cuCadd(make_cuDoubleComplex(mx.x*ex, mx.y*ex),    \
                        make_cuDoubleComplex(my.x*ey, my.y*ey)));    \
            acc = cuCadd(acc, cuCmul(hij, buf[col]));               \
        }}                                                          \
        dst[row] = cuCmul(make_cuDoubleComplex(0.0,-1.0), acc);     \
    }}                                                              \
    __syncthreads();

    for (int s = 0; s < STEPS; ++s) {{
        const double ex1 = Ex3[3*s],   ex2 = Ex3[3*s+1], ex4 = Ex3[3*s+2];
        const double ey1 = Ey3[3*s],   ey2 = Ey3[3*s+1], ey4 = Ey3[3*s+2];

        MATVEC(H0, ex1, ey1, k1)

        if (row < DIM) buf[row] = cuCadd(buf[row],
                 make_cuDoubleComplex(0.5*dt*k1[row].x, 0.5*dt*k1[row].y));
        __syncthreads();

        MATVEC(H0, ex2, ey2, k2)

        if (row < DIM) buf[row] = cuCadd(cuCsub(buf[row],
                 make_cuDoubleComplex(0.5*dt*k1[row].x, 0.5*dt*k1[row].y)),
                 make_cuDoubleComplex(0.5*dt*k2[row].x, 0.5*dt*k2[row].y));
        __syncthreads();

        MATVEC(H0, ex2, ey2, k3)

        if (row < DIM) buf[row] = cuCadd(cuCsub(buf[row],
                 make_cuDoubleComplex(0.5*dt*k2[row].x, 0.5*dt*k2[row].y)),
                 make_cuDoubleComplex(dt*k3[row].x, dt*k3[row].y));
        __syncthreads();

        MATVEC(H0, ex4, ey4, k4)

        if (row < DIM) {{
            cuDoubleComplex inc = cuCadd(k1[row],
                 cuCadd(k4[row], cuCadd(k2[row], k2[row])));
            inc = cuCadd(inc, cuCadd(k3[row], k3[row])); // +2k3
            inc = make_cuDoubleComplex((dt/6.0)*inc.x, (dt/6.0)*inc.y);
            buf[row] = cuCadd(buf[row], inc);
        }}
        __syncthreads();
    }}

    if (row < DIM) psi[row] = buf[row];
}}
"""  # noqa: E501 (long CUDA string)


def _rk4_gpu(H0, mux, muy, Ex, Ey, psi0, dt: float):
    if cp is None:
        raise RuntimeError("backend='cupy' but CuPy is not installed")
    dim = H0.shape[0]
    steps = (Ex.size - 1) // 2  # 必ず整数
    Ex3 = _field_to_triplets(Ex)
    Ey3 = _field_to_triplets(Ey)
    src = _KERNEL_SRC_TEMPLATE.format(dim=dim, steps=steps, dt=dt)
    mod = cp.RawModule(
        code=src, options=("-std=c++17",), name_expressions=("rk4_loop",)
    )
    kern = mod.get_function("rk4_loop")

    H0_d = cp.asarray(H0)
    mux_d = cp.asarray(mux)
    muy_d = cp.asarray(muy)
    Ex3_d = cp.asarray(Ex3)
    Ey3_d = cp.asarray(Ey3)
    psi_d = cp.asarray(psi0)

    shm = dim * 5 * 16  # k1..k4+buf  (complex128=16B)
    kern((1,), (dim,), (H0_d, mux_d, muy_d, Ex3_d, Ey3_d, psi_d), shared_mem=shm)
    return psi_d.get()[None, :]


# ------------------------------------------------------------------ #
# 3.  公開 API                                                       #
# ------------------------------------------------------------------ #
def rk4_schrodinger(
    H0: np.ndarray,
    mux: np.ndarray,
    muy: np.ndarray,
    Ex: np.ndarray,
    Ey: np.ndarray,
    psi0: np.ndarray,
    dt: float,
    return_traj: bool = True,
    stride: int = 1,
    renorm: bool = False,
    sparse: bool = False,
    *,
    backend: Literal["numpy", "cupy"] = "numpy",
) -> np.ndarray:
    """
    TDSE propagator (4th-order RK).

    Returns
    -------
    psi_traj : (n_sample, dim) complex128
        return_traj=False → shape (1, dim)
    """
    psi0 = np.asarray(psi0, np.complex128).ravel()

    if backend == "cupy":
        if return_traj:
            return _rk4_gpu(H0, mux, muy, Ex, Ey, psi0, float(dt))
        else:
            return _rk4_gpu(H0, mux, muy, Ex, Ey, psi0, float(dt))[-1]

    if sparse or isinstance(mux, csr_matrix) or isinstance(muy, csr_matrix):
        return _rk4_cpu_sparse(
            H0, 
            mux,
            muy,
            Ex,
            Ey,
            psi0,
            float(dt),
            return_traj,
            stride,
            renorm,
        )
    else:
        return _rk4_cpu(
        np.ascontiguousarray(H0, np.complex128),
        np.ascontiguousarray(mux, np.complex128),
        np.ascontiguousarray(muy, np.complex128),
        Ex,
        Ey,
        psi0,
        float(dt),
        return_traj,
        stride,
        renorm
    )