# _rk4_schrodinger.py  ----------------------------------------------
"""
4-th order Runge–Kutta propagator
=================================
* backend="numpy"  →  CPU  (NumPy / Numba)
* backend="cupy"   →  GPU  (CuPy RawKernel)

電場配列は 1 propagation step あたり左端・中点・右端を持つ奇数長。
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.sparse as sp
from numba import njit

from ..validation import validate_wavefunction_problem
from .sparse import apply_hamiltonian_csr, prepare_csr_arrays


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


@njit(cache=True, fastmath=True, inline="always")
def _apply_hamiltonian_dense(H0, mu_x, mu_y, field_x, field_y, state, output):
    """Set output to -1j * (H0 - mu_x*Ex - mu_y*Ey) @ state."""
    dimension = state.size
    for row in range(dimension):
        value = 0.0 + 0.0j
        for column in range(dimension):
            value += (
                H0[row, column]
                - field_x * mu_x[row, column]
                - field_y * mu_y[row, column]
            ) * state[column]
        output[row] = -1j * value


@njit(cache=True, fastmath=True)
def _rk4_cpu_numba(H0, mu_x, mu_y, Ex, Ey, psi0, dt, return_traj, stride, renorm):
    """Run allocation-stable dense RK4 entirely inside Numba."""
    steps = (Ex.size - 1) // 2
    psi = psi0.copy()
    dimension = psi.size
    output_rows = steps // stride + 1 if return_traj else 1
    output = np.empty((output_rows, dimension), dtype=np.complex128)
    output_index = 0
    if return_traj:
        output[0] = psi
        output_index = 1

    buffer = np.empty_like(psi)
    k1 = np.empty_like(psi)
    k2 = np.empty_like(psi)
    k3 = np.empty_like(psi)
    k4 = np.empty_like(psi)

    for step_index in range(steps):
        field_index = 2 * step_index

        _apply_hamiltonian_dense(
            H0,
            mu_x,
            mu_y,
            Ex[field_index],
            Ey[field_index],
            psi,
            k1,
        )
        for index in range(dimension):
            buffer[index] = psi[index] + 0.5 * dt * k1[index]

        _apply_hamiltonian_dense(
            H0,
            mu_x,
            mu_y,
            Ex[field_index + 1],
            Ey[field_index + 1],
            buffer,
            k2,
        )
        for index in range(dimension):
            buffer[index] = psi[index] + 0.5 * dt * k2[index]

        _apply_hamiltonian_dense(
            H0,
            mu_x,
            mu_y,
            Ex[field_index + 1],
            Ey[field_index + 1],
            buffer,
            k3,
        )
        for index in range(dimension):
            buffer[index] = psi[index] + dt * k3[index]

        _apply_hamiltonian_dense(
            H0,
            mu_x,
            mu_y,
            Ex[field_index + 2],
            Ey[field_index + 2],
            buffer,
            k4,
        )

        for index in range(dimension):
            psi[index] += (dt / 6.0) * (
                k1[index] + 2.0 * k2[index] + 2.0 * k3[index] + k4[index]
            )

        if renorm:
            norm_squared = 0.0
            for index in range(dimension):
                norm_squared += (
                    psi[index].real * psi[index].real
                    + psi[index].imag * psi[index].imag
                )
            if norm_squared <= 0.0 or not np.isfinite(norm_squared):
                raise ValueError("cannot renormalize a zero or non-finite wavefunction")
            inverse_norm = 1.0 / np.sqrt(norm_squared)
            for index in range(dimension):
                psi[index] *= inverse_norm

        if return_traj and (step_index + 1) % stride == 0:
            output[output_index] = psi
            output_index += 1

    if not return_traj:
        output[0] = psi
    return output


@njit(cache=True)
def _rk4_cpu_numba_csr(
    h0_data,
    h0_indices,
    h0_indptr,
    mu_x_data,
    mu_x_indices,
    mu_x_indptr,
    mu_y_data,
    mu_y_indices,
    mu_y_indptr,
    Ex,
    Ey,
    psi0,
    dt,
    return_traj,
    stride,
    renorm,
):
    """Run RK4 entirely in Numba using pre-canonicalized CSR arrays."""
    steps = (Ex.size - 1) // 2
    psi = psi0.copy()
    dimension = psi.size
    output_rows = steps // stride + 1 if return_traj else 1
    output = np.empty((output_rows, dimension), dtype=np.complex128)
    output_index = 0
    if return_traj:
        output[0] = psi
        output_index = 1

    buffer = np.empty_like(psi)
    k1 = np.empty_like(psi)
    k2 = np.empty_like(psi)
    k3 = np.empty_like(psi)
    k4 = np.empty_like(psi)

    for step_index in range(steps):
        field_index = 2 * step_index

        apply_hamiltonian_csr(
            h0_data,
            h0_indices,
            h0_indptr,
            mu_x_data,
            mu_x_indices,
            mu_x_indptr,
            mu_y_data,
            mu_y_indices,
            mu_y_indptr,
            Ex[field_index],
            Ey[field_index],
            psi,
            k1,
        )
        for index in range(dimension):
            buffer[index] = psi[index] + 0.5 * dt * k1[index]

        apply_hamiltonian_csr(
            h0_data,
            h0_indices,
            h0_indptr,
            mu_x_data,
            mu_x_indices,
            mu_x_indptr,
            mu_y_data,
            mu_y_indices,
            mu_y_indptr,
            Ex[field_index + 1],
            Ey[field_index + 1],
            buffer,
            k2,
        )
        for index in range(dimension):
            buffer[index] = psi[index] + 0.5 * dt * k2[index]

        apply_hamiltonian_csr(
            h0_data,
            h0_indices,
            h0_indptr,
            mu_x_data,
            mu_x_indices,
            mu_x_indptr,
            mu_y_data,
            mu_y_indices,
            mu_y_indptr,
            Ex[field_index + 1],
            Ey[field_index + 1],
            buffer,
            k3,
        )
        for index in range(dimension):
            buffer[index] = psi[index] + dt * k3[index]

        apply_hamiltonian_csr(
            h0_data,
            h0_indices,
            h0_indptr,
            mu_x_data,
            mu_x_indices,
            mu_x_indptr,
            mu_y_data,
            mu_y_indices,
            mu_y_indptr,
            Ex[field_index + 2],
            Ey[field_index + 2],
            buffer,
            k4,
        )

        for index in range(dimension):
            psi[index] += (dt / 6.0) * (
                k1[index] + 2.0 * k2[index] + 2.0 * k3[index] + k4[index]
            )

        if renorm:
            norm_squared = 0.0
            for index in range(dimension):
                norm_squared += (
                    psi[index].real * psi[index].real
                    + psi[index].imag * psi[index].imag
                )
            if norm_squared <= 0.0 or not np.isfinite(norm_squared):
                raise ValueError("cannot renormalize a zero or non-finite wavefunction")
            inverse_norm = 1.0 / np.sqrt(norm_squared)
            for index in range(dimension):
                psi[index] *= inverse_norm

        if return_traj and (step_index + 1) % stride == 0:
            output[output_index] = psi
            output_index += 1

    if not return_traj:
        output[0] = psi
    return output


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
    validate_wavefunction_problem(
        H0,
        (mux, muy),
        (Ex, Ey),
        psi0,
        dt=dt,
        stride=stride,
        backend=backend,
        require_odd_field=True,
    )
    psi0 = np.asarray(psi0, np.complex128).ravel()

    if backend == "cupy":
        if return_traj:
            return _rk4_gpu(H0, mux, muy, Ex, Ey, psi0, float(dt))
        else:
            return _rk4_gpu(H0, mux, muy, Ex, Ey, psi0, float(dt))

    operators = (H0, mux, muy)
    if sparse:
        h0_arrays = prepare_csr_arrays(H0)
        mu_x_arrays = prepare_csr_arrays(mux)
        mu_y_arrays = prepare_csr_arrays(muy)
        return _rk4_cpu_numba_csr(
            *h0_arrays,
            *mu_x_arrays,
            *mu_y_arrays,
            np.ascontiguousarray(Ex, dtype=np.float64),
            np.ascontiguousarray(Ey, dtype=np.float64),
            psi0,
            float(dt),
            return_traj,
            stride,
            renorm,
        )

    if any(sp.issparse(operator) for operator in operators):
        raise ValueError("CSR operator input requires sparse=True")

    return _rk4_cpu_numba(
        np.ascontiguousarray(H0, np.complex128),
        np.ascontiguousarray(mux, np.complex128),
        np.ascontiguousarray(muy, np.complex128),
        np.ascontiguousarray(Ex, dtype=np.float64),
        np.ascontiguousarray(Ey, dtype=np.float64),
        psi0,
        float(dt),
        return_traj,
        stride,
        renorm,
    )
