# _rk4_schrodinger.py  ----------------------------------------------
"""
4-th order Runge–Kutta propagator
=================================
* backend="numpy"  →  CPU  (NumPy / Numba)
* backend="cupy"   →  GPU  (CuPy RawKernel)

電場配列は 1 propagation step あたり左端・中点・右端を持つ奇数長。
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import scipy.sparse as sp
from numba import njit

from ....core.validation import validate_wavefunction_problem
from .schrodinger_cupy import rk4_schrodinger_cupy
from .sparse import apply_hamiltonian_csr, prepare_csr_arrays

# ================================================================== #
# 1.  CPU (NumPy / Numba)                                            #
# ================================================================== #


@njit(cache=True, fastmath=True, inline="always")
def _apply_hamiltonian_dense(
    H0: Any,
    mu_x: Any,
    mu_y: Any,
    field_x: float,
    field_y: float,
    state: Any,
    output: Any,
) -> None:
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
def _rk4_cpu_numba(
    H0: Any,
    mu_x: Any,
    mu_y: Any,
    Ex: Any,
    Ey: Any,
    psi0: Any,
    dt: float,
    return_traj: bool,
    stride: int,
    renorm: bool,
) -> Any:
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
    h0_data: Any,
    h0_indices: Any,
    h0_indptr: Any,
    mu_x_data: Any,
    mu_x_indices: Any,
    mu_x_indptr: Any,
    mu_y_data: Any,
    mu_y_indices: Any,
    mu_y_indptr: Any,
    Ex: Any,
    Ey: Any,
    psi0: Any,
    dt: float,
    return_traj: bool,
    stride: int,
    renorm: bool,
) -> Any:
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
) -> Any:
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
    if backend == "cupy":
        return rk4_schrodinger_cupy(
            H0,
            mux,
            muy,
            Ex,
            Ey,
            psi0,
            float(dt),
            return_traj=return_traj,
            stride=stride,
            renorm=renorm,
        )

    psi0 = np.asarray(psi0, np.complex128).ravel()
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
