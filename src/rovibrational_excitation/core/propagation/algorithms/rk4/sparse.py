"""CSR preparation and Numba kernels for RK4 wavefunction propagation."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numba import njit

CSRArrays = tuple[np.ndarray, np.ndarray, np.ndarray]


def prepare_csr_arrays(matrix) -> CSRArrays:
    """Return canonical contiguous CSR arrays without mutating the input.

    Numerical sparsification is intentionally absent: only stored exact zeros
    are removed. Any approximate truncation must be an explicit preprocessing
    policy outside the propagation kernel.
    """
    csr = sp.csr_matrix(matrix, dtype=np.complex128, copy=True)
    csr.sum_duplicates()
    csr.eliminate_zeros()
    csr.sort_indices()
    return (
        np.ascontiguousarray(csr.data, dtype=np.complex128),
        np.ascontiguousarray(csr.indices),
        np.ascontiguousarray(csr.indptr),
    )


@njit(cache=True)
def apply_hamiltonian_csr(
    h0_data,
    h0_indices,
    h0_indptr,
    mu_x_data,
    mu_x_indices,
    mu_x_indptr,
    mu_y_data,
    mu_y_indices,
    mu_y_indptr,
    field_x,
    field_y,
    state,
    output,
):
    """Set output to -1j * (H0 - mu_x*Ex - mu_y*Ey) @ state."""
    dimension = h0_indptr.size - 1
    for row in range(dimension):
        value = 0.0 + 0.0j

        for index in range(h0_indptr[row], h0_indptr[row + 1]):
            value += h0_data[index] * state[h0_indices[index]]

        for index in range(mu_x_indptr[row], mu_x_indptr[row + 1]):
            value -= field_x * mu_x_data[index] * state[mu_x_indices[index]]

        for index in range(mu_y_indptr[row], mu_y_indptr[row + 1]):
            value -= field_y * mu_y_data[index] * state[mu_y_indices[index]]

        output[row] = -1j * value
