"""Accuracy and dispatch contracts for the NumPy/Numba CSR RK4 path."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from rovibrational_excitation.core.propagation.algorithms.rk4.schrodinger import (
    rk4_schrodinger,
)
from rovibrational_excitation.core.propagation.algorithms.rk4.sparse import (
    apply_hamiltonian_csr,
    prepare_csr_arrays,
)

pytestmark = pytest.mark.physics


def _sparse_problem(dimension: int = 64):
    diagonal = np.linspace(0.05, 1.25, dimension)
    h0 = sp.diags(
        (-0.02 * np.ones(dimension - 1), diagonal, -0.02 * np.ones(dimension - 1)),
        offsets=(-1, 0, 1),
        dtype=np.complex128,
        format="csr",
    )
    coupling = np.linspace(0.1, 0.3, dimension - 1)
    mu_x = sp.diags(
        (coupling, coupling),
        offsets=(-1, 1),
        dtype=np.complex128,
        format="csr",
    )
    mu_y = sp.diags(
        (-1j * coupling, 1j * coupling),
        offsets=(-1, 1),
        dtype=np.complex128,
        format="csr",
    )
    field_phase = np.linspace(0.0, 3.0 * np.pi, 401)
    field_x = 0.4 * np.sin(field_phase)
    field_y = 0.2 * np.cos(field_phase)
    initial = np.zeros(dimension, dtype=np.complex128)
    initial[0] = np.sqrt(0.6)
    initial[1] = np.sqrt(0.3) * np.exp(0.2j)
    initial[dimension // 2] = np.sqrt(0.1) * np.exp(-0.4j)
    return h0, mu_x, mu_y, field_x, field_y, initial


def test_fused_csr_hamiltonian_application_matches_dense_reference():
    h0, mu_x, mu_y, field_x, field_y, initial = _sparse_problem(dimension=8)
    h0_arrays = prepare_csr_arrays(h0)
    mu_x_arrays = prepare_csr_arrays(mu_x)
    mu_y_arrays = prepare_csr_arrays(mu_y)
    actual = np.empty_like(initial)

    apply_hamiltonian_csr(
        *h0_arrays,
        *mu_x_arrays,
        *mu_y_arrays,
        field_x[3],
        field_y[3],
        initial,
        actual,
    )
    expected = -1j * (
        (h0.toarray() - field_x[3] * mu_x.toarray() - field_y[3] * mu_y.toarray())
        @ initial
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-15)


def test_numba_csr_rk4_matches_dense_trajectory_and_final_only():
    h0, mu_x, mu_y, field_x, field_y, initial = _sparse_problem()
    dense = rk4_schrodinger(
        h0.toarray(),
        mu_x.toarray(),
        mu_y.toarray(),
        field_x,
        field_y,
        initial,
        0.002,
        return_traj=True,
        sparse=False,
    )
    sparse = rk4_schrodinger(
        h0,
        mu_x,
        mu_y,
        field_x,
        field_y,
        initial,
        0.002,
        return_traj=True,
        sparse=True,
    )
    sparse_final = rk4_schrodinger(
        h0,
        mu_x,
        mu_y,
        field_x,
        field_y,
        initial,
        0.002,
        return_traj=False,
        sparse=True,
    )

    np.testing.assert_allclose(sparse, dense, rtol=0.0, atol=3.0e-13)
    assert sparse_final.shape == (1, initial.size)
    np.testing.assert_allclose(sparse_final[0], sparse[-1], rtol=0.0, atol=0.0)


def test_csr_input_requires_explicit_sparse_selection():
    h0, mu_x, mu_y, field_x, field_y, initial = _sparse_problem(dimension=8)

    with pytest.raises(ValueError, match="sparse=True"):
        rk4_schrodinger(
            h0,
            mu_x,
            mu_y,
            field_x[:3],
            field_y[:3],
            initial,
            0.002,
            sparse=False,
        )


def test_csr_preparation_is_canonical_without_mutating_input():
    source = sp.csr_matrix(
        (
            np.array([2.0, 0.0, 1.0, -1.0], dtype=np.complex128),
            np.array([1, 0, 1, 1]),
            np.array([0, 2, 4]),
        ),
        shape=(2, 2),
    )
    original_data = source.data.copy()
    original_indices = source.indices.copy()
    original_indptr = source.indptr.copy()

    data, indices, indptr = prepare_csr_arrays(source)

    np.testing.assert_array_equal(source.data, original_data)
    np.testing.assert_array_equal(source.indices, original_indices)
    np.testing.assert_array_equal(source.indptr, original_indptr)
    assert np.all(data != 0.0)
    assert data.flags.c_contiguous
    assert indices.flags.c_contiguous
    assert indptr.flags.c_contiguous


def test_sparse_renormalization_rejects_zero_norm_state():
    h0, mu_x, mu_y, field_x, field_y, initial = _sparse_problem(dimension=8)

    with pytest.raises(ValueError, match="zero or non-finite"):
        rk4_schrodinger(
            h0,
            mu_x,
            mu_y,
            field_x[:3],
            field_y[:3],
            np.zeros_like(initial),
            0.002,
            sparse=True,
            renorm=True,
        )
