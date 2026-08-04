"""Physical and input-contract regression tests for wavefunction solvers."""

import numpy as np
import pytest

from rovibrational_excitation.core.propagation import PropagatorFactory
from rovibrational_excitation.core.propagation.algorithms.rk4 import (
    schrodinger as rk4_module,
)
from rovibrational_excitation.core.propagation.algorithms.rk4.schrodinger import (
    rk4_schrodinger,
)
from rovibrational_excitation.core.propagation.algorithms.split_operator.schrodinger import (
    splitop_schrodinger,
)


def test_split_operator_preserves_permanent_dipole_contribution():
    """A diagonal dipole must contribute a relative phase, not be discarded."""
    h0 = np.zeros((2, 2))
    mu_x = np.diag([1.0, 2.0]).astype(np.complex128)
    mu_y = np.zeros((2, 2), dtype=np.complex128)
    field = np.ones(5)
    psi0 = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    dt = 0.1

    final = splitop_schrodinger(
        h0,
        mu_x,
        mu_y,
        field,
        np.zeros_like(field),
        psi0,
        dt,
        return_traj=False,
    )[0]

    steps = (field.size - 1) // 2
    expected = psi0 * np.exp(1j * np.diag(mu_x) * dt * steps)
    np.testing.assert_allclose(final, expected, atol=1e-12)


@pytest.mark.parametrize(
    ("field_x", "field_y", "message"),
    [
        (np.ones(5), np.ones(3), r"field\[1\]"),
        (np.array([]), np.array([]), "at least 3 points"),
        (np.array([0.0, np.nan, 0.0]), np.zeros(3), "finite"),
    ],
)
def test_rk4_rejects_invalid_fields(field_x, field_y, message):
    h0 = np.diag([0.0, 1.0])
    mu = np.zeros((2, 2), dtype=np.complex128)
    psi0 = np.array([1.0, 0.0], dtype=np.complex128)

    with pytest.raises(ValueError, match=message):
        rk4_schrodinger(h0, mu, mu, field_x, field_y, psi0, dt=0.1)


def test_rk4_rejects_even_field_instead_of_silently_dropping_endpoint():
    h0 = np.diag([0.0, 1.0])
    mu = np.zeros((2, 2), dtype=np.complex128)
    psi0 = np.array([1.0, 0.0], dtype=np.complex128)

    with pytest.raises(ValueError, match=r"2\*n_steps \+ 1"):
        rk4_schrodinger(h0, mu, mu, np.zeros(4), np.zeros(4), psi0, dt=0.1)


def test_split_operator_rejects_nondiagonal_free_hamiltonian():
    h0 = np.array([[0.0, 0.1], [0.1, 1.0]])
    mu = np.zeros((2, 2), dtype=np.complex128)
    psi0 = np.array([1.0, 0.0], dtype=np.complex128)

    with pytest.raises(ValueError, match="diagonal H0"):
        splitop_schrodinger(
            h0,
            mu,
            mu,
            np.zeros(3),
            np.zeros(3),
            psi0,
            dt=0.1,
        )


def test_solver_rejects_zero_timestep():
    h0 = np.diag([0.0, 1.0])
    mu = np.zeros((2, 2), dtype=np.complex128)
    psi0 = np.array([1.0, 0.0], dtype=np.complex128)

    with pytest.raises(ValueError, match="non-zero"):
        rk4_schrodinger(h0, mu, mu, np.zeros(3), np.zeros(3), psi0, dt=0.0)


def test_cupy_final_only_keeps_low_level_row_shape(monkeypatch):
    """CPU and GPU low-level final-only results both have shape (1, dim)."""
    expected = np.array([[0.25 + 0.1j, 0.75 - 0.2j]])

    def fake_gpu(*args):
        del args
        return expected.copy()

    monkeypatch.setattr(rk4_module, "_rk4_gpu", fake_gpu)
    result = rk4_module.rk4_schrodinger(
        np.diag([0.0, 1.0]),
        np.zeros((2, 2)),
        np.zeros((2, 2)),
        np.zeros(3),
        np.zeros(3),
        np.array([1.0, 0.0]),
        dt=0.1,
        return_traj=False,
        backend="cupy",
    )

    assert result.shape == (1, 2)
    np.testing.assert_array_equal(result, expected)


def test_factory_returns_configured_split_operator():
    solver = PropagatorFactory.create_propagator(
        state_type="pure",
        algorithm="split_operator",
    )

    assert solver.algorithm == "split_operator"
    assert solver.get_algorithm_name() == "Schrödinger-split_operator"


def test_factory_automatic_selection_is_observable():
    split_solver = PropagatorFactory.create_propagator(
        state_type="pure", const_polarization=True
    )
    rk4_solver = PropagatorFactory.create_propagator(state_type="pure")

    assert split_solver.algorithm == "split_operator"
    assert rk4_solver.algorithm == "rk4"


def test_factory_rejects_unknown_algorithm():
    with pytest.raises(ValueError, match="algorithm"):
        PropagatorFactory.create_propagator(algorithm="split-operator")
