"""Physical and input-contract regression tests for wavefunction solvers."""

import numpy as np
import pytest

import rovibrational_excitation.dipole.base as dipole_base
from rovibrational_excitation.core.execution import (
    ArrayBackend,
    ExecutionPolicy,
    MatrixStorage,
)
from rovibrational_excitation.dynamics import (
    LiouvillePropagator,
    PropagatorFactory,
)
from rovibrational_excitation.dynamics.algorithms.rk4 import (
    schrodinger as rk4_module,
)
from rovibrational_excitation.dynamics.algorithms.rk4.schrodinger import (
    rk4_schrodinger,
)
from rovibrational_excitation.dynamics.algorithms.split_operator.schrodinger import (
    splitop_schrodinger,
)
from rovibrational_excitation.dynamics.capabilities import (
    PropagationAlgorithm,
    StatePath,
)
from rovibrational_excitation.dynamics.options import (
    PropagationOptions,
    RenormalizationPolicy,
    ScalingMode,
)


def _factory_options(
    *,
    algorithm: PropagationAlgorithm = PropagationAlgorithm.RK4,
    storage: MatrixStorage = MatrixStorage.DENSE,
    renormalization: RenormalizationPolicy = RenormalizationPolicy.DISABLED,
):
    return PropagationOptions(
        algorithm=algorithm,
        execution=ExecutionPolicy(backend=ArrayBackend.NUMPY, storage=storage),
        return_trajectory=True,
        sample_stride=1,
        scaling=ScalingMode.DIMENSIONAL,
        renormalization=renormalization,
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


def test_factory_returns_explicitly_configured_split_operator():
    solver = PropagatorFactory.create_propagator(
        state_path=StatePath.PURE,
        options=_factory_options(
            algorithm=PropagationAlgorithm.SPLIT_OPERATOR,
            storage=MatrixStorage.CSR,
        ),
    )

    assert solver.algorithm == "split_operator"
    assert solver.sparse is True
    assert solver.get_algorithm_name() == "Schrödinger-split_operator"


def test_factory_requires_typed_explicit_choices():
    with pytest.raises(TypeError):
        PropagatorFactory.create_propagator()

    with pytest.raises(TypeError, match="StatePath"):
        PropagatorFactory.create_propagator(
            state_path="pure",
            options=_factory_options(),
        )

    with pytest.raises(TypeError, match="PropagationOptions"):
        PropagatorFactory.create_propagator(
            state_path=StatePath.PURE,
            options="rk4-numpy-dense",
        )


def test_factory_rejects_removed_automatic_selection_inputs():
    with pytest.raises(TypeError, match="const_polarization"):
        PropagatorFactory.create_propagator(
            state_path=StatePath.PURE,
            options=_factory_options(),
            const_polarization=True,
        )


def test_factory_dispatches_explicit_density_path():
    solver = PropagatorFactory.create_propagator(
        state_path=StatePath.DENSITY,
        options=_factory_options(),
    )

    assert isinstance(solver, LiouvillePropagator)


def test_factory_rejects_renormalization_for_density_path():
    with pytest.raises(ValueError, match="renorm is not applicable"):
        PropagatorFactory.create_propagator(
            state_path=StatePath.DENSITY,
            options=_factory_options(renormalization=RenormalizationPolicy.PER_STEP),
        )


def test_dipole_backend_does_not_fall_back_from_cupy(monkeypatch):
    monkeypatch.setattr(dipole_base, "cp", None)

    with pytest.raises(RuntimeError, match="CuPy backend requested"):
        dipole_base._xp("cupy")


def test_dipole_backend_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown backend"):
        dipole_base._xp("cuda")
