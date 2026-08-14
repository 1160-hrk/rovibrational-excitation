"""Physical endpoint references for the typed propagation result boundary."""

import numpy as np

from rovibrational_excitation.core.electric_field import ElectricField
from rovibrational_excitation.core.propagation import (
    LiouvillePropagator,
    SchrodingerPropagator,
)
from rovibrational_excitation.core.states import DensityState, PureState
from rovibrational_excitation.core.time import TimeGrid
from tests.propagation_options import propagation_options
from tests.propagation_problem import propagation_problem


def _field() -> ElectricField:
    grid = TimeGrid.from_bounds(1.0, 1.5, 0.05)
    return ElectricField.from_time_grid(grid)


def test_wavefunction_result_appends_exact_endpoint_without_changing_states():
    field = _field()
    initial = PureState(np.array([np.sqrt(0.3), np.sqrt(0.7)], dtype=np.complex128))
    problem = propagation_problem(initial, field=field)
    solver = SchrodingerPropagator(validate_units=False)
    full_time, full_state = solver._propagate_array(
        problem.model.hamiltonian,
        problem.field,
        problem.model.dipole,
        initial.amplitudes,
        coupling_mode="cartesian",
        axes="xy",
        return_time_psi=True,
    )

    result = solver.propagate(
        problem,
        options=propagation_options(return_trajectory=True, sample_stride=2),
    )

    np.testing.assert_allclose(
        result.times_fs, [1.0, 1.2, 1.4, 1.5], rtol=0.0, atol=5.0e-16
    )
    assert result.times_fs[0] == 1.0
    assert result.times_fs[-1] == 1.5
    np.testing.assert_array_equal(result.state[:-1], full_state[::2])
    np.testing.assert_array_equal(result.state[-1], full_state[-1])
    np.testing.assert_allclose(
        full_time, np.linspace(1.0, 1.5, 6), rtol=0.0, atol=5.0e-16
    )


def test_density_result_appends_exact_endpoint_without_changing_states():
    field = _field()
    initial_vector = np.array(
        [np.sqrt(0.4), np.sqrt(0.6) * np.exp(0.2j)], dtype=np.complex128
    )
    initial = DensityState(np.outer(initial_vector, initial_vector.conj()))
    problem = propagation_problem(initial, field=field)
    solver = LiouvillePropagator(validate_units=False)
    full_time, full_state = solver._propagate_array(
        problem.model.hamiltonian,
        problem.field,
        problem.model.dipole,
        initial.matrix,
        coupling_mode="cartesian",
        axes="xy",
        return_time_rho=True,
    )

    result = solver.propagate(
        problem,
        options=propagation_options(return_trajectory=True, sample_stride=2),
    )

    np.testing.assert_allclose(
        result.times_fs, [1.0, 1.2, 1.4, 1.5], rtol=0.0, atol=5.0e-16
    )
    assert result.times_fs[0] == 1.0
    assert result.times_fs[-1] == 1.5
    np.testing.assert_array_equal(result.state[:-1], full_state[::2])
    np.testing.assert_array_equal(result.state[-1], full_state[-1])
    np.testing.assert_allclose(
        full_time, np.linspace(1.0, 1.5, 6), rtol=0.0, atol=5.0e-16
    )
