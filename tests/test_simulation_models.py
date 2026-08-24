"""Regression tests for model construction extracted from simulation.runner."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rovibrational_excitation.core.execution import (
    ArrayBackend,
    ExecutionPolicy,
    MatrixStorage,
)
from rovibrational_excitation.core.states import PureState
from rovibrational_excitation.core.time import TimeGrid
from rovibrational_excitation.fields import (
    ElectricField as RealElectricField,
)
from rovibrational_excitation.models import build_model
from rovibrational_excitation.simulation.runner import _run_one

_NUMPY_DENSE = ExecutionPolicy(backend=ArrayBackend.NUMPY, storage=MatrixStorage.DENSE)


def _build_model(params):
    return build_model(params, execution_policy=_NUMPY_DENSE)


@pytest.mark.parametrize(
    ("params", "expected_size"),
    [
        (
            {
                "basis_type": "twolevel",
                "energy_gap": 1.0,
                "energy_gap_units": "rad/fs",
                "mu0_Cm": 1e-30,
                "initial_states": [0],
            },
            2,
        ),
        (
            {
                "basis_type": "vibladder",
                "V_max": 2,
                "omega_rad_phz": 1.0,
                "delta_omega_rad_phz": 0.01,
                "potential_type": "harmonic",
                "mu0_Cm": 1e-30,
                "initial_states": [0],
            },
            3,
        ),
        (
            {
                "basis_type": "linmol",
                "V_max": 1,
                "J_max": 1,
                "representation": "m_resolved",
                "axes": "xy",
                "omega_rad_phz": 1.0,
                "delta_omega_rad_phz": 0.01,
                "B_rad_phz": 0.001,
                "alpha_rad_phz": 0.0,
                "potential_type": "harmonic",
                "mu0_Cm": 1e-30,
                "initial_states": [0],
            },
            8,
        ),
    ],
)
def test_build_model_constructs_normalized_existing_components(params, expected_size):
    model = _build_model(params)

    assert model.basis.size() == expected_size
    assert model.hamiltonian.shape == (expected_size, expected_size)
    assert model.state.data.shape == (expected_size, 1)
    np.testing.assert_allclose(np.linalg.norm(model.state.data), 1.0)


def test_linmol_resolved_representation_projects_to_existing_explicit_m_basis():
    params = {
        "basis_type": "linmol",
        "V_max": 1,
        "J_max": 1,
        "representation": "m_resolved",
        "axes": "zx",
        "omega_rad_phz": 1.0,
        "delta_omega_rad_phz": 0.01,
        "alpha_rad_phz": 0.0,
        "B_rad_phz": 0.001,
        "mu0_Cm": 1e-30,
        "potential_type": "harmonic",
        "initial_states": [0],
    }

    model = _build_model(params)

    assert model.basis.use_M is True
    assert model.basis.size() == 8
    assert model.coupling.axes == ("z", "x")


def test_linmol_resolved_representation_requires_explicit_axes():
    params = {
        "basis_type": "linmol",
        "V_max": 0,
        "J_max": 0,
        "representation": "m_resolved",
        "omega_rad_phz": 1.0,
        "delta_omega_rad_phz": 0.0,
        "alpha_rad_phz": 0.0,
        "B_rad_phz": 0.0,
        "mu0_Cm": 1e-30,
        "potential_type": "harmonic",
        "initial_states": [0],
    }

    with pytest.raises(ValueError, match="m_resolved parameter: axes"):
        _build_model(params)


def test_m_incoherent_average_cannot_build_one_pure_state_model():
    params = {
        "basis_type": "linmol",
        "V_max": 0,
        "J_max": 0,
        "representation": "m_incoherent_average",
        "omega_rad_phz": 1.0,
        "delta_omega_rad_phz": 0.0,
        "alpha_rad_phz": 0.0,
        "B_rad_phz": 0.0,
        "mu0_Cm": 1e-30,
        "potential_type": "harmonic",
        "initial_states": [0],
    }

    with pytest.raises(ValueError, match="multi-block workflow"):
        _build_model(params)


def test_linmol_rejects_morse_with_zero_anharmonicity():
    params = {
        "basis_type": "linmol",
        "V_max": 1,
        "J_max": 1,
        "representation": "m_resolved",
        "axes": "xy",
        "omega_rad_phz": 1.0,
        "delta_omega_rad_phz": 0.0,
        "alpha_rad_phz": 0.0,
        "B_rad_phz": 0.001,
        "mu0_Cm": 1e-30,
        "potential_type": "morse",
        "initial_states": [0],
    }

    with pytest.raises(ValueError, match="must be non-zero"):
        _build_model(params)


def test_build_model_constructs_coherent_superposition():
    model = _build_model(
        {
            "basis_type": "twolevel",
            "energy_gap": 1.0,
            "mu0_Cm": 1e-30,
            "initial_states": [0, 1],
            "energy_gap_units": "rad/fs",
        }
    )

    expected = np.array([1.0, 1.0]) / np.sqrt(2.0)
    np.testing.assert_allclose(model.state.data.ravel(), expected)


@pytest.mark.parametrize(
    ("params", "missing"),
    [
        (
            {"basis_type": "twolevel", "energy_gap_units": "rad/fs", "mu0_Cm": 1e-30},
            "energy_gap",
        ),
        (
            {"basis_type": "twolevel", "energy_gap": 1.0, "energy_gap_units": "rad/fs"},
            "mu0_Cm",
        ),
        (
            {
                "basis_type": "vibladder",
                "V_max": 1,
                "omega_rad_phz": 1.0,
                "delta_omega_rad_phz": 0.0,
                "potential_type": "harmonic",
            },
            "mu0_Cm",
        ),
    ],
)
def test_build_model_requires_physical_scale_parameters(params, missing):
    with pytest.raises(ValueError, match=missing):
        _build_model(params)


def test_build_model_rejects_unknown_basis_type():
    with pytest.raises(ValueError, match="Unknown basis_type"):
        _build_model({"basis_type": "unknown"})


def test_build_model_preserves_missing_parameter_error():
    with pytest.raises(ValueError, match="V_max"):
        _build_model(
            {
                "basis_type": "linmol",
                "representation": "m_resolved",
                "axes": "xy",
            }
        )


@pytest.mark.parametrize(
    "model_params",
    [
        {
            "basis_type": "twolevel",
            "energy_gap": 1.0,
            "energy_gap_units": "rad/fs",
            "mu0_Cm": 1e-30,
        },
        {
            "basis_type": "vibladder",
            "V_max": 1,
            "omega_rad_phz": 1.0,
            "delta_omega_rad_phz": 0.0,
            "mu0_Cm": 1e-30,
            "potential_type": "harmonic",
        },
        {
            "basis_type": "linmol",
            "V_max": 0,
            "J_max": 0,
            "representation": "m_incoherent_average",
            "omega_rad_phz": 1.0,
            "delta_omega_rad_phz": 0.0,
            "B_rad_phz": 0.0,
            "alpha_rad_phz": 0.0,
            "mu0_Cm": 1e-30,
            "potential_type": "harmonic",
        },
    ],
)
def test_runner_zero_field_preserves_population_after_model_split(model_params):
    params = {
        "t_start": 0.0,
        "t_end": 0.2,
        "dt": 0.1,
        "duration": 0.1,
        "t_center": 0.1,
        "carrier_freq": 1.0,
        "amplitude": 0.0,
        "polarization": [1.0, 0.0],
        "initial_states": [0],
        "return_time_psi": True,
        "save": False,
        "backend": "numpy",
        "storage": "dense",
        "algorithm": "rk4",
        "return_traj": True,
        "sample_stride": 1,
        "nondimensional": False,
        "renorm": False,
        **model_params,
    }

    population = _run_one(params)

    assert population.ndim == 2
    np.testing.assert_allclose(
        np.sum(population, axis=1),
        1.0,
        atol=1e-7,
    )


@patch("rovibrational_excitation.dynamics.schrodinger.SchrodingerPropagator")
@patch("rovibrational_excitation.fields.ElectricField")
def test_runner_uses_interval_duration_and_one_backend(
    electric_field_cls, propagator_cls
):
    real_grid = TimeGrid.from_bounds(2.0, 6.0, 1.0)
    real_field = RealElectricField.from_time_grid(real_grid)
    real_field.add_dispersed_Efield = MagicMock(wraps=real_field.add_dispersed_Efield)
    electric_field_cls.from_time_grid.return_value = real_field
    host_result = MagicMock(
        times_fs=np.array([2.0, 6.0]),
        state=np.array([[1.0 + 0.0j, 0.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j]]),
    )
    propagation_result = MagicMock()
    propagation_result.to_numpy.return_value = host_result
    propagator_cls.return_value.propagate.return_value = propagation_result
    params = {
        "basis_type": "twolevel",
        "energy_gap": 1.0,
        "mu0_Cm": 1e-30,
        "energy_gap_units": "rad/fs",
        "t_start": 2.0,
        "t_end": 6.0,
        "dt": 1.0,
        "carrier_freq": 1.0,
        "duration": 2.0,
        "amplitude": 0.0,
        "polarization": [1.0, 0.0],
        "phase_rad": 0.37,
        "initial_states": [0],
        "backend": "numpy",
        "save": False,
        "algorithm": "split_operator",
        "storage": "csr",
        "return_traj": True,
        "nondimensional": False,
        "renorm": True,
        "verbose": True,
        "validate_units": False,
        "sample_stride": 2,
    }

    _run_one(params)

    field = electric_field_cls.from_time_grid.return_value
    electric_field_cls.from_time_grid.assert_called_once()
    assert field.add_dispersed_Efield.call_args.kwargs["duration"] == 2.0
    assert field.add_dispersed_Efield.call_args.kwargs["phase_rad"] == 0.37
    propagator_cls.assert_called_once_with(
        backend="numpy",
        algorithm="split_operator",
        split_interaction="cartesian",
        validate_units=False,
        renorm=True,
        sparse=True,
    )
    propagate_call = propagator_cls.return_value.propagate.call_args
    problem = propagate_call.args[0]
    assert isinstance(problem.initial_state, PureState)
    np.testing.assert_array_equal(problem.initial_state.amplitudes, [1.0, 0.0])
    propagate_kwargs = propagate_call.kwargs
    options = propagate_kwargs["options"]
    assert options.algorithm_name == "split_operator"
    assert options.backend_name == "numpy"
    assert options.renorm is True
    assert options.sparse is True
    assert options.sample_stride == 2
    assert options.return_trajectory is True
    assert options.nondimensional is False
    assert propagate_kwargs["split_interaction"] == "cartesian"
    assert propagate_kwargs["verbose"] is True
    assert problem.coupling_mode == "scalar"
    assert problem.coupling_kwargs == {"coupling_axis": "x"}
