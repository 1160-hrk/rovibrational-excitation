"""Regression tests for model construction extracted from simulation.runner."""

from dataclasses import FrozenInstanceError
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
from rovibrational_excitation.models import LinMolParameters, build_model
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
                "dipole_scale": 1e-30,
                "dipole_scale_units": "C*m",
                "initial_states": [0],
            },
            2,
        ),
        (
            {
                "basis_type": "vibladder",
                "V_max": 2,
                "vibrational_frequency": 1.0,
                "vibrational_frequency_units": "rad/fs",
                "anharmonic_shift": 0.01,
                "anharmonic_shift_units": "rad/fs",
                "potential_type": "harmonic",
                "dipole_scale": 1e-30,
                "dipole_scale_units": "C*m",
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
                "vibrational_frequency": 1.0,
                "vibrational_frequency_units": "rad/fs",
                "anharmonic_shift": 0.01,
                "anharmonic_shift_units": "rad/fs",
                "rotational_constant": 0.001,
                "rotational_constant_units": "rad/fs",
                "vibration_rotation_coupling": 0.0,
                "vibration_rotation_coupling_units": "rad/fs",
                "potential_type": "harmonic",
                "dipole_scale": 1e-30,
                "dipole_scale_units": "C*m",
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


def _frequency_value(rad_per_fs, unit):
    factors = {
        "rad/fs": 1.0,
        "PHz": 2.0 * np.pi,
        "THz": 2.0 * np.pi * 1.0e-3,
        "cm^-1": 2.0 * np.pi * 2.99792458e8 * 1.0e-13,
    }
    return rad_per_fs / factors[unit]


def _linmol_frequency_params(unit):
    return {
        "basis_type": "linmol",
        "V_max": 1,
        "J_max": 1,
        "representation": "m_resolved",
        "axes": "xy",
        "vibrational_frequency": _frequency_value(0.2, unit),
        "vibrational_frequency_units": unit,
        "anharmonic_shift": _frequency_value(0.01, unit),
        "anharmonic_shift_units": unit,
        "rotational_constant": _frequency_value(0.001, unit),
        "rotational_constant_units": unit,
        "vibration_rotation_coupling": _frequency_value(0.0001, unit),
        "vibration_rotation_coupling_units": unit,
        "dipole_scale": 1.0e-30,
        "dipole_scale_units": "C*m",
        "potential_type": "morse",
        "initial_states": [0],
    }


@pytest.mark.parametrize("unit", ["PHz", "THz", "cm^-1"])
def test_linmol_frequency_units_preserve_model_arrays(unit):
    reference = _build_model(_linmol_frequency_params("rad/fs"))
    actual = _build_model(_linmol_frequency_params(unit))

    np.testing.assert_array_equal(actual.basis.basis, reference.basis.basis)
    np.testing.assert_allclose(
        actual.hamiltonian.matrix,
        reference.hamiltonian.matrix,
        rtol=3.0e-15,
        atol=0.0,
    )
    for axis in ("x", "y", "z"):
        np.testing.assert_allclose(
            actual.dipole.mu(axis),
            reference.dipole.mu(axis),
            rtol=4.0e-15,
            atol=0.0,
        )


@pytest.mark.parametrize("unit", ["PHz", "THz", "cm^-1"])
def test_vibladder_frequency_units_preserve_model_arrays(unit):
    def params(selected_unit):
        return {
            "basis_type": "vibladder",
            "V_max": 3,
            "vibrational_frequency": _frequency_value(0.2, selected_unit),
            "vibrational_frequency_units": selected_unit,
            "anharmonic_shift": _frequency_value(0.01, selected_unit),
            "anharmonic_shift_units": selected_unit,
            "dipole_scale": 1.0e-30,
            "dipole_scale_units": "C*m",
            "potential_type": "morse",
            "initial_states": [0],
        }

    reference = _build_model(params("rad/fs"))
    actual = _build_model(params(unit))

    np.testing.assert_array_equal(actual.basis.basis, reference.basis.basis)
    np.testing.assert_allclose(
        actual.hamiltonian.matrix,
        reference.hamiltonian.matrix,
        rtol=3.0e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(
        actual.dipole.mu("z"),
        reference.dipole.mu("z"),
        rtol=4.0e-15,
        atol=0.0,
    )


def test_linmol_parameters_are_frozen():
    model_params = LinMolParameters.from_mapping(_linmol_frequency_params("rad/fs"))

    with pytest.raises(FrozenInstanceError):
        model_params.v_max = 4


@pytest.mark.parametrize(
    ("energy_gap", "energy_gap_units"),
    [
        (0.2 / (2.0 * np.pi), "PHz"),
        (0.2 / (2.0 * np.pi * 1.0e-3), "THz"),
        (0.2 / (2.0 * np.pi * 2.99792458e8 * 1.0e-13), "cm^-1"),
    ],
)
def test_twolevel_energy_gap_units_preserve_model_arrays(energy_gap, energy_gap_units):
    reference = _build_model(
        {
            "basis_type": "twolevel",
            "energy_gap": 0.2,
            "energy_gap_units": "rad/fs",
            "dipole_scale": 1.0e-30,
            "dipole_scale_units": "C*m",
            "initial_states": [0],
        }
    )
    actual = _build_model(
        {
            "basis_type": "twolevel",
            "energy_gap": energy_gap,
            "energy_gap_units": energy_gap_units,
            "dipole_scale": 1.0e-30,
            "dipole_scale_units": "C*m",
            "initial_states": [0],
        }
    )

    np.testing.assert_allclose(
        actual.hamiltonian.matrix,
        reference.hamiltonian.matrix,
        rtol=3.0e-15,
        atol=0.0,
    )
    np.testing.assert_array_equal(actual.dipole.mu("x"), reference.dipole.mu("x"))


def test_linmol_resolved_representation_projects_to_existing_explicit_m_basis():
    params = {
        "basis_type": "linmol",
        "V_max": 1,
        "J_max": 1,
        "representation": "m_resolved",
        "axes": "zx",
        "vibrational_frequency": 1.0,
        "vibrational_frequency_units": "rad/fs",
        "anharmonic_shift": 0.01,
        "anharmonic_shift_units": "rad/fs",
        "vibration_rotation_coupling": 0.0,
        "vibration_rotation_coupling_units": "rad/fs",
        "rotational_constant": 0.001,
        "rotational_constant_units": "rad/fs",
        "dipole_scale": 1e-30,
        "dipole_scale_units": "C*m",
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
        "vibrational_frequency": 1.0,
        "vibrational_frequency_units": "rad/fs",
        "anharmonic_shift": 0.0,
        "anharmonic_shift_units": "rad/fs",
        "vibration_rotation_coupling": 0.0,
        "vibration_rotation_coupling_units": "rad/fs",
        "rotational_constant": 0.0,
        "rotational_constant_units": "rad/fs",
        "dipole_scale": 1e-30,
        "dipole_scale_units": "C*m",
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
        "vibrational_frequency": 1.0,
        "vibrational_frequency_units": "rad/fs",
        "anharmonic_shift": 0.0,
        "anharmonic_shift_units": "rad/fs",
        "vibration_rotation_coupling": 0.0,
        "vibration_rotation_coupling_units": "rad/fs",
        "rotational_constant": 0.0,
        "rotational_constant_units": "rad/fs",
        "dipole_scale": 1e-30,
        "dipole_scale_units": "C*m",
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
        "vibrational_frequency": 1.0,
        "vibrational_frequency_units": "rad/fs",
        "anharmonic_shift": 0.0,
        "anharmonic_shift_units": "rad/fs",
        "vibration_rotation_coupling": 0.0,
        "vibration_rotation_coupling_units": "rad/fs",
        "rotational_constant": 0.001,
        "rotational_constant_units": "rad/fs",
        "dipole_scale": 1e-30,
        "dipole_scale_units": "C*m",
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
            "dipole_scale": 1e-30,
            "dipole_scale_units": "C*m",
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
            {
                "basis_type": "twolevel",
                "energy_gap_units": "rad/fs",
                "dipole_scale": 1e-30,
                "dipole_scale_units": "C*m",
            },
            "energy_gap",
        ),
        (
            {
                "basis_type": "twolevel",
                "energy_gap": 1.0,
                "energy_gap_units": "rad/fs",
                "dipole_scale": 1e-30,
            },
            "dipole_scale_units",
        ),
        (
            {"basis_type": "twolevel", "energy_gap": 1.0, "energy_gap_units": "rad/fs"},
            "dipole_scale",
        ),
        (
            {
                "basis_type": "vibladder",
                "V_max": 1,
                "vibrational_frequency": 1.0,
                "vibrational_frequency_units": "rad/fs",
                "anharmonic_shift": 0.0,
                "anharmonic_shift_units": "rad/fs",
                "potential_type": "harmonic",
            },
            "dipole_scale",
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
            "dipole_scale": 1e-30,
            "dipole_scale_units": "C*m",
        },
        {
            "basis_type": "vibladder",
            "V_max": 1,
            "vibrational_frequency": 1.0,
            "vibrational_frequency_units": "rad/fs",
            "anharmonic_shift": 0.0,
            "anharmonic_shift_units": "rad/fs",
            "dipole_scale": 1e-30,
            "dipole_scale_units": "C*m",
            "potential_type": "harmonic",
        },
        {
            "basis_type": "linmol",
            "V_max": 0,
            "J_max": 0,
            "representation": "m_incoherent_average",
            "vibrational_frequency": 1.0,
            "vibrational_frequency_units": "rad/fs",
            "anharmonic_shift": 0.0,
            "anharmonic_shift_units": "rad/fs",
            "rotational_constant": 0.0,
            "rotational_constant_units": "rad/fs",
            "vibration_rotation_coupling": 0.0,
            "vibration_rotation_coupling_units": "rad/fs",
            "dipole_scale": 1e-30,
            "dipole_scale_units": "C*m",
            "potential_type": "harmonic",
        },
    ],
)
def test_runner_zero_field_preserves_population_after_model_split(model_params):
    params = {
        "t_start": 0.0,
        "t_start_units": "fs",
        "t_end": 0.2,
        "t_end_units": "fs",
        "dt": 0.1,
        "dt_units": "fs",
        "duration": 0.1,
        "duration_units": "fs",
        "t_center": 0.1,
        "t_center_units": "fs",
        "envelope_kind": "gaussian_fwhm",
        "modulation_kind": "none",
        "carrier_frequency": 1.0,
        "carrier_frequency_units": "PHz",
        "amplitude": 0.0,
        "amplitude_units": "V/m",
        "initial_states": [0],
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
        "dipole_scale": 1e-30,
        "dipole_scale_units": "C*m",
        "energy_gap_units": "rad/fs",
        "t_start": 2.0,
        "t_start_units": "fs",
        "t_end": 6.0,
        "t_end_units": "fs",
        "dt": 1.0,
        "dt_units": "fs",
        "carrier_frequency": 1.0,
        "carrier_frequency_units": "PHz",
        "duration": 2.0,
        "duration_units": "fs",
        "t_center": 0.0,
        "t_center_units": "fs",
        "envelope_kind": "gaussian_fwhm",
        "modulation_kind": "none",
        "amplitude": 0.0,
        "amplitude_units": "V/m",
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


def test_model_dipole_scale_units_preserve_dipole_array():
    canonical = {
        "basis_type": "twolevel",
        "energy_gap": 0.2,
        "energy_gap_units": "rad/fs",
        "dipole_scale": 1.0e-30,
        "dipole_scale_units": "C*m",
        "initial_states": [0],
    }
    in_debye = {
        **canonical,
        "dipole_scale": 1.0e-30 / 3.33564e-30,
        "dipole_scale_units": "D",
    }

    reference = _build_model(canonical)
    actual = _build_model(in_debye)

    np.testing.assert_allclose(
        actual.dipole.mu("z"),
        reference.dipole.mu("z"),
        rtol=2.0e-15,
        atol=0.0,
    )
