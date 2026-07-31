"""Regression tests for model construction extracted from simulation.runner."""

from unittest.mock import patch

import numpy as np
import pytest

from rovibrational_excitation.simulation.models import build_model
from rovibrational_excitation.simulation.runner import _run_one


@pytest.mark.parametrize(
    ("params", "expected_size"),
    [
        (
            {
                "basis_type": "twolevel",
                "energy_gap": 1.0,
                "energy_gap_units": "rad/fs",
                "mu0_Cm": 1e-30,
            },
            2,
        ),
        (
            {
                "basis_type": "vibladder",
                "V_max": 2,
                "omega_rad_phz": 1.0,
                "delta_omega_rad_phz": 0.01,
                "mu0_Cm": 1e-30,
            },
            3,
        ),
        (
            {
                "basis_type": "linmol",
                "V_max": 1,
                "J_max": 1,
                "use_M": False,
                "omega_rad_phz": 1.0,
                "delta_omega_rad_phz": 0.01,
                "B_rad_phz": 0.001,
                "mu0_Cm": 1e-30,
            },
            4,
        ),
    ],
)
def test_build_model_constructs_normalized_existing_components(params, expected_size):
    model = build_model(params)

    assert model.basis.size() == expected_size
    assert model.hamiltonian.shape == (expected_size, expected_size)
    assert model.state.data.shape == (expected_size, 1)
    np.testing.assert_allclose(np.linalg.norm(model.state.data), 1.0)


def test_linmol_rejects_morse_with_zero_anharmonicity():
    params = {
        "basis_type": "linmol",
        "V_max": 1,
        "J_max": 1,
        "use_M": False,
        "omega_rad_phz": 1.0,
        "delta_omega_rad_phz": 0.0,
        "B_rad_phz": 0.001,
        "mu0_Cm": 1e-30,
        "potential_type": "morse",
    }

    with pytest.raises(ValueError, match="must be non-zero"):
        build_model(params)


def test_build_model_constructs_coherent_superposition():
    model = build_model(
        {
            "basis_type": "twolevel",
            "energy_gap": 1.0,
            "mu0_Cm": 1e-30,
            "initial_states": [0, 1],
        }
    )

    expected = np.array([1.0, 1.0]) / np.sqrt(2.0)
    np.testing.assert_allclose(model.state.data.ravel(), expected)


@pytest.mark.parametrize(
    ("params", "missing"),
    [
        ({"basis_type": "twolevel", "mu0_Cm": 1e-30}, "energy_gap"),
        ({"basis_type": "twolevel", "energy_gap": 1.0}, "mu0_Cm"),
        (
            {
                "basis_type": "vibladder",
                "V_max": 1,
                "omega_rad_phz": 1.0,
            },
            "mu0_Cm",
        ),
    ],
)
def test_build_model_requires_physical_scale_parameters(params, missing):
    with pytest.raises(KeyError, match=missing):
        build_model(params)


def test_build_model_rejects_unknown_basis_type():
    with pytest.raises(ValueError, match="Unknown basis_type"):
        build_model({"basis_type": "unknown"})


def test_build_model_preserves_missing_parameter_error():
    with pytest.raises(KeyError, match="V_max"):
        build_model({"basis_type": "linmol"})


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
        },
        {
            "basis_type": "linmol",
            "V_max": 0,
            "J_max": 0,
            "use_M": False,
            "omega_rad_phz": 1.0,
            "delta_omega_rad_phz": 0.0,
            "B_rad_phz": 0.0,
            "alpha_rad_phz": 0.0,
            "mu0_Cm": 1e-30,
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
        **model_params,
    }

    population = _run_one(params)

    assert population.ndim == 2
    np.testing.assert_allclose(
        np.sum(population, axis=1),
        1.0,
        atol=1e-7,
    )


@patch("rovibrational_excitation.core.propagation.schrodinger.SchrodingerPropagator")
@patch("rovibrational_excitation.core.electric_field.ElectricField")
def test_runner_uses_interval_duration_and_one_backend(
    electric_field_cls, propagator_cls
):
    propagator_cls.return_value.propagate.return_value = (
        np.array([0.0]),
        np.array([[1.0 + 0.0j, 0.0 + 0.0j]]),
    )
    params = {
        "basis_type": "twolevel",
        "energy_gap": 1.0,
        "mu0_Cm": 1e-30,
        "t_start": 2.0,
        "t_end": 6.0,
        "dt": 1.0,
        "carrier_freq": 1.0,
        "amplitude": 0.0,
        "polarization": [1.0, 0.0],
        "initial_states": [0],
        "backend": "numpy",
        "save": False,
        "algorithm": "split_operator",
        "renorm": True,
        "dense": False,
        "auto_timestep": True,
        "target_accuracy": "fast",
        "verbose": True,
        "validate_units": False,
        "sample_stride": 2,
    }

    _run_one(params)

    field = electric_field_cls.return_value
    assert field.add_dispersed_Efield.call_args.kwargs["duration"] == 2.0
    propagator_cls.assert_called_once_with(
        backend="numpy",
        algorithm="split_operator",
        validate_units=False,
        renorm=True,
        sparse=True,
    )
    propagate_kwargs = propagator_cls.return_value.propagate.call_args.kwargs
    assert "backend" not in propagate_kwargs
    assert propagate_kwargs["algorithm"] == "split_operator"
    assert propagate_kwargs["renorm"] is True
    assert propagate_kwargs["sparse"] is True
    assert propagate_kwargs["auto_timestep"] is True
    assert propagate_kwargs["target_accuracy"] == "fast"
    assert propagate_kwargs["verbose"] is True
    assert propagate_kwargs["sample_stride"] == 2
    assert propagate_kwargs["return_time_psi"] is True
    assert propagate_kwargs["coupling_mode"] == "scalar"
    assert propagate_kwargs["coupling_axis"] == "x"
