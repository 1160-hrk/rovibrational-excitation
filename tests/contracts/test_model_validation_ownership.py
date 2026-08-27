"""Ownership and exception contracts for model configuration validation."""

from __future__ import annotations

import numpy as np
import pytest

from rovibrational_excitation.models.validation import (
    LinMolRepresentation,
    ModelConfigurationError,
    validate_linmol_representation,
    validate_model_parameters,
)
from rovibrational_excitation.simulation.validation import (
    SimulationConfigurationError,
    validate_simulation_case,
)


def _linmol_params(**overrides):
    params = {
        "basis_type": "linmol",
        "V_max": 1,
        "J_max": 1,
        "vibrational_frequency": 1.0,
        "vibrational_frequency_units": "rad/fs",
        "anharmonic_shift": 0.01,
        "anharmonic_shift_units": "rad/fs",
        "rotational_constant": 0.001,
        "rotational_constant_units": "rad/fs",
        "vibration_rotation_coupling": 0.0,
        "vibration_rotation_coupling_units": "rad/fs",
        "mu0_Cm": 1.0e-30,
        "potential_type": "harmonic",
        "initial_states": [0],
    }
    params.update(overrides)
    return params


def test_linmol_representation_is_required_and_use_m_is_removed():
    with pytest.raises(ModelConfigurationError, match="representation"):
        validate_model_parameters(_linmol_params())

    with pytest.raises(ModelConfigurationError, match="use_M was removed"):
        validate_model_parameters(_linmol_params(use_M=True))

    with pytest.raises(ModelConfigurationError, match="use_M was removed"):
        validate_model_parameters(
            {
                "basis_type": "twolevel",
                "energy_gap": 0.2,
                "energy_gap_units": "rad/fs",
                "mu0_Cm": 3.0e-30,
                "initial_states": [0],
                "use_M": False,
            }
        )


@pytest.mark.parametrize("value", ["explicit_m", "average", True, None])
def test_linmol_representation_rejects_unknown_or_non_string_values(value):
    with pytest.raises(
        ModelConfigurationError,
        match="m_resolved.*m_incoherent_average",
    ):
        validate_model_parameters(_linmol_params(representation=value))


def test_linmol_representation_parser_returns_typed_accepted_values():
    assert (
        validate_linmol_representation(_linmol_params(representation="m_resolved"))
        is LinMolRepresentation.M_RESOLVED
    )
    assert (
        validate_linmol_representation(
            _linmol_params(representation="m_incoherent_average")
        )
        is LinMolRepresentation.M_INCOHERENT_AVERAGE
    )


def test_model_validation_requires_explicit_model_and_initial_state():
    with pytest.raises(
        ModelConfigurationError,
        match="Missing required model parameter: basis_type",
    ):
        validate_model_parameters({})

    with pytest.raises(
        ModelConfigurationError,
        match="Missing required model parameter: initial_states",
    ):
        validate_model_parameters(
            {
                "basis_type": "twolevel",
                "energy_gap": 0.2,
                "energy_gap_units": "rad/fs",
                "mu0_Cm": 3.0e-30,
            }
        )


def test_model_frequency_schema_requires_neutral_names_and_units():
    params = {
        "basis_type": "linmol",
        "V_max": 1,
        "J_max": 1,
        "representation": "m_resolved",
        "vibrational_frequency": 100.0,
        "vibrational_frequency_units": "THz",
        "anharmonic_shift": 1.0,
        "anharmonic_shift_units": "THz",
        "rotational_constant": 0.1,
        "rotational_constant_units": "THz",
        "vibration_rotation_coupling": 0.0,
        "vibration_rotation_coupling_units": "THz",
        "mu0_Cm": 1.0e-30,
        "potential_type": "harmonic",
        "initial_states": [0],
    }

    assert validate_model_parameters(params) == "linmol"


@pytest.mark.parametrize(
    "key",
    [
        "vibrational_frequency_units",
        "anharmonic_shift_units",
        "rotational_constant_units",
        "vibration_rotation_coupling_units",
    ],
)
def test_model_frequency_schema_requires_each_unit(key):
    params = _linmol_params(representation="m_resolved")
    del params[key]

    with pytest.raises(ModelConfigurationError, match=key):
        validate_model_parameters(params)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("vibrational_frequency", np.nan),
        ("vibrational_frequency", np.array([1.0])),
        ("anharmonic_shift", True),
        ("rotational_constant_units", "cycles_per_second"),
    ],
)
def test_model_frequency_schema_rejects_invalid_quantities(key, value):
    params = _linmol_params(representation="m_resolved")
    params[key] = value

    with pytest.raises(ModelConfigurationError, match=key):
        validate_model_parameters(params)


def test_model_schema_rejects_morse_with_zero_shift_before_construction():
    params = _linmol_params(
        representation="m_resolved",
        potential_type="morse",
        anharmonic_shift=0.0,
    )

    with pytest.raises(ModelConfigurationError, match="anharmonic_shift.*non-zero"):
        validate_model_parameters(params)


def test_model_frequency_schema_rejects_removed_unit_encoded_names():
    with pytest.raises(ModelConfigurationError, match="omega_rad_phz was removed"):
        validate_model_parameters(
            _linmol_params(
                omega_rad_phz=1.0,
                representation="m_resolved",
                vibrational_frequency=1.0,
                vibrational_frequency_units="rad/fs",
                anharmonic_shift=0.01,
                anharmonic_shift_units="rad/fs",
                rotational_constant=0.001,
                rotational_constant_units="rad/fs",
                vibration_rotation_coupling=0.0,
                vibration_rotation_coupling_units="rad/fs",
            )
        )


def test_twolevel_schema_rejects_unknown_energy_gap_unit_before_construction():
    with pytest.raises(ModelConfigurationError, match="energy_gap_units"):
        validate_model_parameters(
            {
                "basis_type": "twolevel",
                "energy_gap": 0.2,
                "energy_gap_units": "cycles_per_second",
                "mu0_Cm": 3.0e-30,
                "initial_states": [0],
            }
        )


def test_simulation_boundary_translates_missing_model_selection():
    with pytest.raises(
        SimulationConfigurationError,
        match="Missing required model parameter: basis_type",
    ) as captured:
        validate_simulation_case({})

    assert isinstance(captured.value.__cause__, ModelConfigurationError)


def test_model_validation_is_owned_by_models_and_normalizes_the_existing_key():
    params = {
        "basis_type": "TwoLevel",
        "energy_gap": 0.2,
        "energy_gap_units": "rad/fs",
        "mu0_Cm": 3.0e-30,
        "initial_states": [0],
    }

    assert validate_model_parameters(params) == "twolevel"
    assert validate_model_parameters.__module__ == (
        "rovibrational_excitation.models.validation"
    )


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"basis_type": 1}, "basis_type must be a string"),
        ({"basis_type": "unknown"}, "Unknown basis_type: unknown"),
        (
            {"basis_type": "twolevel"},
            "Missing required model parameters: energy_gap, energy_gap_units, mu0_Cm",
        ),
        (
            {
                "basis_type": "vibladder",
                "V_max": 2,
                "vibrational_frequency": 0.2,
                "vibrational_frequency_units": "rad/fs",
                "anharmonic_shift": 0.0,
                "anharmonic_shift_units": "rad/fs",
                "mu0_Cm": 3.0e-30,
                "potential_type": "quadratic",
                "initial_states": [0],
            },
            "potential_type must be 'harmonic' or 'morse'",
        ),
    ],
)
def test_model_validation_preserves_existing_failures(params, message):
    with pytest.raises(ModelConfigurationError, match=message):
        validate_model_parameters(params)


def test_simulation_boundary_translates_model_error_without_changing_message():
    with pytest.raises(
        SimulationConfigurationError,
        match="Missing required model parameters: energy_gap, energy_gap_units, mu0_Cm",
    ) as captured:
        validate_simulation_case({"basis_type": "twolevel"})

    assert isinstance(captured.value.__cause__, ModelConfigurationError)
