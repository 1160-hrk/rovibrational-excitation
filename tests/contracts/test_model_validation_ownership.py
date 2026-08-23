"""Ownership and exception contracts for model configuration validation."""

from __future__ import annotations

import pytest

from rovibrational_excitation.models.validation import (
    ModelConfigurationError,
    validate_model_parameters,
)
from rovibrational_excitation.simulation.validation import (
    SimulationConfigurationError,
    validate_simulation_case,
)


def test_model_validation_is_owned_by_models_and_normalizes_the_existing_key():
    params = {
        "basis_type": "TwoLevel",
        "energy_gap": 0.2,
        "energy_gap_units": "rad/fs",
        "mu0_Cm": 3.0e-30,
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
                "omega_rad_phz": 0.2,
                "delta_omega_rad_phz": 0.0,
                "mu0_Cm": 3.0e-30,
                "potential_type": "quadratic",
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
