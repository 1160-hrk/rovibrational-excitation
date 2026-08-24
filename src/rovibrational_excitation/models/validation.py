"""Model selection and required-input validation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class ModelConfigurationError(ValueError):
    """Raised before construction when model configuration is invalid."""


_MODEL_REQUIRED = {
    "linmol": {
        "V_max",
        "J_max",
        "omega_rad_phz",
        "delta_omega_rad_phz",
        "B_rad_phz",
        "alpha_rad_phz",
        "mu0_Cm",
        "potential_type",
    },
    "twolevel": {"energy_gap", "energy_gap_units", "mu0_Cm"},
    "vibladder": {
        "V_max",
        "omega_rad_phz",
        "delta_omega_rad_phz",
        "mu0_Cm",
        "potential_type",
    },
}


def validate_model_parameters(params: Mapping[str, Any]) -> str:
    """Validate model selection and physically defining model inputs."""
    if "basis_type" not in params:
        raise ModelConfigurationError("Missing required model parameter: basis_type")
    basis_type_raw = params["basis_type"]
    if not isinstance(basis_type_raw, str):
        raise ModelConfigurationError("basis_type must be a string")
    basis_type = basis_type_raw.lower()
    if basis_type not in _MODEL_REQUIRED:
        raise ModelConfigurationError(f"Unknown basis_type: {basis_type}")

    missing = sorted(_MODEL_REQUIRED[basis_type] - params.keys())
    if missing:
        raise ModelConfigurationError(
            "Missing required model parameters: " + ", ".join(missing)
        )
    if "initial_states" not in params:
        raise ModelConfigurationError(
            "Missing required model parameter: initial_states"
        )
    if basis_type in {"linmol", "vibladder"} and params["potential_type"] not in {
        "harmonic",
        "morse",
    }:
        raise ModelConfigurationError("potential_type must be 'harmonic' or 'morse'")
    return basis_type
