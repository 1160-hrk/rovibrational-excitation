"""Model selection and required-input validation."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Any

from .parameters import (
    LinMolParameters,
    TwoLevelParameters,
    VibLadderParameters,
)

ModelParameters = LinMolParameters | TwoLevelParameters | VibLadderParameters


class ModelConfigurationError(ValueError):
    """Raised before construction when model configuration is invalid."""


class LinMolRepresentation(str, Enum):
    """Explicit normal-simulation treatment of magnetic degeneracy."""

    M_RESOLVED = "m_resolved"
    M_INCOHERENT_AVERAGE = "m_incoherent_average"


def validate_linmol_representation(
    params: Mapping[str, Any],
) -> LinMolRepresentation:
    """Return the required LinMol representation without inferring a default.

    The removed ``use_M`` boolean is rejected even if ``representation`` is
    also present, so stale configurations cannot appear to use the new schema.
    """
    if "use_M" in params:
        raise ModelConfigurationError(
            "use_M was removed; use required representation="
            "m_resolved or m_incoherent_average"
        )
    if "representation" not in params:
        raise ModelConfigurationError(
            "Missing required LinMol model parameter: representation "
            "(m_resolved or m_incoherent_average)"
        )
    value = params["representation"]
    if not isinstance(value, str):
        raise ModelConfigurationError(
            "representation must be m_resolved or m_incoherent_average"
        )
    try:
        return LinMolRepresentation(value)
    except ValueError as exc:
        raise ModelConfigurationError(
            "representation must be m_resolved or m_incoherent_average"
        ) from exc


_MODEL_REQUIRED = {
    "linmol": {
        "V_max",
        "J_max",
        "vibrational_frequency",
        "vibrational_frequency_units",
        "anharmonic_shift",
        "anharmonic_shift_units",
        "rotational_constant",
        "rotational_constant_units",
        "vibration_rotation_coupling",
        "vibration_rotation_coupling_units",
        "mu0_Cm",
        "potential_type",
    },
    "twolevel": {"energy_gap", "energy_gap_units", "mu0_Cm"},
    "vibladder": {
        "V_max",
        "vibrational_frequency",
        "vibrational_frequency_units",
        "anharmonic_shift",
        "anharmonic_shift_units",
        "mu0_Cm",
        "potential_type",
    },
}


_REMOVED_FREQUENCY_KEYS = {
    "omega_rad_phz": "vibrational_frequency with vibrational_frequency_units",
    "delta_omega_rad_phz": "anharmonic_shift with anharmonic_shift_units",
    "B_rad_phz": "rotational_constant with rotational_constant_units",
    "alpha_rad_phz": (
        "vibration_rotation_coupling with vibration_rotation_coupling_units"
    ),
    "vibrational_frequency_rad_per_fs": (
        "vibrational_frequency with vibrational_frequency_units"
    ),
    "rotational_constant_rad_per_fs": (
        "rotational_constant with rotational_constant_units"
    ),
    "vibration_rotation_coupling_rad_per_fs": (
        "vibration_rotation_coupling with vibration_rotation_coupling_units"
    ),
    "anharmonicity_correction_rad_per_fs": (
        "anharmonic_shift with anharmonic_shift_units"
    ),
}
_REMOVED_FREQUENCY_KEYS.update(
    {
        f"{key}_units": replacement
        for key, replacement in tuple(_REMOVED_FREQUENCY_KEYS.items())
    }
)


def _construct_model_parameters(
    basis_type: str, params: Mapping[str, Any]
) -> ModelParameters:
    if basis_type == "linmol":
        return LinMolParameters.from_mapping(params)
    if basis_type == "twolevel":
        return TwoLevelParameters.from_mapping(params)
    return VibLadderParameters.from_mapping(params)


def validate_model_parameters(params: Mapping[str, Any]) -> str:
    """Validate model selection and physically defining model inputs."""
    removed = sorted(_REMOVED_FREQUENCY_KEYS.keys() & params.keys())
    if removed:
        key = removed[0]
        raise ModelConfigurationError(
            f"{key} was removed; use {_REMOVED_FREQUENCY_KEYS[key]}"
        )
    if "use_M" in params:
        raise ModelConfigurationError(
            "use_M was removed from normal simulation; use required LinMol "
            "representation=m_resolved or representation=m_incoherent_average"
        )
    if "basis_type" not in params:
        raise ModelConfigurationError("Missing required model parameter: basis_type")
    basis_type_raw = params["basis_type"]
    if not isinstance(basis_type_raw, str):
        raise ModelConfigurationError("basis_type must be a string")
    basis_type = basis_type_raw.lower()
    if basis_type not in _MODEL_REQUIRED:
        raise ModelConfigurationError(f"Unknown basis_type: {basis_type}")

    if basis_type == "linmol":
        validate_linmol_representation(params)

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
    try:
        _construct_model_parameters(basis_type, params)
    except (TypeError, ValueError) as exc:
        raise ModelConfigurationError(str(exc)) from exc
    return basis_type


def model_parameters_from_mapping(
    params: Mapping[str, Any],
) -> LinMolParameters | TwoLevelParameters | VibLadderParameters:
    """Return the frozen parameters after applying the public validation."""
    basis_type = validate_model_parameters(params)
    return _construct_model_parameters(basis_type, params)
