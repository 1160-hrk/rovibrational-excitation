"""Strict shared model construction for optimization workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from rovibrational_excitation.core.execution import ExecutionPolicy
from rovibrational_excitation.models.linear_molecule import (
    LinMolParameters,
    build_linmol_operators_from_parameters,
)
from rovibrational_excitation.models.parameters import SymmetricTopParameters
from rovibrational_excitation.models.two_level import (
    TwoLevelParameters,
    build_twolevel_operators_from_parameters,
)
from rovibrational_excitation.models.validation import (
    LinMolRepresentation,
    ModelConfigurationError,
    known_model_input_keys,
    model_parameter_keys,
    model_parameters_from_physical_mapping,
    validate_linmol_representation,
)
from rovibrational_excitation.models.vib_ladder import (
    VibLadderParameters,
    build_vibladder_operators_from_parameters,
)


class OptimizationModelConfigurationError(ValueError):
    """Raised before an optimizer allocates a model with invalid input."""


@dataclass(frozen=True, slots=True)
class OptimizationModel:
    """Basis and operators shared with production model construction."""

    name: str
    basis: Any
    hamiltonian: Any
    dipole: Any


_TYPE_ALIASES = {
    "viblad": "vibladder",
}

_LEGACY_KEYS = {
    "omega_cm": "vibrational_frequency with vibrational_frequency_units",
    "delta_omega_cm": "anharmonic_shift with anharmonic_shift_units",
    "B_cm": "rotational_constant with rotational_constant_units",
    "alpha_cm": ("vibration_rotation_coupling with vibration_rotation_coupling_units"),
    "energy_gap_cm": "energy_gap with energy_gap_units",
    "mu0": "dipole_scale with dipole_scale_units",
    "unit_dipole": "dipole_scale_units",
    "input_units": "a unit paired with each physical quantity",
    "output_units": "the fixed internal Hamiltonian unit policy",
}


def _strict_system_mapping(system_cfg: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    unknown_sections = sorted(set(system_cfg) - {"type", "params"})
    if unknown_sections:
        raise OptimizationModelConfigurationError(
            "Unknown optimization system keys: " + ", ".join(unknown_sections)
        )
    missing_sections = sorted({"type", "params"} - set(system_cfg))
    if missing_sections:
        raise OptimizationModelConfigurationError(
            "Missing required optimization system keys: " + ", ".join(missing_sections)
        )
    raw_type = system_cfg["type"]
    if not isinstance(raw_type, str):
        raise OptimizationModelConfigurationError("system.type must be a string")
    name = raw_type.lower()
    if name in _TYPE_ALIASES:
        replacement = _TYPE_ALIASES[name]
        raise OptimizationModelConfigurationError(
            f"system.type={name} was removed; use system.type={replacement}"
        )
    if name == "symtop":
        raise OptimizationModelConfigurationError(
            "SymTop optimization is not supported yet. The production SymTop "
            "model is available through the normal simulation runner, but the "
            "optimization objective and controls have not been independently "
            "validated for that model."
        )
    if name not in {"linmol", "vibladder", "twolevel"}:
        raise OptimizationModelConfigurationError(f"Unknown system.type: {name}")
    raw_params = system_cfg["params"]
    if not isinstance(raw_params, Mapping):
        raise OptimizationModelConfigurationError("system.params must be a mapping")
    return name, dict(raw_params)


def _validate_parameter_keys(name: str, params: Mapping[str, Any]) -> None:
    legacy = sorted(_LEGACY_KEYS.keys() & params.keys())
    if legacy:
        key = legacy[0]
        raise OptimizationModelConfigurationError(
            f"{key} was removed; use {_LEGACY_KEYS[key]}"
        )
    allowed = set(model_parameter_keys(name))
    if name == "linmol":
        allowed.add("representation")
    unknown = sorted(set(params) - allowed)
    if unknown:
        known_but_inapplicable = set(unknown) & set(known_model_input_keys())
        prefix = (
            f"Model parameters not applicable to system.type={name}: "
            if known_but_inapplicable
            else "Unknown optimization model parameters: "
        )
        raise OptimizationModelConfigurationError(prefix + ", ".join(unknown))


def build_optimization_model(system_cfg: Mapping[str, Any]) -> OptimizationModel:
    """Build optimizer basis/operators through the frozen production schemas."""
    if not isinstance(system_cfg, Mapping):
        raise OptimizationModelConfigurationError("system must be a mapping")
    name, params = _strict_system_mapping(system_cfg)
    _validate_parameter_keys(name, params)
    merged = {"basis_type": name, **params}
    try:
        if name == "linmol":
            representation = validate_linmol_representation(merged)
            if representation is not LinMolRepresentation.M_RESOLVED:
                raise OptimizationModelConfigurationError(
                    "optimization currently requires "
                    "representation=m_resolved; M-incoherent averaging is a "
                    "separate multi-block workflow"
                )
        model_params = model_parameters_from_physical_mapping(merged)
    except ModelConfigurationError as exc:
        raise OptimizationModelConfigurationError(str(exc)) from exc

    execution = ExecutionPolicy.from_strings(backend="numpy", storage="csr")
    if isinstance(model_params, LinMolParameters):
        basis, hamiltonian, dipole = build_linmol_operators_from_parameters(
            model_params,
            execution_policy=execution,
        )
    elif isinstance(model_params, VibLadderParameters):
        basis, hamiltonian, dipole = build_vibladder_operators_from_parameters(
            model_params,
            execution_policy=execution,
        )
    elif isinstance(model_params, TwoLevelParameters):
        basis, hamiltonian, dipole = build_twolevel_operators_from_parameters(
            model_params,
            execution_policy=execution,
        )
    elif isinstance(model_params, SymmetricTopParameters):  # pragma: no cover
        raise AssertionError("SymTop must be rejected before construction")
    else:  # pragma: no cover
        raise AssertionError("unsupported validated optimization model")

    # The optimizer historically stores H0 in rad/fs. Retain that representation
    # while sharing the production construction and physical unit boundary.
    hamiltonian = hamiltonian.to_frequency_units()
    return OptimizationModel(name, basis, hamiltonian, dipole)


def validate_optimization_state(
    basis: Any,
    state: Sequence[Any],
    *,
    label: str,
) -> tuple[int, ...]:
    """Validate an exact quantum-number tuple without adding or dropping M."""
    if isinstance(state, (str, bytes)) or not isinstance(state, Sequence):
        raise OptimizationModelConfigurationError(
            f"states.{label} must be a sequence of integer quantum numbers"
        )
    values = tuple(state)
    if not values or any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
        for value in values
    ):
        raise OptimizationModelConfigurationError(
            f"states.{label} must contain only integer quantum numbers"
        )
    normalized = tuple(int(value) for value in values)
    try:
        basis.get_index(normalized)
    except (TypeError, ValueError) as exc:
        raise OptimizationModelConfigurationError(
            f"states.{label}={normalized} is not present in the selected basis"
        ) from exc
    return normalized


__all__ = [
    "OptimizationModel",
    "OptimizationModelConfigurationError",
    "build_optimization_model",
    "validate_optimization_state",
]
