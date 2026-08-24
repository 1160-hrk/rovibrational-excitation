"""Strict validation for one simulation case."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from rovibrational_excitation.core.execution import ExecutionPolicy
from rovibrational_excitation.core.time import TimeGrid
from rovibrational_excitation.dynamics.capabilities import (
    PropagationAlgorithm,
    StatePath,
    validate_execution_capability,
)
from rovibrational_excitation.dynamics.options import (
    PropagationOptions,
    RenormalizationPolicy,
    ScalingMode,
)
from rovibrational_excitation.dynamics.utils import validate_axes
from rovibrational_excitation.io import deserialize_polarization
from rovibrational_excitation.models.validation import (
    LinMolRepresentation,
    ModelConfigurationError,
    validate_linmol_representation,
    validate_model_parameters,
)


class SimulationConfigurationError(ValueError):
    """Raised before propagation when a simulation case is invalid."""


_COMMON_REQUIRED = {
    "t_start",
    "t_end",
    "dt",
    "carrier_freq",
    "amplitude",
    "polarization",
    "duration",
    "backend",
    "storage",
    "algorithm",
    "return_traj",
    "sample_stride",
    "nondimensional",
    "renorm",
}
_FINITE_PARAMETERS = {
    "t_start",
    "t_end",
    "dt",
    "carrier_freq",
    "amplitude",
    "duration",
    "t_center",
    "gdd",
    "tod",
    "omega_rad_phz",
    "delta_omega_rad_phz",
    "B_rad_phz",
    "alpha_rad_phz",
    "mu0_Cm",
    "energy_gap",
    "amplitude_sin_mod",
    "carrier_freq_sin_mod",
    "phase_rad_sin_mod",
}


def _require_finite_scalar(params: Mapping[str, Any], key: str) -> None:
    if key not in params:
        return
    value = params[key]
    if isinstance(value, (bool, np.bool_)):
        raise SimulationConfigurationError(f"{key} must be a finite number")
    try:
        finite = np.asarray(value).ndim == 0 and np.isfinite(float(value))
    except (TypeError, ValueError):
        finite = False
    if not finite:
        raise SimulationConfigurationError(f"{key} must be a finite number")


def validate_simulation_case(
    params: Mapping[str, Any],
) -> PropagationOptions:
    """Validate one fully-expanded case without changing its values."""
    removed_options = {
        key for key in ("auto_timestep", "target_accuracy") if key in params
    }
    if removed_options:
        names = ", ".join(sorted(removed_options))
        raise SimulationConfigurationError(
            f"{names} were removed; define dt explicitly and validate convergence"
        )
    if "pulse_duration" in params:
        raise SimulationConfigurationError(
            "pulse_duration was removed; use the required parameter duration"
        )
    legacy_storage = {key for key in ("dense", "sparse") if key in params}
    if legacy_storage:
        names = ", ".join(sorted(legacy_storage))
        raise SimulationConfigurationError(
            f"{names} were removed; use required storage='dense' or storage='csr'"
        )

    try:
        basis_type = validate_model_parameters(params)
    except ModelConfigurationError as exc:
        raise SimulationConfigurationError(str(exc)) from exc
    missing = sorted(_COMMON_REQUIRED - params.keys())
    if missing:
        raise SimulationConfigurationError(
            "Missing required simulation parameters: " + ", ".join(missing)
        )

    for key in _FINITE_PARAMETERS:
        _require_finite_scalar(params, key)

    try:
        TimeGrid.from_bounds(params["t_start"], params["t_end"], params["dt"])
    except (TypeError, ValueError) as exc:
        raise SimulationConfigurationError(str(exc)) from exc

    duration = params["duration"]
    if duration <= 0:
        raise SimulationConfigurationError("duration must be positive")

    for key in ("V_max", "J_max"):
        if key in params and (
            isinstance(params[key], (bool, np.bool_))
            or not isinstance(params[key], (int, np.integer))
            or params[key] < 0
        ):
            raise SimulationConfigurationError(f"{key} must be a non-negative integer")

    try:
        polarization = deserialize_polarization(params["polarization"])
    except (TypeError, ValueError) as exc:
        raise SimulationConfigurationError(f"Invalid polarization: {exc}") from exc
    norm = np.linalg.norm(polarization)
    if not np.all(np.isfinite(polarization)) or not np.isfinite(norm) or norm == 0:
        raise SimulationConfigurationError("polarization must be finite and non-zero")

    try:
        execution_policy = ExecutionPolicy.from_strings(
            backend=params["backend"],
            storage=params["storage"],
        )
        algorithm = PropagationAlgorithm(params["algorithm"])
        if not isinstance(params["nondimensional"], bool):
            raise TypeError("nondimensional must be a bool")
        if not isinstance(params["renorm"], bool):
            raise TypeError("renorm must be a bool")
        options = PropagationOptions(
            algorithm=algorithm,
            execution=execution_policy,
            return_trajectory=params["return_traj"],
            sample_stride=params["sample_stride"],
            scaling=(
                ScalingMode.NONDIMENSIONAL
                if params["nondimensional"]
                else ScalingMode.DIMENSIONAL
            ),
            renormalization=(
                RenormalizationPolicy.PER_STEP
                if params["renorm"]
                else RenormalizationPolicy.DISABLED
            ),
        )
    except (TypeError, ValueError) as exc:
        raise SimulationConfigurationError(str(exc)) from exc

    representation = (
        validate_linmol_representation(params) if basis_type == "linmol" else None
    )
    state_path = (
        StatePath.INCOHERENT_ENSEMBLE
        if representation is LinMolRepresentation.M_INCOHERENT_AVERAGE
        else StatePath.PURE
    )
    try:
        validate_execution_capability(
            state_path=state_path,
            algorithm=options.algorithm,
            policy=options.execution,
        )
    except (RuntimeError, ValueError) as exc:
        raise SimulationConfigurationError(str(exc)) from exc

    # Only m_resolved LinMol has a Cartesian laboratory-axis mapping.
    if representation is LinMolRepresentation.M_RESOLVED:
        if "axes" not in params:
            raise SimulationConfigurationError(
                "Missing required LinMol m_resolved parameter: axes"
            )
        try:
            validate_axes(params["axes"])
        except (AttributeError, ValueError) as exc:
            raise SimulationConfigurationError(str(exc)) from exc
    elif representation is LinMolRepresentation.M_INCOHERENT_AVERAGE:
        if "axes" in params:
            raise SimulationConfigurationError(
                "axes is not applicable when representation="
                "m_incoherent_average; fixed linear polarization is aligned "
                "with the internal z axis"
            )
        from .m_average import (
            canonicalize_fixed_linear_polarization,
            validate_m_average_initial_states,
        )

        try:
            canonicalize_fixed_linear_polarization(polarization)
            validate_m_average_initial_states(dict(params))
        except ValueError as exc:
            raise SimulationConfigurationError(str(exc)) from exc

    if params.get("split_interaction", "cartesian") not in {
        "cartesian",
        "helicity_projected",
    }:
        raise SimulationConfigurationError(
            "split_interaction must be 'cartesian' or 'helicity_projected'"
        )
    return options
