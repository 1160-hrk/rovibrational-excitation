"""Strict validation for one simulation case."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from rovibrational_excitation.core.execution import ExecutionPolicy
from rovibrational_excitation.core.time import TimeGrid
from rovibrational_excitation.core.units import Frequency, TimeQuantity
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
from rovibrational_excitation.fields import CartesianField, ScalarField
from rovibrational_excitation.io import deserialize_polarization
from rovibrational_excitation.models.validation import (
    LinMolRepresentation,
    ModelConfigurationError,
    known_model_input_keys,
    model_parameter_keys,
    validate_linmol_representation,
    validate_model_parameters,
)


class SimulationConfigurationError(ValueError):
    """Raised before propagation when a simulation case is invalid."""


_EXECUTION_REQUIRED = {
    "backend",
    "storage",
    "algorithm",
    "return_traj",
    "sample_stride",
    "nondimensional",
    "renorm",
}
_SELECTION_KEYS = {"basis_type", "initial_states", "representation", "axes"}
_WORKFLOW_KEYS = {
    "description",
    "save",
    "outdir",
    "validate_units",
    "verbose",
    "split_interaction",
}
_PREPROCESSED_UNIT_KEYS = {
    "mu0_Cm_units",
    "amplitude_units",
    "duration_units",
    "t_center_units",
    "t_start_units",
    "t_end_units",
    "dt_units",
}
_REMOVED_SIMULATION_KEYS = {
    "carrier_freq",
    "auto_timestep",
    "target_accuracy",
    "pulse_duration",
    "dense",
    "sparse",
    "use_M",
    "amplitude_sin_mod",
    "carrier_freq_sin_mod",
    "phase_rad_sin_mod",
    "type_mod_sin_mod",
}
_GENERATED_REQUIRED = {
    "t_start",
    "t_end",
    "dt",
    "carrier_frequency",
    "carrier_frequency_units",
    "amplitude",
    "duration",
    "t_center",
    "envelope_kind",
    "modulation_kind",
}
_GENERATED_FIELD_KEYS = {
    *_GENERATED_REQUIRED,
    "polarization",
    "envelope_func",
    "phase_rad",
    "gdd",
    "tod",
    "modulation_depth",
    "modulation_delay",
    "modulation_delay_units",
    "modulation_phase_rad",
    "modulation_mode",
    "amplitude_units",
    "duration_units",
    "t_center_units",
    "t_start_units",
    "t_end_units",
    "dt_units",
}
_FINITE_PARAMETERS = {
    "t_start",
    "t_end",
    "dt",
    "carrier_frequency",
    "amplitude",
    "duration",
    "t_center",
    "gdd",
    "tod",
    "mu0_Cm",
    "energy_gap",
    "modulation_depth",
    "modulation_delay",
    "phase_rad",
    "modulation_phase_rad",
}

_KNOWN_SIMULATION_KEYS = frozenset(
    _SELECTION_KEYS
    | _EXECUTION_REQUIRED
    | _GENERATED_FIELD_KEYS
    | _WORKFLOW_KEYS
    | _PREPROCESSED_UNIT_KEYS
    | _REMOVED_SIMULATION_KEYS
    | set(known_model_input_keys())
)


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


_SINUSOIDAL_MODULATION_KEYS = frozenset(
    {
        "modulation_depth",
        "modulation_delay",
        "modulation_delay_units",
        "modulation_phase_rad",
        "modulation_mode",
    }
)
_SINUSOIDAL_MODULATION_REQUIRED = frozenset(
    {
        "modulation_depth",
        "modulation_delay",
        "modulation_delay_units",
        "modulation_mode",
    }
)


def _validate_generated_field_schema(params: Mapping[str, Any]) -> None:
    from rovibrational_excitation.fields.envelopes import get_generated_envelope

    try:
        get_generated_envelope(params["envelope_kind"])
    except ValueError as exc:
        raise SimulationConfigurationError(str(exc)) from exc

    try:
        Frequency(
            params["carrier_frequency"],
            params["carrier_frequency_units"],
        )
    except (TypeError, ValueError) as exc:
        raise SimulationConfigurationError(
            "invalid carrier_frequency/carrier_frequency_units: " + str(exc)
        ) from exc

    modulation_kind = params["modulation_kind"]
    if not isinstance(modulation_kind, str) or modulation_kind not in {
        "none",
        "sinusoidal",
    }:
        raise SimulationConfigurationError(
            "modulation_kind must be one of: none, sinusoidal"
        )

    supplied_modulation_keys = _SINUSOIDAL_MODULATION_KEYS & params.keys()
    if modulation_kind == "none":
        if supplied_modulation_keys:
            names = ", ".join(sorted(supplied_modulation_keys))
            raise SimulationConfigurationError(
                f"{names} are not applicable when modulation_kind=none"
            )
        return

    missing = sorted(_SINUSOIDAL_MODULATION_REQUIRED - params.keys())
    if missing:
        raise SimulationConfigurationError(
            "Missing required sinusoidal modulation parameters: " + ", ".join(missing)
        )
    try:
        TimeQuantity(
            params["modulation_delay"],
            params["modulation_delay_units"],
        )
    except (TypeError, ValueError) as exc:
        raise SimulationConfigurationError(
            "invalid modulation_delay/modulation_delay_units: " + str(exc)
        ) from exc

    modulation_type = params["modulation_mode"]
    if not isinstance(modulation_type, str) or modulation_type not in {
        "phase",
        "amplitude",
    }:
        raise SimulationConfigurationError(
            "modulation_mode must be one of: phase, amplitude"
        )
    if modulation_type == "amplitude" and not 0.0 <= params["modulation_depth"] <= 1.0:
        raise SimulationConfigurationError(
            "amplitude modulation_depth must satisfy 0 <= depth <= 1"
        )


def validate_simulation_case(
    params: Mapping[str, Any],
    *,
    field: ScalarField | CartesianField | None = None,
) -> PropagationOptions:
    """Validate one generated or externally sampled simulation case."""
    removed_field_selectors = {
        key
        for key in ("envelope_func", "Sinusoidal_modulation", "carrier_freq")
        if key in params
    }
    if removed_field_selectors:
        names = ", ".join(sorted(removed_field_selectors))
        raise SimulationConfigurationError(
            f"{names} were removed; use envelope_kind, modulation_kind, and "
            "carrier_frequency with carrier_frequency_units, or inject an "
            "external sampled field for a custom waveform"
        )
    legacy_modulation = {
        "amplitude_sin_mod",
        "carrier_freq_sin_mod",
        "phase_rad_sin_mod",
        "type_mod_sin_mod",
    } & params.keys()
    if legacy_modulation:
        names = ", ".join(sorted(legacy_modulation))
        raise SimulationConfigurationError(
            f"{names} were removed; use modulation_depth, modulation_delay with "
            "modulation_delay_units, modulation_phase_rad, and modulation_mode"
        )
    removed_options = {
        key for key in ("auto_timestep", "target_accuracy") if key in params
    }
    if removed_options:
        names = ", ".join(sorted(removed_options))
        raise SimulationConfigurationError(
            f"{names} were removed; provide an exact TimeGrid and validate "
            "convergence explicitly"
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

    unknown = [key for key in params if key not in _KNOWN_SIMULATION_KEYS]
    if unknown:
        names = ", ".join(
            sorted(key if isinstance(key, str) else repr(key) for key in unknown)
        )
        raise SimulationConfigurationError("Unknown simulation parameters: " + names)

    try:
        basis_type = validate_model_parameters(params)
    except ModelConfigurationError as exc:
        raise SimulationConfigurationError(str(exc)) from exc

    inapplicable_model = sorted(
        (known_model_input_keys() - model_parameter_keys(basis_type)) & params.keys()
    )
    if inapplicable_model:
        raise SimulationConfigurationError(
            f"Model parameters not applicable to basis_type={basis_type}: "
            + ", ".join(inapplicable_model)
        )
    if basis_type != "linmol":
        inapplicable_selection = sorted({"representation", "axes"} & params.keys())
        if inapplicable_selection:
            raise SimulationConfigurationError(
                f"Parameters not applicable to basis_type={basis_type}: "
                + ", ".join(inapplicable_selection)
            )
    representation = (
        validate_linmol_representation(params) if basis_type == "linmol" else None
    )
    expects_cartesian = representation is LinMolRepresentation.M_RESOLVED

    required = set(_EXECUTION_REQUIRED)
    if field is None:
        required.update(_GENERATED_REQUIRED)
        if expects_cartesian:
            required.add("polarization")
    elif not isinstance(field, (ScalarField, CartesianField)):
        raise SimulationConfigurationError(
            "external field must be a ScalarField or CartesianField"
        )
    else:
        inapplicable = sorted(_GENERATED_FIELD_KEYS & params.keys())
        if inapplicable:
            raise SimulationConfigurationError(
                "generated field parameters are not applicable to external field "
                "injection: " + ", ".join(inapplicable)
            )

    missing = sorted(required - params.keys())
    if missing:
        raise SimulationConfigurationError(
            "Missing required simulation parameters: " + ", ".join(missing)
        )

    for key in ("validate_units", "verbose", "save"):
        if key in params and not isinstance(params[key], bool):
            raise SimulationConfigurationError(f"{key} must be a bool")

    if field is None and not expects_cartesian and basis_type != "linmol":
        if "polarization" in params:
            raise SimulationConfigurationError(
                f"polarization is not applicable to scalar model {basis_type}"
            )

    if field is None:
        _validate_generated_field_schema(params)

    if field is not None:
        if expects_cartesian and not isinstance(field, CartesianField):
            raise SimulationConfigurationError(
                "m_resolved LinMol requires a Cartesian field"
            )
        if not expects_cartesian and not isinstance(field, ScalarField):
            raise SimulationConfigurationError(
                f"{basis_type} scalar coupling requires a scalar field"
            )

    for key in _FINITE_PARAMETERS:
        _require_finite_scalar(params, key)

    polarization: np.ndarray | None = None
    if field is None:
        try:
            TimeGrid.from_bounds(params["t_start"], params["t_end"], params["dt"])
        except (TypeError, ValueError) as exc:
            raise SimulationConfigurationError(str(exc)) from exc

        duration = params["duration"]
        if duration <= 0:
            raise SimulationConfigurationError("duration must be positive")

        if "polarization" in params:
            try:
                polarization = deserialize_polarization(params["polarization"])
            except (TypeError, ValueError) as exc:
                raise SimulationConfigurationError(
                    f"Invalid polarization: {exc}"
                ) from exc
            norm = np.linalg.norm(polarization)
            if (
                not np.all(np.isfinite(polarization))
                or not np.isfinite(norm)
                or norm == 0
            ):
                raise SimulationConfigurationError(
                    "polarization must be finite and non-zero"
                )

    for key in ("V_max", "J_max"):
        if key in params and (
            isinstance(params[key], (bool, np.bool_))
            or not isinstance(params[key], (int, np.integer))
            or params[key] < 0
        ):
            raise SimulationConfigurationError(f"{key} must be a non-negative integer")

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

    split_selector_is_applicable = (
        representation is LinMolRepresentation.M_RESOLVED
        and options.algorithm is PropagationAlgorithm.SPLIT_OPERATOR
    )
    if split_selector_is_applicable:
        if "split_interaction" not in params:
            raise SimulationConfigurationError(
                "split_interaction is required for m_resolved LinMol "
                "split_operator propagation"
            )
        if params["split_interaction"] not in {
            "cartesian",
            "helicity_projected",
        }:
            raise SimulationConfigurationError(
                "split_interaction must be 'cartesian' or 'helicity_projected'"
            )
    elif "split_interaction" in params:
        if options.algorithm is PropagationAlgorithm.RK4:
            reason = "algorithm=rk4 does not use a split interaction"
        elif representation is LinMolRepresentation.M_INCOHERENT_AVERAGE:
            reason = "m_incoherent_average uses its scalar split interaction"
        else:
            reason = f"scalar model {basis_type} has no Cartesian split selector"
        raise SimulationConfigurationError(
            f"split_interaction is not applicable: {reason}"
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
                "m_incoherent_average; scalar field uses the internal z axis"
            )
        from .m_average import (
            canonicalize_fixed_linear_polarization,
            validate_m_average_initial_states,
        )

        try:
            if polarization is not None:
                canonicalize_fixed_linear_polarization(polarization)
            validate_m_average_initial_states(dict(params))
        except ValueError as exc:
            raise SimulationConfigurationError(str(exc)) from exc

    return options
