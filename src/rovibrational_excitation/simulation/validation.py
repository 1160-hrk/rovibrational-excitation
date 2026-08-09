"""Strict validation for one simulation case."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from rovibrational_excitation.core.propagation.utils import validate_axes

from .serialization import deserialize_polarization
from .timegrid import build_time_grid


class SimulationConfigurationError(ValueError):
    """Raised before propagation when a simulation case is invalid."""


_COMMON_REQUIRED = {
    "t_start",
    "t_end",
    "dt",
    "carrier_freq",
    "amplitude",
    "polarization",
}
_MODEL_REQUIRED = {
    "linmol": {"V_max", "J_max", "omega_rad_phz", "mu0_Cm"},
    "twolevel": {"energy_gap", "mu0_Cm"},
    "vibladder": {"V_max", "omega_rad_phz", "mu0_Cm"},
}
_FINITE_PARAMETERS = {
    "t_start",
    "t_end",
    "dt",
    "carrier_freq",
    "amplitude",
    "duration",
    "pulse_duration",
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


def validate_simulation_case(params: Mapping[str, Any]) -> None:
    """Validate one fully-expanded case without changing its values."""
    removed_options = {
        key for key in ("auto_timestep", "target_accuracy") if key in params
    }
    if removed_options:
        names = ", ".join(sorted(removed_options))
        raise SimulationConfigurationError(
            f"{names} were removed; define dt explicitly and validate convergence"
        )

    basis_type_raw = params.get("basis_type", "linmol")
    if not isinstance(basis_type_raw, str):
        raise SimulationConfigurationError("basis_type must be a string")
    basis_type = basis_type_raw.lower()
    if basis_type not in _MODEL_REQUIRED:
        raise SimulationConfigurationError(f"Unknown basis_type: {basis_type}")

    missing = sorted((_COMMON_REQUIRED | _MODEL_REQUIRED[basis_type]) - params.keys())
    if missing:
        raise SimulationConfigurationError(
            "Missing required simulation parameters: " + ", ".join(missing)
        )

    for key in _FINITE_PARAMETERS:
        _require_finite_scalar(params, key)

    try:
        build_time_grid(params["t_start"], params["t_end"], params["dt"])
    except (TypeError, ValueError) as exc:
        raise SimulationConfigurationError(str(exc)) from exc

    duration = params.get("duration", params.get("pulse_duration"))
    if duration is not None and duration <= 0:
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

    sparse = params.get("sparse", not params.get("dense", True))
    if params.get("backend", "numpy") == "cupy" and sparse:
        raise SimulationConfigurationError(
            "sparse=True is not supported by the CuPy propagator"
        )

    # Only LinMol has a physical Cartesian polarization mapping.
    if basis_type == "linmol":
        try:
            validate_axes(params.get("axes", "xy"))
        except (AttributeError, ValueError) as exc:
            raise SimulationConfigurationError(str(exc)) from exc
        if not params.get("use_M", True):
            if "axes" in params:
                raise SimulationConfigurationError(
                    "axes is not applicable when use_M=False; fixed linear "
                    "polarization is aligned with the internal z axis"
                )
            from .models.linmol_m_average import (
                canonicalize_fixed_linear_polarization,
                validate_m_average_initial_states,
            )

            try:
                canonicalize_fixed_linear_polarization(polarization)
                validate_m_average_initial_states(dict(params))
            except ValueError as exc:
                raise SimulationConfigurationError(str(exc)) from exc

    if params.get("algorithm", "rk4") not in {"rk4", "split_operator"}:
        raise SimulationConfigurationError(
            "algorithm must be 'rk4' or 'split_operator'"
        )
    if params.get("split_interaction", "cartesian") not in {
        "cartesian",
        "helicity_projected",
    }:
        raise SimulationConfigurationError(
            "split_interaction must be 'cartesian' or 'helicity_projected'"
        )
    if params.get("backend", "numpy") not in {"numpy", "cupy"}:
        raise SimulationConfigurationError("backend must be 'numpy' or 'cupy'")
