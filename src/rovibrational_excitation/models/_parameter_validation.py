"""Shared strict validators for frozen model-parameter schemas."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np

from rovibrational_excitation.core.units import DipoleMoment, Frequency
from rovibrational_excitation.core.units.converters import converter


def nonnegative_integer(params: Mapping[str, Any], key: str) -> int:
    value = params[key]
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{key} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return result


def finite_scalar(params: Mapping[str, Any], key: str) -> float:
    value = params[key]
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{key} must be a finite number")
    if np.asarray(value).ndim != 0:
        raise TypeError(f"{key} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{key} must be a finite number") from exc
    if not np.isfinite(result):
        raise ValueError(f"{key} must be a finite number")
    return result


def unit(params: Mapping[str, Any], key: str) -> str:
    value = params[key]
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string")
    return value


def energy_or_frequency_unit(params: Mapping[str, Any], key: str) -> str:
    value = unit(params, key)
    supported = set(converter.get_supported_units("frequency")) | set(
        converter.get_supported_units("energy")
    )
    if value not in supported:
        raise ValueError(f"unsupported {key}: {value}")
    return value


def potential_type(params: Mapping[str, Any]) -> Literal["harmonic", "morse"]:
    value = params["potential_type"]
    if value == "harmonic":
        return "harmonic"
    if value == "morse":
        return "morse"
    raise ValueError("potential_type must be 'harmonic' or 'morse'")


def frequency(params: Mapping[str, Any], key: str) -> Frequency:
    unit_key = f"{key}_units"
    try:
        return Frequency(params[key], params[unit_key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {key}/{unit_key}: {exc}") from exc


def dipole_scale(params: Mapping[str, Any]) -> float:
    try:
        return DipoleMoment(
            params["dipole_scale"], params["dipole_scale_units"]
        ).coulomb_meters
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid dipole_scale/dipole_scale_units: {exc}") from exc
