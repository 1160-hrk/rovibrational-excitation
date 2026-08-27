"""Frozen model-parameter schemas with canonical frequency values."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from rovibrational_excitation.core.units import Frequency
from rovibrational_excitation.core.units.converters import converter


def _nonnegative_integer(params: Mapping[str, Any], key: str) -> int:
    value = params[key]
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{key} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return result


def _finite_scalar(params: Mapping[str, Any], key: str) -> float:
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


def _unit(params: Mapping[str, Any], key: str) -> str:
    value = params[key]
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string")
    return value


def _energy_or_frequency_unit(params: Mapping[str, Any], key: str) -> str:
    value = _unit(params, key)
    supported = set(converter.get_supported_units("frequency")) | set(
        converter.get_supported_units("energy")
    )
    if value not in supported:
        raise ValueError(f"unsupported {key}: {value}")
    return value


def _potential_type(params: Mapping[str, Any]) -> str:
    value = params["potential_type"]
    if value not in {"harmonic", "morse"}:
        raise ValueError("potential_type must be 'harmonic' or 'morse'")
    return str(value)


def _frequency(params: Mapping[str, Any], key: str) -> Frequency:
    unit_key = f"{key}_units"
    try:
        return Frequency(params[key], params[unit_key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {key}/{unit_key}: {exc}") from exc


@dataclass(frozen=True, slots=True)
class LinMolParameters:
    """Validated linear-molecule parameters before basis allocation."""

    v_max: int
    j_max: int
    vibrational_frequency: Frequency
    anharmonic_shift: Frequency
    rotational_constant: Frequency
    vibration_rotation_coupling: Frequency
    dipole_c_m: float
    potential_type: str

    @classmethod
    def from_mapping(cls, params: Mapping[str, Any]) -> LinMolParameters:
        result = cls(
            v_max=_nonnegative_integer(params, "V_max"),
            j_max=_nonnegative_integer(params, "J_max"),
            vibrational_frequency=_frequency(params, "vibrational_frequency"),
            anharmonic_shift=_frequency(params, "anharmonic_shift"),
            rotational_constant=_frequency(params, "rotational_constant"),
            vibration_rotation_coupling=_frequency(
                params,
                "vibration_rotation_coupling",
            ),
            dipole_c_m=_finite_scalar(params, "mu0_Cm"),
            potential_type=_potential_type(params),
        )
        if (
            result.potential_type == "morse"
            and result.anharmonic_shift.angular_rad_per_fs == 0.0
        ):
            raise ValueError(
                "anharmonic_shift must be non-zero when potential_type='morse'"
            )
        return result


@dataclass(frozen=True, slots=True)
class VibLadderParameters:
    """Validated vibrational-ladder parameters before basis allocation."""

    v_max: int
    vibrational_frequency: Frequency
    anharmonic_shift: Frequency
    dipole_c_m: float
    potential_type: str

    @classmethod
    def from_mapping(cls, params: Mapping[str, Any]) -> VibLadderParameters:
        result = cls(
            v_max=_nonnegative_integer(params, "V_max"),
            vibrational_frequency=_frequency(params, "vibrational_frequency"),
            anharmonic_shift=_frequency(params, "anharmonic_shift"),
            dipole_c_m=_finite_scalar(params, "mu0_Cm"),
            potential_type=_potential_type(params),
        )
        if (
            result.potential_type == "morse"
            and result.anharmonic_shift.angular_rad_per_fs == 0.0
        ):
            raise ValueError(
                "anharmonic_shift must be non-zero when potential_type='morse'"
            )
        return result


@dataclass(frozen=True, slots=True)
class TwoLevelParameters:
    """Validated two-level parameters without changing energy-gap units."""

    energy_gap: float
    energy_gap_units: str
    dipole_c_m: float

    @classmethod
    def from_mapping(cls, params: Mapping[str, Any]) -> TwoLevelParameters:
        return cls(
            energy_gap=_finite_scalar(params, "energy_gap"),
            energy_gap_units=_energy_or_frequency_unit(params, "energy_gap_units"),
            dipole_c_m=_finite_scalar(params, "mu0_Cm"),
        )
