"""Frozen model-parameter schemas with canonical frequency values."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from rovibrational_excitation.core.units import DipoleMoment, Frequency
from rovibrational_excitation.core.units.converters import converter

from .symmetry import MoleculeSymmetryPreset, resolve_molecule_preset


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


def _potential_type(params: Mapping[str, Any]) -> Literal["harmonic", "morse"]:
    value = params["potential_type"]
    if value == "harmonic":
        return "harmonic"
    if value == "morse":
        return "morse"
    raise ValueError("potential_type must be 'harmonic' or 'morse'")


def _frequency(params: Mapping[str, Any], key: str) -> Frequency:
    unit_key = f"{key}_units"
    try:
        return Frequency(params[key], params[unit_key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {key}/{unit_key}: {exc}") from exc


def _dipole_scale(params: Mapping[str, Any]) -> float:
    try:
        return DipoleMoment(
            params["dipole_scale"], params["dipole_scale_units"]
        ).coulomb_meters
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid dipole_scale/dipole_scale_units: {exc}") from exc


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
    potential_type: Literal["harmonic", "morse"]

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
            dipole_c_m=_dipole_scale(params),
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
    potential_type: Literal["harmonic", "morse"]

    @classmethod
    def from_mapping(cls, params: Mapping[str, Any]) -> VibLadderParameters:
        result = cls(
            v_max=_nonnegative_integer(params, "V_max"),
            vibrational_frequency=_frequency(params, "vibrational_frequency"),
            anharmonic_shift=_frequency(params, "anharmonic_shift"),
            dipole_c_m=_dipole_scale(params),
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
class SymmetricTopParameters:
    """Validated rigid parallel-band symmetric-top parameters."""

    v_max: int
    j_max: int
    molecule_preset: MoleculeSymmetryPreset
    nuclear_spin_isomer: str
    vibrational_frequency: Frequency
    anharmonic_shift: Frequency
    rotational_constant_perpendicular: Frequency
    rotational_constant_parallel: Frequency
    vibration_rotation_coupling_perpendicular: Frequency
    vibration_rotation_coupling_parallel: Frequency
    dipole_c_m: float
    potential_type: Literal["harmonic", "morse"]

    @classmethod
    def from_mapping(cls, params: Mapping[str, Any]) -> SymmetricTopParameters:
        molecule = params["molecule"]
        if not isinstance(molecule, str):
            raise TypeError("molecule must be a string")
        preset = resolve_molecule_preset(molecule)
        if preset.model_family != "symmetric_top":
            raise ValueError(
                "SymTop requires a symmetric_top molecule preset; "
                f"{preset.canonical_id} is {preset.model_family}"
            )
        nuclear_spin_isomer = params["nuclear_spin_isomer"]
        if not isinstance(nuclear_spin_isomer, str):
            raise TypeError("nuclear_spin_isomer must be a string")
        if nuclear_spin_isomer not in {"ortho", "para"}:
            raise ValueError(
                "nuclear_spin_isomer must be ortho or para for the CH3F "
                "pure-state SymTop model"
            )

        result = cls(
            v_max=_nonnegative_integer(params, "V_max"),
            j_max=_nonnegative_integer(params, "J_max"),
            molecule_preset=preset,
            nuclear_spin_isomer=nuclear_spin_isomer,
            vibrational_frequency=_frequency(params, "vibrational_frequency"),
            anharmonic_shift=_frequency(params, "anharmonic_shift"),
            rotational_constant_perpendicular=_frequency(
                params,
                "rotational_constant_perpendicular",
            ),
            rotational_constant_parallel=_frequency(
                params,
                "rotational_constant_parallel",
            ),
            vibration_rotation_coupling_perpendicular=_frequency(
                params,
                "vibration_rotation_coupling_perpendicular",
            ),
            vibration_rotation_coupling_parallel=_frequency(
                params,
                "vibration_rotation_coupling_parallel",
            ),
            dipole_c_m=_dipole_scale(params),
            potential_type=_potential_type(params),
        )
        if result.vibrational_frequency.angular_rad_per_fs <= 0.0:
            raise ValueError("vibrational_frequency must be positive")
        if result.anharmonic_shift.angular_rad_per_fs < 0.0:
            raise ValueError("anharmonic_shift must be non-negative")
        if result.rotational_constant_perpendicular.angular_rad_per_fs <= 0.0:
            raise ValueError("rotational_constant_perpendicular must be positive")
        if result.rotational_constant_parallel.angular_rad_per_fs <= 0.0:
            raise ValueError("rotational_constant_parallel must be positive")
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
            dipole_c_m=_dipole_scale(params),
        )
