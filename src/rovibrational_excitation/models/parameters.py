"""Frozen model-parameter schemas with canonical frequency values."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from rovibrational_excitation.core.units import Frequency

from ._parameter_validation import (
    dipole_scale,
    frequency,
    nonnegative_integer,
    potential_type,
)
from .symmetry import MoleculeSymmetryPreset, resolve_molecule_preset


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
            v_max=nonnegative_integer(params, "V_max"),
            j_max=nonnegative_integer(params, "J_max"),
            vibrational_frequency=frequency(params, "vibrational_frequency"),
            anharmonic_shift=frequency(params, "anharmonic_shift"),
            rotational_constant=frequency(params, "rotational_constant"),
            vibration_rotation_coupling=frequency(
                params,
                "vibration_rotation_coupling",
            ),
            dipole_c_m=dipole_scale(params),
            potential_type=potential_type(params),
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
            v_max=nonnegative_integer(params, "V_max"),
            j_max=nonnegative_integer(params, "J_max"),
            molecule_preset=preset,
            nuclear_spin_isomer=nuclear_spin_isomer,
            vibrational_frequency=frequency(params, "vibrational_frequency"),
            anharmonic_shift=frequency(params, "anharmonic_shift"),
            rotational_constant_perpendicular=frequency(
                params,
                "rotational_constant_perpendicular",
            ),
            rotational_constant_parallel=frequency(
                params,
                "rotational_constant_parallel",
            ),
            vibration_rotation_coupling_perpendicular=frequency(
                params,
                "vibration_rotation_coupling_perpendicular",
            ),
            vibration_rotation_coupling_parallel=frequency(
                params,
                "vibration_rotation_coupling_parallel",
            ),
            dipole_c_m=dipole_scale(params),
            potential_type=potential_type(params),
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
