"""Frozen linear-molecule parameters with canonical frequency values."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from rovibrational_excitation.core.units import Frequency

from .._parameter_validation import (
    dipole_scale,
    frequency,
    nonnegative_integer,
    potential_type,
)


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
