"""Validated physical parameters for the two-level model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .._parameter_validation import (
    dipole_scale,
    energy_or_frequency_unit,
    finite_scalar,
)


@dataclass(frozen=True, slots=True)
class TwoLevelParameters:
    """Validated two-level parameters without changing energy-gap units."""

    energy_gap: float
    energy_gap_units: str
    dipole_c_m: float

    @classmethod
    def from_mapping(cls, params: Mapping[str, Any]) -> TwoLevelParameters:
        return cls(
            energy_gap=finite_scalar(params, "energy_gap"),
            energy_gap_units=energy_or_frequency_unit(params, "energy_gap_units"),
            dipole_c_m=dipole_scale(params),
        )
