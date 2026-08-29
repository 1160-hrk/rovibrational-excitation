"""Typed scalar time input with one canonical femtosecond value."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .converters import converter


@dataclass(frozen=True, slots=True)
class TimeQuantity:
    """Finite scalar time-like quantity with an explicit input unit."""

    value: float
    unit: str
    femtoseconds: float = field(init=False)

    def __post_init__(self) -> None:
        if isinstance(self.value, (bool, np.bool_)):
            raise TypeError("time value must be a finite scalar")
        if np.asarray(self.value).ndim != 0:
            raise TypeError("time value must be a finite scalar")
        try:
            value = float(self.value)
        except (TypeError, ValueError) as exc:
            raise TypeError("time value must be a finite scalar") from exc
        if not np.isfinite(value):
            raise ValueError("time value must be finite")
        if not isinstance(self.unit, str):
            raise TypeError("time unit must be a string")
        try:
            femtoseconds = float(converter.convert_time(value, self.unit, "fs"))
        except ValueError as exc:
            raise ValueError(f"invalid time unit {self.unit!r}: {exc}") from exc
        if not np.isfinite(femtoseconds):
            raise ValueError("converted time must be finite")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "femtoseconds", femtoseconds)


__all__ = ["TimeQuantity"]
