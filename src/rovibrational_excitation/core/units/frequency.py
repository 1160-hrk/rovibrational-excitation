"""Typed frequency input with one canonical angular-frequency value."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .converters import converter


@dataclass(frozen=True, slots=True)
class Frequency:
    """Finite scalar frequency with an explicit input unit.

    User-facing values may be ordinary frequency, wavenumber, or angular
    frequency. Numerical consumers use :attr:`angular_rad_per_fs` only.
    """

    value: float
    unit: str
    angular_rad_per_fs: float = field(init=False)

    def __post_init__(self) -> None:
        if isinstance(self.value, (bool, np.bool_)):
            raise TypeError("frequency value must be a finite scalar")
        if np.asarray(self.value).ndim != 0:
            raise TypeError("frequency value must be a finite scalar")
        try:
            value = float(self.value)
        except (TypeError, ValueError) as exc:
            raise TypeError("frequency value must be a finite scalar") from exc
        if not np.isfinite(value):
            raise ValueError("frequency value must be finite")
        if not isinstance(self.unit, str):
            raise TypeError("frequency unit must be a string")
        try:
            angular = float(converter.convert_frequency(value, self.unit, "rad/fs"))
        except ValueError as exc:
            raise ValueError(f"invalid frequency unit {self.unit!r}: {exc}") from exc
        if not np.isfinite(angular):
            raise ValueError("converted frequency must be finite")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "angular_rad_per_fs", angular)

    @property
    def cycles_per_fs(self) -> float:
        """Return ordinary frequency in cycles/fs (equivalent to PHz)."""
        return float(
            converter.convert_frequency(
                self.angular_rad_per_fs,
                "rad/fs",
                "PHz",
            )
        )
