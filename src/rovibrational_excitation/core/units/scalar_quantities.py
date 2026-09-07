"""Frozen scalar quantities converted once to internal canonical units."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from numbers import Real

import numpy as np

from .converters import converter


def _finite_scalar(value: float, *, quantity: str) -> float:
    if isinstance(value, (bool, np.bool_)) or np.asarray(value).ndim != 0:
        raise TypeError(f"{quantity} value must be a finite scalar")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{quantity} value must be a finite scalar") from exc
    if not np.isfinite(result):
        raise ValueError(f"{quantity} value must be finite")
    return result


def _canonical_value(
    value: float,
    unit: str,
    *,
    quantity: str,
    canonical_unit: str,
    convert: Callable[[float, str, str], float | np.ndarray],
) -> tuple[float, float]:
    scalar = _finite_scalar(value, quantity=quantity)
    if not isinstance(unit, str):
        raise TypeError(f"{quantity} unit must be a string")
    try:
        canonical = float(convert(scalar, unit, canonical_unit))
    except ValueError as exc:
        raise ValueError(f"invalid {quantity} unit {unit!r}: {exc}") from exc
    if not np.isfinite(canonical):
        raise ValueError(f"converted {quantity} must be finite")
    return scalar, canonical


@dataclass(frozen=True, slots=True)
class DipoleMoment:
    """Dipole input with canonical internal value in C*m."""

    value: float
    unit: str
    coulomb_meters: float = field(init=False)

    def __post_init__(self) -> None:
        value, canonical = _canonical_value(
            self.value,
            self.unit,
            quantity="dipole moment",
            canonical_unit="C*m",
            convert=converter.convert_dipole_moment,
        )
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "coulomb_meters", canonical)


@dataclass(frozen=True, slots=True)
class ElectricFieldAmplitude:
    """Field or cycle-averaged-intensity input with peak value in V/m."""

    value: float
    unit: str
    volts_per_meter: float = field(init=False)

    def __post_init__(self) -> None:
        value, canonical = _canonical_value(
            self.value,
            self.unit,
            quantity="electric-field amplitude",
            canonical_unit="V/m",
            convert=converter.convert_electric_field,
        )
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "volts_per_meter", canonical)


@dataclass(frozen=True, slots=True)
class LocalControlGain:
    """Local-control gain with canonical value in (V/m)^2 fs."""

    value: float
    unit: str
    volts_per_meter_squared_femtoseconds: float = field(init=False)

    def __post_init__(self) -> None:
        if isinstance(self.value, (bool, np.bool_)) or not isinstance(self.value, Real):
            raise TypeError("local control gain value must be a finite scalar")
        value, canonical = _canonical_value(
            self.value,
            self.unit,
            quantity="local control gain",
            canonical_unit="(V/m)^2 fs",
            convert=converter.convert_local_control_gain,
        )
        if canonical <= 0.0:
            raise ValueError("local control gain must be positive")
        object.__setattr__(self, "value", value)
        object.__setattr__(
            self,
            "volts_per_meter_squared_femtoseconds",
            canonical,
        )


@dataclass(frozen=True, slots=True)
class GroupDelayDispersion:
    """GDD input with canonical internal value in fs^2."""

    value: float
    unit: str
    femtoseconds_squared: float = field(init=False)

    def __post_init__(self) -> None:
        value, canonical = _canonical_value(
            self.value,
            self.unit,
            quantity="GDD",
            canonical_unit="fs^2",
            convert=converter.convert_gdd,
        )
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "femtoseconds_squared", canonical)


@dataclass(frozen=True, slots=True)
class ThirdOrderDispersion:
    """TOD input with canonical internal value in fs^3."""

    value: float
    unit: str
    femtoseconds_cubed: float = field(init=False)

    def __post_init__(self) -> None:
        value, canonical = _canonical_value(
            self.value,
            self.unit,
            quantity="TOD",
            canonical_unit="fs^3",
            convert=converter.convert_tod,
        )
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "femtoseconds_cubed", canonical)


__all__ = [
    "DipoleMoment",
    "ElectricFieldAmplitude",
    "LocalControlGain",
    "GroupDelayDispersion",
    "ThirdOrderDispersion",
]
