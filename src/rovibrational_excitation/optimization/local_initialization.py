"""Strict initialization policies for the local-control workflow."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Literal, TypeAlias

import numpy as np

from rovibrational_excitation.core.units import ElectricFieldAmplitude, converter

_SEED_FIELD_KEYS = {"method", "amplitude", "amplitude_units", "max_segments"}


def _names(values: set[Any]) -> str:
    return ", ".join(
        sorted(value if isinstance(value, str) else repr(value) for value in values)
    )


@dataclass(frozen=True, slots=True)
class LocalNoInitialization:
    """Use the local-control law directly, with no injected starter field."""

    method: Literal["none"] = field(default="none", init=False)


@dataclass(frozen=True, slots=True)
class LocalSeedFieldInitialization:
    """A finite starter field converted once to canonical V/m."""

    amplitude: float
    amplitude_units: str
    max_segments: int
    amplitude_v_per_m: float = field(init=False)
    method: Literal["seed_field"] = field(default="seed_field", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.amplitude_units, str) or self.amplitude_units not in (
            converter.get_supported_units("field_amplitude")
        ):
            raise ValueError(
                "initialization.amplitude_units must be a supported direct "
                "electric-field amplitude unit"
            )
        try:
            amplitude = ElectricFieldAmplitude(
                self.amplitude,
                self.amplitude_units,
            ).volts_per_meter
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid initialization amplitude: {exc}") from exc
        if amplitude <= 0.0:
            raise ValueError("initialization.amplitude must be positive")
        if isinstance(self.max_segments, (bool, np.bool_)) or not isinstance(
            self.max_segments, Integral
        ):
            raise ValueError("initialization.max_segments must be a positive integer")
        max_segments = int(self.max_segments)
        if max_segments <= 0:
            raise ValueError("initialization.max_segments must be a positive integer")

        object.__setattr__(self, "amplitude", float(self.amplitude))
        object.__setattr__(self, "max_segments", max_segments)
        object.__setattr__(self, "amplitude_v_per_m", amplitude)


LocalControlInitialization: TypeAlias = (
    LocalNoInitialization | LocalSeedFieldInitialization
)


def parse_local_initialization(value: Any) -> LocalControlInitialization:
    """Parse an explicit local-control initialization without fallback."""
    if not isinstance(value, Mapping):
        raise TypeError("initialization must be a mapping")
    if "method" not in value:
        raise ValueError("missing required local initialization option: method")

    method = value["method"]
    if method == "none":
        inapplicable = set(value) - {"method"}
        if inapplicable:
            raise ValueError(
                "local initialization options are not applicable when "
                "initialization.method='none': " + _names(inapplicable)
            )
        return LocalNoInitialization()

    if method == "seed_field":
        unknown = set(value) - _SEED_FIELD_KEYS
        if unknown:
            raise ValueError(
                "unsupported seed-field initialization options: " + _names(unknown)
            )
        missing = _SEED_FIELD_KEYS - set(value)
        if missing:
            raise ValueError(
                "missing required seed-field initialization options: " + _names(missing)
            )
        return LocalSeedFieldInitialization(
            amplitude=value["amplitude"],
            amplitude_units=value["amplitude_units"],
            max_segments=value["max_segments"],
        )

    raise ValueError("initialization.method must be one of: none, seed_field")


__all__ = [
    "LocalControlInitialization",
    "LocalNoInitialization",
    "LocalSeedFieldInitialization",
    "parse_local_initialization",
]
