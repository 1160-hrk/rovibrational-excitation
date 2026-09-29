"""Explicit experimental conditions for spectroscopy calculations."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rovibrational_excitation.core.units.constants import CONSTANTS


def require_exact_units(value: object, *, name: str, expected: str) -> None:
    """Validate a public unit label without guessing or converting it."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if value != expected:
        raise ValueError(f"{name} must be {expected!r}, got {value!r}")


@dataclass(frozen=True, slots=True)
class ExperimentalConditions:
    """Experimental inputs with required units and fixed canonical storage.

    This boundary currently accepts only the units used by the spectroscopy
    formulas: K, Pa, m, ps, and kg per molecule. Canonical ``*_k``, ``*_pa``,
    ``*_m``, ``*_ps``, and ``*_kg`` attributes are the only values consumed by
    numerical code. No unit is inferred and no value is silently converted.
    """

    temperature: float
    temperature_units: str
    pressure: float
    pressure_units: str
    optical_length: float
    optical_length_units: str
    coherence_time: float
    coherence_time_units: str
    molecular_mass: float
    molecular_mass_units: str
    temperature_k: float = field(init=False)
    pressure_pa: float = field(init=False)
    optical_length_m: float = field(init=False)
    coherence_time_ps: float = field(init=False)
    molecular_mass_kg: float = field(init=False)

    def __post_init__(self) -> None:
        unit_contracts = {
            "temperature_units": "K",
            "pressure_units": "Pa",
            "optical_length_units": "m",
            "coherence_time_units": "ps",
            "molecular_mass_units": "kg",
        }
        for name, expected in unit_contracts.items():
            require_exact_units(getattr(self, name), name=name, expected=expected)

        canonical_names = {
            "temperature": "temperature_k",
            "pressure": "pressure_pa",
            "optical_length": "optical_length_m",
            "coherence_time": "coherence_time_ps",
            "molecular_mass": "molecular_mass_kg",
        }
        for name, canonical_name in canonical_names.items():
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
            object.__setattr__(self, canonical_name, value)

    @property
    def number_density(self) -> float:
        """Return number density in m^-3."""
        return self.pressure_pa / (CONSTANTS.BOLTZMANN * self.temperature_k)

    @property
    def coherence_decay_rate(self) -> float:
        """Return the coherence decay rate in rad/s."""
        return 1 / (self.coherence_time_ps * 1e-12)
