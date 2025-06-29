"""Vibration transition-dipole elements (stateless)."""

from .harmonic import tdm_vib_harm
from .morse import omega01_domega_to_N, tdm_vib_morse

__all__ = ["tdm_vib_harm", "tdm_vib_morse", "omega01_domega_to_N"]
