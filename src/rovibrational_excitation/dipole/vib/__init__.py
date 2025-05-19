"""Vibration transition-dipole elements (stateless)."""
from .harmonic import tdm_vib_harm
from .morse    import tdm_vib_morse, omega01_domega_to_N
__all__ = ["tdm_vib_harm", "tdm_vib_morse", "omega01_domega_to_N"]
