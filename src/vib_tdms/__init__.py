"""Vibration transition-dipole elements (stateless)."""
from .harmonic import tdm_vib_harm
from .morse    import tdm_vib_morse
__all__ = ["tdm_vib_harm", "tdm_vib_morse"]
