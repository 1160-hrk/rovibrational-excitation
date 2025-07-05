"""
Physical constants used in rovibrational excitation calculations.

All constants are in SI base units (no prefixes).
"""

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class PhysicalConstants:
    """
    Physical constants in SI base units.
    
    This class provides a centralized, immutable collection of physical
    constants used throughout the codebase. All values are in SI base
    units to ensure consistency.
    """
    
    # Fundamental constants
    C: ClassVar[float] = 299792458  # Speed of light [m/s]
    H: ClassVar[float] = 6.62607015e-34  # Planck constant [J·s]
    HBAR: ClassVar[float] = 1.054571817e-34  # Reduced Planck constant [J·s]
    E: ClassVar[float] = 1.602176634e-19  # Elementary charge [C]
    
    # Atomic units
    A0: ClassVar[float] = 5.29177210903e-11  # Bohr radius [m]
    
    # Electromagnetic constants
    MU0: ClassVar[float] = 1.25663706212e-6  # Vacuum permeability [H/m]
    EPSILON0: ClassVar[float] = 8.8541878128e-12  # Vacuum permittivity [F/m]
    
    # Unit conversions
    DEBYE_TO_CM: ClassVar[float] = 3.33564e-30  # Debye to C·m
    EV_TO_J: ClassVar[float] = 1.602176634e-19  # eV to J
    
    @classmethod
    def get_hbar_in_units(cls, units: str) -> float:
        """Get ℏ in specified units."""
        if units == "J·s":
            return cls.HBAR
        elif units == "eV·s":
            return cls.HBAR / cls.EV_TO_J
        elif units == "J·fs":
            return cls.HBAR * 1e15
        else:
            raise ValueError(f"Unknown units for ℏ: {units}")


# Create a singleton instance for easy access
CONSTANTS = PhysicalConstants() 