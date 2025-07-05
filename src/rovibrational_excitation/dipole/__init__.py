"""
Dipole moment matrices for various quantum systems.

This package provides dipole moment matrix classes with internal unit management
for different quantum systems:

- LinMolDipoleMatrix: Linear molecules (vibration + rotation + magnetic quantum numbers)
- TwoLevelDipoleMatrix: Two-level systems
- VibLadderDipoleMatrix: Vibrational ladder systems (rotation-free)
- SymTopDipoleMatrix: Symmetric top molecules

All classes support automatic unit conversion between:
- C·m (SI units)
- D (Debye)  
- ea0 (atomic units)
"""

from .linmol import LinMolDipoleMatrix
from .twolevel import TwoLevelDipoleMatrix
from .viblad import VibLadderDipoleMatrix
from .symtop import SymTopDipoleMatrix

__all__ = [
    "LinMolDipoleMatrix",
    "TwoLevelDipoleMatrix", 
    "VibLadderDipoleMatrix",
    "SymTopDipoleMatrix",
] 