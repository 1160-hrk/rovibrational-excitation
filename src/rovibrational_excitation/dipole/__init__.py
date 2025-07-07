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

The package also provides a factory function `create_dipole_matrix` that
automatically selects the appropriate dipole matrix class based on the
basis type.
"""

from .linmol import LinMolDipoleMatrix
from .twolevel import TwoLevelDipoleMatrix
from .viblad import VibLadderDipoleMatrix
from .symtop import SymTopDipoleMatrix
from .factory import create_dipole_matrix

__all__ = [
    "LinMolDipoleMatrix",
    "TwoLevelDipoleMatrix", 
    "VibLadderDipoleMatrix",
    "SymTopDipoleMatrix",
    "create_dipole_matrix",
] 