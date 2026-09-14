"""
Dipole moment matrices for various quantum systems.

This package provides dipole moment matrix classes with internal unit management
for different quantum systems:

- LinMolDipoleMatrix: Linear molecules (vibration + rotation + magnetic quantum numbers)
- VibLadderDipoleMatrix: Vibrational ladder systems (rotation-free)
- SymTopDipoleMatrix: Symmetric top molecules

All classes support automatic unit conversion between:
- C·m (SI units)
- D (Debye)
- ea0 (atomic units)

Two-level basis and dipole types are owned by
``rovibrational_excitation.models.two_level``. The generic factory is available
explicitly from ``rovibrational_excitation.dipole.factory``.
"""

from .linmol import LinMolDipoleMatrix
from .symtop import SymTopDipoleMatrix
from .viblad import VibLadderDipoleMatrix

__all__ = [
    "LinMolDipoleMatrix",
    "VibLadderDipoleMatrix",
    "SymTopDipoleMatrix",
]
