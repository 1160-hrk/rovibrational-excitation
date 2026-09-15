"""
Dipole moment matrices for various quantum systems.

This package provides dipole moment matrix classes with internal unit management
for different quantum systems:

- LinMolDipoleMatrix: Linear molecules (vibration + rotation + magnetic quantum numbers)
- SymTopDipoleMatrix: Symmetric top molecules

All classes support automatic unit conversion between:
- C·m (SI units)
- D (Debye)
- ea0 (atomic units)

Two-level and vibrational-ladder basis/dipole types are owned by
``rovibrational_excitation.models.two_level`` and
``rovibrational_excitation.models.vib_ladder``. The generic factory is
available explicitly from ``rovibrational_excitation.dipole.factory`` only for
the remaining legacy LinMol and SymTop paths.
"""

from .linmol import LinMolDipoleMatrix
from .symtop import SymTopDipoleMatrix

__all__ = [
    "LinMolDipoleMatrix",
    "SymTopDipoleMatrix",
]
