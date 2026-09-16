"""
Dipole moment matrices for various quantum systems.

This package provides dipole moment matrix classes with internal unit management
for different quantum systems:

- SymTopDipoleMatrix: Symmetric top molecules

All classes support automatic unit conversion between:
- C·m (SI units)
- D (Debye)
- ea0 (atomic units)

Two-level, vibrational-ladder, and linear-molecule basis/dipole types are owned by
``rovibrational_excitation.models.two_level`` and
``rovibrational_excitation.models.vib_ladder`` and
``rovibrational_excitation.models.linear_molecule``. The generic factory is
available explicitly from ``rovibrational_excitation.dipole.factory`` only for
the remaining legacy SymTop path.
"""

from .symtop import SymTopDipoleMatrix

__all__ = [
    "SymTopDipoleMatrix",
]
