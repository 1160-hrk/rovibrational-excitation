"""
DEPRECATED: Legacy basis module. Use rovibrational_excitation.core.basis package instead.

This module is kept for backward compatibility.
"""

import warnings

from .basis.hamiltonian import Hamiltonian
from .basis.linmol import LinMolBasis

warnings.warn(
    "Importing from rovibrational_excitation.core.basis is deprecated. "
    "Use 'from rovibrational_excitation.core.basis import LinMolBasis, Hamiltonian' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["LinMolBasis", "Hamiltonian"]
