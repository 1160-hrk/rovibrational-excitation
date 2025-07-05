"""
DEPRECATED: Legacy states module. Use rovibrational_excitation.core.basis instead.

This module is kept for backward compatibility.
"""

import warnings

from .basis.states import StateVector, DensityMatrix

warnings.warn(
    "Importing from rovibrational_excitation.core.states is deprecated. "
    "Use 'from rovibrational_excitation.core.basis import StateVector, DensityMatrix' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["StateVector", "DensityMatrix"] 