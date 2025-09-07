"""
Real-space (grid-based) utilities and propagators.

This module provides a minimal, self-contained split-operator implementation
for real-space wavefunction propagation on 1D/2D/3D Cartesian grids. It does
not affect the existing basis/matrix-based machinery and can be used
independently or injected as a custom propagator if desired.
"""

from .grids import Grid1D, Grid2D, Grid3D
from .potentials import morse_potential
from .dipoles import dipole_x_cartesian
from .split_operator import splitop_realspace

__all__ = [
    "Grid1D",
    "Grid2D",
    "Grid3D",
    "morse_potential",
    "dipole_x_cartesian",
    "splitop_realspace",
]

