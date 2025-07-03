"""
Two-level system dipole matrix with unit management.

This module provides TwoLevelDipoleMatrix with internal unit preservation
similar to LinMolDipoleMatrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Union

import numpy as np

try:
    import cupy as cp  # optional GPU backend
except ImportError:
    cp = None  # noqa: N816  (keep lower-case)

# ----------------------------------------------------------------------
# Forward-refs for static type checkers only
# ----------------------------------------------------------------------
if TYPE_CHECKING:
    from rovibrational_excitation.core.basis import TwoLevelBasis

# Runtime用の型エイリアス
if cp is not None:
    Array: type = Union[np.ndarray, cp.ndarray]  # type: ignore[assignment]
else:
    Array: type = np.ndarray  # type: ignore[assignment,no-redef]


# ----------------------------------------------------------------------
# Helper
# ----------------------------------------------------------------------
def _xp(backend: str):
    return cp if (backend == "cupy" and cp is not None) else np


# ----------------------------------------------------------------------
# Main class
# ----------------------------------------------------------------------
@dataclass(slots=True)
class TwoLevelDipoleMatrix:
    """
    Two-level system dipole matrix with unit management.

    For a two-level system, the dipole matrix typically has the form:
    μ = μ₀ * (|0⟩⟨1| + |1⟩⟨0|)  (x-direction, σ_x)
    μ = μ₀ * i(|1⟩⟨0| - |0⟩⟨1|)  (y-direction, σ_y)
    μ = 0                         (z-direction, typically)
    
    Includes automatic unit conversion between C·m (SI), D (Debye), and ea0 (atomic units).
    """
    
    basis: TwoLevelBasis
    mu0: float = 1.0
    backend: Literal["numpy", "cupy"] = "numpy"
    units: Literal["C*m", "D", "ea0"] = "C*m"  # Unit information

    _cache: dict[str, Array] = field(  # type: ignore[valid-type]
        init=False, default_factory=dict, repr=False
    )

    def __post_init__(self):
        """Validate basis after initialization."""
        if not hasattr(self.basis, 'size') or self.basis.size() != 2:
            raise ValueError("basis must be TwoLevelBasis with exactly 2 states")

    # ------------------------------------------------------------------
    # Core method
    # ------------------------------------------------------------------
    def mu(self, axis: str = "x") -> Array:  # type: ignore[valid-type]
        """
        Return μ_axis; build and cache on first request.

        Parameters
        ----------
        axis : {'x', 'y', 'z'}
            Dipole axis direction.

        Returns
        -------
        Array
            2x2 dipole matrix in current units.
        """
        axis_normalized = axis.lower()
        if axis_normalized not in ("x", "y", "z"):
            raise ValueError("axis must be 'x', 'y', or 'z'")

        if axis_normalized in self._cache:
            return self._cache[axis_normalized]

        xp = _xp(self.backend)
        
        if axis_normalized == "x":
            # σ_x = |0⟩⟨1| + |1⟩⟨0| (Pauli-X matrix)
            matrix = self.mu0 * xp.array([[0, 1], [1, 0]], dtype=xp.complex128)
        elif axis_normalized == "y":
            # σ_y = i(|1⟩⟨0| - |0⟩⟨1|) (Pauli-Y matrix)
            matrix = self.mu0 * xp.array([[0, -1j], [1j, 0]], dtype=xp.complex128)
        else:  # axis_normalized == "z"
            # Typically zero for electric dipole transitions in two-level atoms
            # Could also be σ_z = |0⟩⟨0| - |1⟩⟨1| for some applications
            matrix = xp.zeros((2, 2), dtype=xp.complex128)
        
        self._cache[axis_normalized] = matrix
        return matrix

    # convenience properties
    @property
    def mu_x(self) -> Array:  # type: ignore[valid-type]
        """x-component of dipole matrix."""
        return self.mu("x")

    @property
    def mu_y(self) -> Array:  # type: ignore[valid-type]
        """y-component of dipole matrix."""
        return self.mu("y")

    @property
    def mu_z(self) -> Array:  # type: ignore[valid-type]
        """z-component of dipole matrix."""
        return self.mu("z")

    # ------------------------------------------------------------------
    # Unit management
    # ------------------------------------------------------------------
    def get_mu_in_units(self, axis: str, target_units: str) -> Array:  # type: ignore[valid-type]
        """
        Get dipole matrix in specified units.
        
        Parameters
        ----------
        axis : str
            Dipole axis ('x', 'y', 'z')
        target_units : str
            Target units ('C*m', 'D', 'ea0')
            
        Returns
        -------
        Array
            Dipole matrix converted to target units
        """
        # Physical constants for unit conversion
        _DEBYE_TO_CM = 3.33564e-30  # D → C·m
        _EA0_TO_CM = 1.602176634e-19 * 5.29177210903e-11  # ea0 → C·m
        
        # Get matrix in current units
        matrix = self.mu(axis)
        
        if self.units == target_units:
            return matrix
        
        # Convert from current units to C·m
        if self.units == "D":
            matrix_cm = matrix * _DEBYE_TO_CM
        elif self.units == "ea0":
            matrix_cm = matrix * _EA0_TO_CM
        else:  # self.units == "C*m"
            matrix_cm = matrix
        
        # Convert from C·m to target units
        if target_units == "D":
            return matrix_cm / _DEBYE_TO_CM
        elif target_units == "ea0":
            return matrix_cm / _EA0_TO_CM
        else:  # target_units == "C*m"
            return matrix_cm
    
    def get_mu_x_SI(self) -> Array:  # type: ignore[valid-type]
        """Get μ_x in SI units (C·m)."""
        return self.get_mu_in_units("x", "C*m")
    
    def get_mu_y_SI(self) -> Array:  # type: ignore[valid-type]
        """Get μ_y in SI units (C·m)."""
        return self.get_mu_in_units("y", "C*m")
    
    def get_mu_z_SI(self) -> Array:  # type: ignore[valid-type]
        """Get μ_z in SI units (C·m)."""
        return self.get_mu_in_units("z", "C*m")

    # ------------------------------------------------------------------
    def stacked(self, order: str = "xyz") -> Array:  # type: ignore[valid-type]
        """
        Return stacked dipole matrices.

        Parameters
        ----------
        order : str
            Order of axes (e.g., 'xyz', 'xy').

        Returns
        -------
        Array
            Array of shape (len(order), 2, 2).
        """
        mats = [self.mu(ax) for ax in order]
        return _xp(self.backend).stack(mats)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        """String representation."""
        cached = ", ".join(self._cache.keys())
        return (
            f"<TwoLevelDipoleMatrix mu0={self.mu0} "
            f"units='{self.units}' backend='{self.backend}' "
            f"cached=[{cached}]>"
        ) 