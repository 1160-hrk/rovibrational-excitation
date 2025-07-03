"""
Vibrational ladder system dipole matrix with unit management.

This module provides VibLadderDipoleMatrix with internal unit preservation
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
    from rovibrational_excitation.core.basis import VibLadderBasis

# Runtime用の型エイリアス
if cp is not None:
    Array: type = Union[np.ndarray, cp.ndarray]  # type: ignore[assignment]
else:
    Array: type = np.ndarray  # type: ignore[assignment,no-redef]

from rovibrational_excitation.dipole.vib.harmonic import tdm_vib_harm
from rovibrational_excitation.dipole.vib.morse import omega01_domega_to_N, tdm_vib_morse


# ----------------------------------------------------------------------
# Helper
# ----------------------------------------------------------------------
def _xp(backend: str):
    return cp if (backend == "cupy" and cp is not None) else np


# ----------------------------------------------------------------------
# Main class
# ----------------------------------------------------------------------
@dataclass(slots=True)
class VibLadderDipoleMatrix:
    """
    Vibrational ladder system dipole matrix with unit management.

    For vibrational systems without rotation, only the z-component
    of the dipole moment is typically non-zero (parallel transitions).
    
    Includes automatic unit conversion between C·m (SI), D (Debye), and ea0 (atomic units).
    """
    
    basis: VibLadderBasis
    mu0: float = 1.0
    potential_type: Literal["harmonic", "morse"] = "harmonic"
    backend: Literal["numpy", "cupy"] = "numpy"
    units: Literal["C*m", "D", "ea0"] = "C*m"  # Unit information

    _cache: dict[str, Array] = field(  # type: ignore[valid-type]
        init=False, default_factory=dict, repr=False
    )

    def __post_init__(self):
        """Initialize Morse parameters if needed."""
        if self.potential_type == "morse":
            omega01_domega_to_N(self.basis.omega_rad_pfs, self.basis.delta_omega_rad_pfs)

    # ------------------------------------------------------------------
    # Core method
    # ------------------------------------------------------------------
    def mu(self, axis: str = "z") -> Array:  # type: ignore[valid-type]
        """
        Return μ_axis; build and cache on first request.

        Parameters
        ----------
        axis : {'x', 'y', 'z'}
            Dipole axis direction.

        Returns
        -------
        Array
            Dipole matrix of shape (V_max+1, V_max+1) in current units.
        """
        axis_normalized = axis.lower()
        if axis_normalized not in ("x", "y", "z"):
            raise ValueError("axis must be 'x', 'y', or 'z'")

        if axis_normalized in self._cache:
            return self._cache[axis_normalized]

        xp = _xp(self.backend)
        dim = self.basis.size()
        matrix = xp.zeros((dim, dim), dtype=xp.complex128)

        if axis_normalized == "z":
            # For vibrational transitions, z-component is typically the relevant one
            vib_func = tdm_vib_morse if self.potential_type == "morse" else tdm_vib_harm

            for i in range(dim):
                v1 = self.basis.V_array[i]
                for j in range(dim):
                    v2 = self.basis.V_array[j]
                    vib_element = vib_func(v1, v2)
                    if vib_element != 0.0:
                        matrix[i, j] = self.mu0 * vib_element

        # For pure vibrational systems, x and y components are typically zero
        # (no rotational mixing)
        
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
            Order of axes (e.g., 'xyz', 'z').

        Returns
        -------
        Array
            Array of shape (len(order), dim, dim).
        """
        mats = [self.mu(ax) for ax in order]
        return _xp(self.backend).stack(mats)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        """String representation."""
        cached = ", ".join(self._cache.keys())
        return (
            f"<VibLadderDipoleMatrix mu0={self.mu0} "
            f"potential='{self.potential_type}' units='{self.units}' "
            f"backend='{self.backend}' cached=[{cached}]>"
        ) 