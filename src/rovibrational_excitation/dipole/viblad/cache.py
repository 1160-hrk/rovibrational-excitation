"""
Vibrational ladder system dipole matrix with unit management.

This module provides VibLadderDipoleMatrix with internal unit preservation
similar to LinMolDipoleMatrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Union

import numpy as np

from rovibrational_excitation.dipole.base import Array, DipoleMatrixBase, _xp

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
from rovibrational_excitation.dipole.vib.morse import (
    omega01_domega_to_N,
    tdm_vib_morse,
    validate_morse_v_max,
)


# ----------------------------------------------------------------------
# Main class
# ----------------------------------------------------------------------
@dataclass(slots=True)
class VibLadderDipoleMatrix(DipoleMatrixBase):
    """
    Vibrational ladder system dipole matrix with unit management.

    For vibrational systems without rotation, only the z-component
    of the dipole moment is typically non-zero (parallel transitions).

    Includes automatic unit conversion between C·m (SI), D (Debye), and ea0 (atomic units).
    """

    basis: VibLadderBasis
    mu0: float
    potential_type: Literal["harmonic", "morse"]
    backend: Literal["numpy", "cupy"] = "numpy"
    units: Literal["C*m", "D", "ea0"] = "C*m"  # internal storage units
    units_input: Literal["C*m", "D", "ea0"] = "C*m"  # units in which mu0 is provided

    _cache: dict[tuple[str, bool], Array] = field(  # type: ignore[type-arg]
        init=False, default_factory=dict, repr=False
    )
    _morse_level_parameter: float | None = field(init=False, default=None, repr=False)

    def __post_init__(self):
        if self.potential_type not in {"harmonic", "morse"}:
            raise ValueError("potential_type must be harmonic or morse")
        # Potential-specific initialisation
        if self.potential_type == "morse":
            self._morse_level_parameter = omega01_domega_to_N(
                self.basis.omega_rad_pfs,
                self.basis.delta_omega_rad_pfs,
            )
            validate_morse_v_max(self.basis.V_max, self._morse_level_parameter)

        # Convert mu0 if units differ
        self._convert_mu0_if_needed()

    # ------------------------------------------------------------------
    # concrete builder required by DipoleMatrixBase
    # ------------------------------------------------------------------
    def _build_mu_axis(self, axis: Literal["x", "y", "z"], *, dense: bool) -> Array:  # type: ignore[override]
        xp = _xp(self.backend)
        dim = self.basis.size()
        matrix = xp.zeros((dim, dim), dtype=xp.complex128)

        if axis == "z":
            for i in range(dim):
                v1 = self.basis.V_array[i]
                for j in range(dim):
                    v2 = self.basis.V_array[j]
                    if self._morse_level_parameter is None:
                        val = tdm_vib_harm(v1, v2)
                    else:
                        val = tdm_vib_morse(
                            v1,
                            v2,
                            self._morse_level_parameter,
                        )
                    if val != 0.0:
                        matrix[i, j] = self.mu0 * val
        elif axis == "x":
            # matrix = xp.diag(xp.ones(len(self.basis.V_array)-1), 1)
            # matrix += xp.diag(xp.ones(len(self.basis.V_array)-1), -1)
            # matrix *= self.mu0
            matrix = xp.zeros((dim, dim), dtype=xp.complex128)
        elif axis == "y":
            matrix = xp.zeros((dim, dim), dtype=xp.complex128)
        return matrix

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        """String representation."""
        cached = ", ".join(self._cache.keys())
        return (
            f"<VibLadderDipoleMatrix mu0={self.mu0} "
            f"potential='{self.potential_type}' units='{self.units}' "
            f"backend='{self.backend}' cached=[{cached}]>"
        )
