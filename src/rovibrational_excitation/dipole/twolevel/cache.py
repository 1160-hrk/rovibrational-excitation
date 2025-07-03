"""
Two-level system dipole matrix with unit management.

This module provides TwoLevelDipoleMatrix with internal unit preservation
similar to LinMolDipoleMatrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Union

from rovibrational_excitation.dipole.base import DipoleMatrixBase, _xp, Array
from rovibrational_excitation.core.units.converters import converter

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
# Main class
# ----------------------------------------------------------------------
@dataclass(slots=True)
class TwoLevelDipoleMatrix(DipoleMatrixBase):
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
    units: Literal["C*m", "D", "ea0"] = "C*m"           # internal storage units
    units_input: Literal["C*m", "D", "ea0"] = "C*m"     # units in which mu0 is provided

    _cache: dict[tuple[str, bool], Array] = field(  # type: ignore[type-arg]
        init=False, default_factory=dict, repr=False
    )

    def __post_init__(self):
        # basis validation
        if not hasattr(self.basis, "size") or self.basis.size() != 2:
            raise ValueError("basis must be TwoLevelBasis with exactly 2 states")

        # convert mu0 if needed
        self._convert_mu0_if_needed()

    # ------------------------------------------------------------------
    # concrete builder required by DipoleMatrixBase
    # ------------------------------------------------------------------
    def _build_mu_axis(self, axis: Literal["x", "y", "z"], *, dense: bool) -> Array:  # type: ignore[override]
        xp = _xp(self.backend)
        if axis == "x":
            return self.mu0 * xp.array([[0, 1], [1, 0]], dtype=xp.complex128)
        elif axis == "y":
            return self.mu0 * xp.array([[0, -1j], [1j, 0]], dtype=xp.complex128)
        else:  # z
            return xp.zeros((2, 2), dtype=xp.complex128)

    # DipoleMatrixBase supplies mu_x/y/z, unit conversion, persistence, repr

    # __repr__, stacked, unit-conversion helpers provided by DipoleMatrixBase 