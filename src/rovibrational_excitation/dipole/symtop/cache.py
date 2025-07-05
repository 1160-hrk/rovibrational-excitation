"""
rovibrational_excitation.dipole.symtop/cache.py
==============================================
Lazy, cached wrapper around ``rovibrational_excitation.dipole.symtop.builder``
that supports NumPy backend (GPU TBD), dense or CSR matrices, and vibrational
potential switch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Union

import numpy as np

from rovibrational_excitation.dipole.base import DipoleMatrixBase, _xp, Array
from rovibrational_excitation.core.units.converters import converter

try:
    import cupy as cp  # optional GPU backend
except ImportError:  # pragma: no cover – CPU-only
    cp = None  # noqa: N816

if TYPE_CHECKING:
    from rovibrational_excitation.core.basis.symtop import SymTopBasis

# Runtime type alias ----------------------------------------------------------
if cp is not None:
    Array: type = Union[np.ndarray, cp.ndarray]  # type: ignore[assignment]
else:
    Array: type = np.ndarray  # type: ignore[assignment,no-redef]

from rovibrational_excitation.dipole.symtop.builder import build_mu  # noqa: E402


@dataclass(slots=True)
class SymTopDipoleMatrix(DipoleMatrixBase):
    """Dipole matrix container for symmetric-top molecules."""

    basis: "SymTopBasis"
    mu0: float = 1.0
    potential_type: Literal["harmonic", "morse"] = "harmonic"
    backend: Literal["numpy", "cupy"] = "numpy"
    dense: bool = True
    units: Literal["C*m", "D", "ea0"] = "C*m"  # internal storage units
    units_input: Literal["C*m", "D", "ea0"] = "C*m"  # units in which mu0 is provided

    _cache: dict[tuple[str, bool], Array] = field(  # type: ignore[type-arg]
        init=False, default_factory=dict, repr=False
    )

    # ------------------------------------------------------------------
    # DipoleMatrixBase requirement
    # ------------------------------------------------------------------
    def _build_mu_axis(self, axis: Literal["x", "y", "z"], *, dense: bool) -> Array:  # type: ignore[override]
        return build_mu(
            self.basis,
            axis,
            self.mu0,
            potential_type=self.potential_type,
            backend=self.backend,
            dense=dense,
        )

    # ------------------------------------------------------------------
    def __post_init__(self):
        # Convert mu0 if necessary (helper from DipoleMatrixBase)
        self._convert_mu0_if_needed() 