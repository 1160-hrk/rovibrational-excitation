"""
rovibrational_excitation.dipole.linmol/cache.py
======================
Lazy, cached wrapper around ``rovibrational_excitation.dipole.linmol.builder`` that supports

* NumPy / CuPy backend
* dense or CSR-sparse matrices
* vibrational potential switch: ``potential_type = "harmonic" | "morse"``

Typical usage
-------------
>>> dip = LinMolDipoleMatrix(basis,
...                          mu0=0.3,
...                          potential_type="morse",
...                          backend="cupy",
...                          dense=False)
>>> mu_x = dip.mu_x
>>> mu_xyz = dip.stacked()
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
    from rovibrational_excitation.core.basis import LinMolBasis

# Runtime用の型エイリアス
if cp is not None:
    Array: type = Union[np.ndarray, cp.ndarray]  # type: ignore[assignment]
else:
    Array: type = np.ndarray  # type: ignore[assignment,no-redef]

from rovibrational_excitation.dipole.linmol.builder import build_mu


# ----------------------------------------------------------------------
# Helper
# ----------------------------------------------------------------------
def _xp(backend: str):
    return cp if (backend == "cupy" and cp is not None) else np


# ----------------------------------------------------------------------
# Main class
# ----------------------------------------------------------------------
@dataclass(slots=True)
class LinMolDipoleMatrix(DipoleMatrixBase):
    basis: LinMolBasis
    mu0: float = 1.0
    potential_type: Literal["harmonic", "morse"] = "harmonic"
    backend: Literal["numpy", "cupy"] = "numpy"
    dense: bool = True
    units: Literal["C*m", "D", "ea0"] = "C*m"           # internal storage units
    units_input: Literal["C*m", "D", "ea0"] = "C*m"     # units in which mu0 is provided

    _cache: dict[tuple[str, bool], Array] = field(  # type: ignore[type-arg]
        init=False, default_factory=dict, repr=False
    )

    # ------------------------------------------------------------------
    # concrete implementation required by DipoleMatrixBase
    # ------------------------------------------------------------------
    def _build_mu_axis(self, axis: Literal["x", "y", "z"], *, dense: bool) -> Array:  # type: ignore[override]
        """Build dipole matrix for LinMol system (no caching here)."""
        return build_mu(
            self.basis,
            axis,
            self.mu0,
            potential_type=self.potential_type,
            backend=self.backend,
            dense=dense,
        )

    # ------------------------------------------------------------------
    # DipoleMatrixBase already provides unit conversion, stacking, persistence, __repr__
    # ------------------------------------------------------------------
    def __post_init__(self):
        # convert mu0 units using base helper
        self._convert_mu0_if_needed()
