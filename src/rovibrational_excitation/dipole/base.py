"""
Common base class for dipole-moment matrices with unit management.

Concrete subclasses (LinMolDipoleMatrix, VibLadderDipoleMatrix,
TwoLevelDipoleMatrix, …) should inherit from this class and implement
``_build_mu_axis(axis: str, dense: bool) -> Array`` which returns the
matrix for the requested axis in *internal* units.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Union, Dict, Tuple

import numpy as np
from rovibrational_excitation.core.units.converters import converter

try:
    import cupy as cp  # optional GPU backend
except ImportError:
    cp = None  # noqa: N816

# -----------------------------------------------------------------------------
# Typing helper
# -----------------------------------------------------------------------------
if cp is not None:
    Array: type = Union[np.ndarray, cp.ndarray]  # type: ignore[assignment]
else:
    Array: type = np.ndarray  # type: ignore[assignment,no-redef]


# -----------------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------------

def _xp(backend: str):
    """Return NumPy or CuPy module depending on backend string."""
    return cp if (backend == "cupy" and cp is not None) else np


# -----------------------------------------------------------------------------
# Base class
# -----------------------------------------------------------------------------

class DipoleMatrixBase(ABC):
    """Mixin / abstract base that consolidates common dipole-matrix logic."""

    # These attributes must be supplied by subclasses (e.g. via @dataclass)
    mu0: float
    backend: Literal["numpy", "cupy"]
    dense: bool
    units: Literal["C*m", "D", "ea0"]
    units_input: Literal["C*m", "D", "ea0"]

    # Cache: key = (axis, dense)
    _cache: Dict[Tuple[str, bool], Array]

    # ------------------------------------------------------------------
    # Initialisation helper
    # ------------------------------------------------------------------
    def _convert_mu0_if_needed(self) -> None:
        """Convert *input* mu0 to internal units once at construction."""
        if getattr(self, "units_input", self.units) != self.units:
            self.mu0 = float(
                converter.convert_dipole_moment(self.mu0, self.units_input, self.units)
            )
            # After conversion keep units_input consistent
            self.units_input = self.units  # type: ignore[attr-defined]

    # Subclasses should call this in their __post_init__ (dataclass) or __init__.

    # ------------------------------------------------------------------
    # Abstract builder implemented by concrete classes
    # ------------------------------------------------------------------
    @abstractmethod
    def _build_mu_axis(self, axis: Literal["x", "y", "z"], *, dense: bool) -> Array:
        """Return dipole matrix for given axis in *internal* units."""

    # ------------------------------------------------------------------
    # Public API – shared implementation
    # ------------------------------------------------------------------
    def mu(self, axis: str = "x", *, dense: bool | None = None) -> Array:  # type: ignore[valid-type]
        axis_norm = axis.lower()
        if axis_norm not in ("x", "y", "z"):
            raise ValueError("axis must be 'x', 'y' or 'z'")
        if dense is None:
            dense = getattr(self, "dense", True)
        key = (axis_norm, dense)
        if key not in self._cache:
            self._cache[key] = self._build_mu_axis(axis_norm, dense=dense)
        return self._cache[key]

    # Convenience properties --------------------------------------------------
    @property
    def mu_x(self):
        return self.mu("x")

    @property
    def mu_y(self):
        return self.mu("y")

    @property
    def mu_z(self):
        return self.mu("z")

    # ------------------------------------------------------------------
    # Unit conversion (shared)
    # ------------------------------------------------------------------
    def get_mu_in_units(self, axis: str, target_units: str, *, dense: bool | None = None) -> Array:  # type: ignore[valid-type]
        """Return dipole matrix converted to *target_units*."""
        mat = self.mu(axis, dense=dense)
        if self.units == target_units:
            return mat
        factor = converter.convert_dipole_moment(1.0, self.units, target_units)
        return mat * factor

    # SI helpers --------------------------------------------------------------
    def get_mu_x_SI(self, *, dense: bool | None = None) -> Array:  # type: ignore[valid-type]
        return self.get_mu_in_units("x", "C*m", dense=dense)

    def get_mu_y_SI(self, *, dense: bool | None = None) -> Array:  # type: ignore[valid-type]
        return self.get_mu_in_units("y", "C*m", dense=dense)

    def get_mu_z_SI(self, *, dense: bool | None = None) -> Array:  # type: ignore[valid-type]
        return self.get_mu_in_units("z", "C*m", dense=dense)

    # ------------------------------------------------------------------
    def stacked(self, order: str = "xyz", *, dense: bool | None = None) -> Array:  # type: ignore[valid-type]
        """Return stack with shape (len(order), dim, dim)."""
        mats = [self.mu(ax, dense=dense) for ax in order]
        return _xp(self.backend).stack(mats)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        cached = ", ".join(f"{ax}({'dense' if d else 'sparse'})" for (ax, d) in self._cache)
        return (
            f"<{self.__class__.__name__} mu0={self.mu0} units='{self.units}' "
            f"backend='{self.backend}' cached=[{cached}]>"
        )

    # ------------------------------------------------------------------
    # Persistence (shared, optional)
    # ------------------------------------------------------------------
    def to_hdf5(self, path: str) -> None:
        """Save cached dipole matrices to HDF5 (requires h5py)."""
        import h5py
        import scipy.sparse as sp

        with h5py.File(path, "w") as h5:
            h5.attrs.update(dict(mu0=self.mu0, backend=self.backend, dense=self.dense))
            for (ax, dn), mat in self._cache.items():
                g = h5.create_group(f"{ax}_{'dense' if dn else 'sparse'}")
                if dn:
                    g.create_dataset("data", data=np.asarray(mat))
                else:
                    mat_coo = mat.tocoo() if sp.issparse(mat) else mat.tocoo()  # type: ignore[attr-defined]
                    g.create_dataset("row", data=np.asarray(mat_coo.row))
                    g.create_dataset("col", data=np.asarray(mat_coo.col))
                    g.create_dataset("data", data=np.asarray(mat_coo.data))
                    g.attrs["shape"] = mat_coo.shape

    @classmethod
    def from_hdf5(cls, path: str, **kwargs):  # type: ignore[override]
        """Generic loader; subclasses pass their own `basis` via kwargs."""
        import h5py
        import scipy.sparse as sp

        with h5py.File(path, "r") as h5:
            obj = cls(
                mu0=float(h5.attrs["mu0"]),
                backend=h5.attrs["backend"],
                dense=bool(h5.attrs["dense"]),
                **kwargs,  # e.g., basis=...
            )  # type: ignore[arg-type]

            for name, g in h5.items():
                ax, typ = name.split("_")
                dn = typ == "dense"
                if dn:
                    arr = g["data"][...]
                    obj._cache[(ax, True)] = arr.astype(np.complex128)
                else:
                    shape = g.attrs["shape"]
                    row = g["row"][...]
                    col = g["col"][...]
                    dat = g["data"][...]
                    mat = sp.coo_matrix((dat, (row, col)), shape=shape).tocsr()
                    obj._cache[(ax, False)] = mat
        return obj 