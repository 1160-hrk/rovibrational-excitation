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
from typing import Dict, Literal, TYPE_CHECKING, Union

import numpy as np

try:
    import cupy as cp                      # optional GPU backend
except ImportError:
    cp = None                              # noqa: N816  (keep lower-case)

# ----------------------------------------------------------------------
# Forward-refs for static type checkers only
# ----------------------------------------------------------------------
if TYPE_CHECKING:
    from rovibrational_excitation.core.basis import LinMolBasis

    # runtime-independent Array alias
    if cp is not None:
        Array = Union[np.ndarray, "cp.ndarray"]
    else:
        Array = np.ndarray
else:
    Array = np.ndarray                     # runtime alias (for annotations)

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
class LinMolDipoleMatrix:
    basis: "LinMolBasis"
    mu0: float = 1.0
    potential_type: Literal["harmonic", "morse"] = "harmonic"
    backend: Literal["numpy", "cupy"] = "numpy"
    dense: bool = True

    _cache: Dict[tuple[str, bool], Array] = field(
        init=False, default_factory=dict, repr=False
    )

    # ------------------------------------------------------------------
    # Core method
    # ------------------------------------------------------------------
    def mu(
        self,
        axis: Literal["x", "y", "z"] = "x",
        *,
        dense: bool | None = None,
    ) -> Array:
        """
        Return μ_axis; build and cache on first request.

        Parameters
        ----------
        axis   : 'x' | 'y' | 'z'
        dense  : override class-level dense flag
        """
        if dense is None:
            dense = self.dense
        key = (axis, dense)
        if key not in self._cache:
            self._cache[key] = build_mu(
                self.basis,
                axis,
                self.mu0,
                potential_type=self.potential_type,
                backend=self.backend,
                dense=dense,
            )
        return self._cache[key]

    # convenience properties
    @property
    def mu_x(self): return self.mu("x")

    @property
    def mu_y(self): return self.mu("y")

    @property
    def mu_z(self): return self.mu("z")

    # ------------------------------------------------------------------
    def stacked(self, order: str = "xyz", *, dense: bool | None = None) -> Array:
        """Return stack with shape (len(order), dim, dim)."""
        mats = [self.mu(ax, dense=dense) for ax in order]
        return _xp(self.backend).stack(mats)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def to_hdf5(self, path: str) -> None:
        """Save cached matrices to HDF5 (requires h5py)."""
        import h5py
        import scipy.sparse as sp

        with h5py.File(path, "w") as h5:
            h5.attrs.update(
                dict(
                    mu0=self.mu0,
                    backend=self.backend,
                    dense=self.dense,
                    potential_type=self.potential_type,
                )
            )
            for (ax, dn), mat in self._cache.items():
                g = h5.create_group(f"{ax}_{'dense' if dn else 'sparse'}")
                if dn:                                 # dense ndarray / cupy
                    g.create_dataset("data", data=np.asarray(mat))
                else:                                 # CSR sparse
                    mat_coo = mat.tocoo() if sp.issparse(mat) else mat.tocoo()
                    g.create_dataset("row", data=_xp(self.backend).asnumpy(mat_coo.row))
                    g.create_dataset("col", data=_xp(self.backend).asnumpy(mat_coo.col))
                    g.create_dataset("data", data=_xp(self.backend).asnumpy(mat_coo.data))
                    g.attrs["shape"] = mat_coo.shape

    @classmethod
    def from_hdf5(cls, path: str, basis: "LinMolBasis") -> "LinMolDipoleMatrix":
        """Load object saved by :meth:`to_hdf5`."""
        import h5py
        import scipy.sparse as sp

        with h5py.File(path, "r") as h5:
            obj = cls(
                basis=basis,
                mu0=float(h5.attrs["mu0"]),
                potential_type=h5.attrs.get("potential_type", "harmonic"),
                backend=h5.attrs["backend"],
                dense=bool(h5.attrs["dense"]),
            )
            for name, g in h5.items():
                ax, typ = name.split("_")
                dn = typ == "dense"
                if dn:
                    arr = g["data"][...]
                    if obj.backend == "cupy":
                        arr = cp.asarray(arr)          # type: ignore[attr-defined]
                    obj._cache[(ax, True)] = arr.astype(np.complex128)
                else:
                    shape = g.attrs["shape"]
                    row = g["row"][...]
                    col = g["col"][...]
                    dat = g["data"][...]
                    mat = sp.coo_matrix((dat, (row, col)), shape=shape).tocsr()
                    obj._cache[(ax, False)] = mat
        return obj

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        cached = ", ".join(f"{ax}({'dense' if d else 'sparse'})"
                           for (ax, d) in self._cache)
        return (
            f"<LinMolDipoleMatrix mu0={self.mu0} "
            f"potential='{self.potential_type}' "
            f"backend='{self.backend}' dense={self.dense} "
            f"cached=[{cached}]>"
        )
