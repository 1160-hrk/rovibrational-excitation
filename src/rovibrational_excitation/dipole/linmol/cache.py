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
class LinMolDipoleMatrix:
    basis: LinMolBasis
    mu0: float = 1.0
    potential_type: Literal["harmonic", "morse"] = "harmonic"
    backend: Literal["numpy", "cupy"] = "numpy"
    dense: bool = True
    units: Literal["C*m", "D", "ea0"] = "C*m"  # Unit information

    _cache: dict[tuple[str, bool], Array] = field(  # type: ignore[valid-type]
        init=False, default_factory=dict, repr=False
    )

    # ------------------------------------------------------------------
    # Core method
    # ------------------------------------------------------------------
    def mu(
        self,
        axis: str = "x",
        *,
        dense: bool | None = None,
    ) -> Array:  # type: ignore[valid-type]
        """
        Return μ_axis; build and cache on first request.

        Parameters
        ----------
        axis   : 'x' | 'y' | 'z'
        dense  : override class-level dense flag
        """
        # 大文字小文字を区別しないように正規化してからチェック
        axis_normalized = axis.lower()
        if axis_normalized not in ("x", "y", "z"):
            raise ValueError("axis must be x, y or z")

        # 型アサーションでmypyを満足させる
        axis_literal: Literal["x", "y", "z"] = axis_normalized  # type: ignore[assignment]

        if dense is None:
            dense = self.dense
        key = (axis_normalized, dense)
        if key not in self._cache:
            self._cache[key] = build_mu(
                self.basis,
                axis_literal,
                self.mu0,
                potential_type=self.potential_type,
                backend=self.backend,
                dense=dense,
            )
        return self._cache[key]

    # convenience properties
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
    # Unit management
    # ------------------------------------------------------------------
    def get_mu_in_units(self, axis: str, target_units: str, *, dense: bool | None = None) -> Array:  # type: ignore[valid-type]
        """
        Get dipole matrix in specified units.
        
        Parameters
        ----------
        axis : str
            Dipole axis ('x', 'y', 'z')
        target_units : str
            Target units ('C*m', 'D', 'ea0')
        dense : bool, optional
            Override class-level dense setting
            
        Returns
        -------
        Array
            Dipole matrix converted to target units
        """
        # Physical constants for unit conversion
        _DEBYE_TO_CM = 3.33564e-30  # D → C·m
        _EA0_TO_CM = 1.602176634e-19 * 5.29177210903e-11  # ea0 → C·m
        
        # Get matrix in current units
        matrix = self.mu(axis, dense=dense)
        
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
    
    def get_mu_x_SI(self, *, dense: bool | None = None) -> Array:  # type: ignore[valid-type]
        """Get μ_x in SI units (C·m)."""
        return self.get_mu_in_units("x", "C*m", dense=dense)
    
    def get_mu_y_SI(self, *, dense: bool | None = None) -> Array:  # type: ignore[valid-type]
        """Get μ_y in SI units (C·m)."""
        return self.get_mu_in_units("y", "C*m", dense=dense)
    
    def get_mu_z_SI(self, *, dense: bool | None = None) -> Array:  # type: ignore[valid-type]
        """Get μ_z in SI units (C·m)."""
        return self.get_mu_in_units("z", "C*m", dense=dense)

    # ------------------------------------------------------------------
    def stacked(self, order: str = "xyz", *, dense: bool | None = None) -> Array:  # type: ignore[valid-type]
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
                if dn:  # dense ndarray / cupy
                    g.create_dataset("data", data=np.asarray(mat))
                else:  # CSR sparse
                    if sp.issparse(mat):
                        mat_coo = mat.tocoo()
                    else:
                        mat_coo = mat.tocoo()  # type: ignore[attr-defined]

                    if hasattr(_xp(self.backend), "asnumpy"):
                        g.create_dataset(
                            "row", data=_xp(self.backend).asnumpy(mat_coo.row)
                        )  # type: ignore[attr-defined]
                        g.create_dataset(
                            "col", data=_xp(self.backend).asnumpy(mat_coo.col)
                        )  # type: ignore[attr-defined]
                        g.create_dataset(
                            "data", data=_xp(self.backend).asnumpy(mat_coo.data)
                        )  # type: ignore[attr-defined]
                    else:
                        g.create_dataset("row", data=np.asarray(mat_coo.row))
                        g.create_dataset("col", data=np.asarray(mat_coo.col))
                        g.create_dataset("data", data=np.asarray(mat_coo.data))
                    g.attrs["shape"] = mat_coo.shape

    @classmethod
    def from_hdf5(cls, path: str, basis: LinMolBasis) -> LinMolDipoleMatrix:
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
                        arr = cp.asarray(arr)  # type: ignore[attr-defined]
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
        cached = ", ".join(
            f"{ax}({'dense' if d else 'sparse'})" for (ax, d) in self._cache
        )
        return (
            f"<LinMolDipoleMatrix mu0={self.mu0} "
            f"potential='{self.potential_type}' "
            f"backend='{self.backend}' dense={self.dense} "
            f"cached=[{cached}]>"
        )
