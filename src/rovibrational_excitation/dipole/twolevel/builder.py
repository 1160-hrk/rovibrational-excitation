"""Stateless two-level dipole builder.

The authoritative stateful class lives in :mod:`.cache`; this module only
provides the explicit one-shot :func:`build_mu` convenience.
"""

from __future__ import annotations

from typing import Literal

from rovibrational_excitation.dipole.twolevel.cache import (
    TwoLevelDipoleMatrix as _CacheTwoLevelDipoleMatrix,
)


def build_mu(
    basis,
    axis: Literal["x", "y", "z"],
    mu0: float,
    *,
    backend: Literal["numpy", "cupy"] = "numpy",
    dense: bool = True,
):
    """Stateless builder for μ_axis in a two-level system.

    NumPy supports dense and CSR storage. CuPy supports dense storage only.
    """
    obj = _CacheTwoLevelDipoleMatrix(basis=basis, mu0=mu0, backend=backend, dense=dense)
    return obj.mu(axis)
