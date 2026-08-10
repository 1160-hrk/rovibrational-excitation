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

    Notes
    -----
    - Only dense matrices are provided; ``dense=False`` raises NotImplementedError.
    - Units management and backend selection are handled by the cache implementation.
    """
    if not dense:
        raise NotImplementedError("TwoLevel builder does not provide sparse matrices")
    obj = _CacheTwoLevelDipoleMatrix(basis=basis, mu0=mu0, backend=backend)
    return obj.mu(axis)
