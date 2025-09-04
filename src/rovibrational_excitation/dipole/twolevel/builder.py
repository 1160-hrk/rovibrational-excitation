"""
Two-level dipole: builder facade and compatibility shim.

This module now defers the authoritative implementation to
``rovibrational_excitation.dipole.twolevel.cache`` (DipoleMatrixBase-based).

Provided here:
- a stateless ``build_mu(...)`` convenience that constructs μ_axis; and
- a thin, deprecated wrapper class ``TwoLevelDipoleMatrix`` that delegates to
  the cache implementation for backwards compatibility with older imports.
"""

from __future__ import annotations

import warnings
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


class TwoLevelDipoleMatrix(_CacheTwoLevelDipoleMatrix):
    """Deprecated wrapper over the cache-based implementation.

    Prefer importing from ``rovibrational_excitation.dipole.twolevel``
    or ``rovibrational_excitation.dipole`` instead of ``.twolevel.builder``.
    """

    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        warnings.warn(
            "rovibrational_excitation.dipole.twolevel.builder.TwoLevelDipoleMatrix"
            " is deprecated; please import from rovibrational_excitation.dipole.twolevel"
            " (cache) or rovibrational_excitation.dipole.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
