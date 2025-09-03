"""
Vibrational ladder dipole: builder facade and compatibility shim.

This module used to provide a stateful class implementation. As part of
responsibility separation, the authoritative implementation now lives in
``rovibrational_excitation.dipole.viblad.cache`` (DipoleMatrixBase-based).

Here we provide:
- a stateless ``build_mu(...)`` convenience that constructs μ_axis; and
- a thin, deprecated wrapper class ``VibLadderDipoleMatrix`` that
  delegates to the cache implementation for backwards compatibility.
"""

from __future__ import annotations

import warnings
from typing import Literal

from rovibrational_excitation.dipole.viblad.cache import (
    VibLadderDipoleMatrix as _CacheVibLadderDipoleMatrix,
)


def build_mu(
    basis,
    axis: Literal["x", "y", "z"],
    mu0: float,
    *,
    potential_type: Literal["harmonic", "morse"] = "harmonic",
    backend: Literal["numpy", "cupy"] = "numpy",
    dense: bool = True,
):
    """Stateless builder for μ_axis.

    Notes
    -----
    - The vibrational ladder currently provides dense matrices only.
      If ``dense=False`` is requested, ``NotImplementedError`` is raised.
    - Units management is handled by the underlying cache implementation.
    """
    if not dense:
        raise NotImplementedError("VibLadder builder does not provide sparse matrices")
    obj = _CacheVibLadderDipoleMatrix(
        basis=basis,
        mu0=mu0,
        potential_type=potential_type,
        backend=backend,
    )
    return obj.mu(axis)


class VibLadderDipoleMatrix(_CacheVibLadderDipoleMatrix):
    """Deprecated wrapper over the cache-based implementation.

    Import ``VibLadderDipoleMatrix`` from ``rovibrational_excitation.dipole.viblad``
    or from ``rovibrational_excitation.dipole`` instead.
    """

    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        warnings.warn(
            "rovibrational_excitation.dipole.viblad.builder.VibLadderDipoleMatrix"
            " is deprecated; please import from rovibrational_excitation.dipole.viblad"
            " (cache) or rovibrational_excitation.dipole instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
