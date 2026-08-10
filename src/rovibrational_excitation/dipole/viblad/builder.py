"""Stateless vibrational-ladder dipole builder.

The authoritative stateful class lives in :mod:`.cache`; this module only
provides the explicit one-shot :func:`build_mu` convenience.
"""

from __future__ import annotations

from typing import Literal

from rovibrational_excitation.dipole.viblad.cache import (
    VibLadderDipoleMatrix as _CacheVibLadderDipoleMatrix,
)


def build_mu(
    basis,
    axis: Literal["x", "y", "z"],
    mu0: float,
    *,
    potential_type: Literal["harmonic", "morse"],
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
