"""Stateless vibrational-ladder dipole builder.

The authoritative stateful class lives in :mod:`.dipole`; this module only
provides the explicit one-shot :func:`build_mu` convenience.
"""

from __future__ import annotations

from typing import Literal

from .dipole import (
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

    NumPy supports dense and CSR storage. CuPy supports dense storage only.
    """
    obj = _CacheVibLadderDipoleMatrix(
        basis=basis,
        mu0=mu0,
        potential_type=potential_type,
        backend=backend,
        dense=dense,
    )
    return obj.mu(axis)
