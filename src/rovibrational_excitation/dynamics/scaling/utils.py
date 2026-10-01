"""Small conversions shared by the strict nondimensionalization path."""

from __future__ import annotations

from typing import Any

import numpy as np

from rovibrational_excitation.core.units.constants import CONSTANTS

_HBAR = CONSTANTS.HBAR
_EV_TO_J = CONSTANTS.EV_TO_J


def dimensionalize_wavefunction(
    psi_prime: np.ndarray,
    scales: Any,
) -> np.ndarray:
    """Return amplitudes unchanged; wavefunctions carry no dimensional scale."""
    del scales
    return psi_prime


def get_physical_time(
    tau: np.ndarray,
    scales: Any,
) -> np.ndarray:
    """Convert dimensionless time to femtoseconds."""
    return tau * scales.t0 * 1e15
