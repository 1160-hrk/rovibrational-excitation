"""Conversion of molecular response functions to measured observables."""

from __future__ import annotations

import numpy as np

from rovibrational_excitation.core.units.constants import CONSTANTS


def response_to_absorbance(
    omega: np.ndarray,
    response: np.ndarray,
    *,
    number_density: float,
    optical_length_m: float,
) -> np.ndarray:
    """Convert a linear molecular response to absorbance in mOD."""
    result = np.sqrt(1 + response / CONSTANTS.EPSILON0 * number_density)
    absorbance = 2 * optical_length_m * omega / CONSTANTS.C * result.imag
    absorbance *= np.log10(np.exp(1)) * 1000
    return np.asarray(absorbance)
