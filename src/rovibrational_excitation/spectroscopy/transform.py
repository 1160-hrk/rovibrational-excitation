"""Frequency-domain radiation and PFID response transforms."""

from __future__ import annotations

import numpy as np


def radiation_response(
    rho: np.ndarray,
    omega: np.ndarray,
    transition_indices: np.ndarray,
    mu_det: np.ndarray,
    complex_bohr_frequencies: np.ndarray,
) -> np.ndarray:
    """Return the existing direct post-probe radiation response."""
    response = np.zeros(len(omega), dtype=np.complex128)

    for transition in transition_indices.T:
        i, j = tuple(transition)
        response += -(
            mu_det[j, i] * rho[i, j] / (1j * (omega + complex_bohr_frequencies[i, j]))
        )

    return response
