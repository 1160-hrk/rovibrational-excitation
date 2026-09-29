"""Exact dense linear-response kernels for spectroscopy."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np

from rovibrational_excitation.core.units.constants import CONSTANTS

DopplerBroadener = Callable[[np.ndarray, np.ndarray, float], np.ndarray]


def prepare_2d_denominators(
    wavenumber: np.ndarray,
    transition_indices: np.ndarray,
    complex_bohr_frequencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the existing 2D angular-frequency and denominator arrays."""
    omega = 2 * np.pi * CONSTANTS.C * 1e2 * wavenumber
    omega_2d = omega.reshape(-1, 1)

    n_freq = len(omega)
    n_trans = transition_indices.shape[1]
    one_over_denominator = np.zeros((n_freq, n_trans), dtype=np.complex128)

    for idx, transition in enumerate(transition_indices.T):
        i, j = tuple(transition)
        one_over_denominator[:, idx] = 1 / (
            1j * (omega + complex_bohr_frequencies[i, j])
        )

    return omega_2d, one_over_denominator


def calculate_2d_response(
    rho: np.ndarray,
    mu_int: np.ndarray,
    mu_det: np.ndarray,
    transition_indices: np.ndarray,
    one_over_denominator: np.ndarray,
) -> np.ndarray:
    """Evaluate the existing cached two-dimensional response route."""
    rho_after_int = mu_int @ rho - rho @ mu_int

    intensity_factors = np.zeros(
        transition_indices.shape[1],
        dtype=np.complex128,
    )
    for idx, transition in enumerate(transition_indices.T):
        i, j = tuple(transition)
        intensity_factors[idx] = (
            -1j / CONSTANTS.HBAR * mu_det[i, j] * rho_after_int[j, i]
        )

    response_2d = one_over_denominator * intensity_factors
    return np.asarray(np.sum(response_2d, axis=1))


def calculate_matrix_response(
    rho: np.ndarray,
    wavenumber: np.ndarray,
    mu_int: np.ndarray,
    mu_det: np.ndarray,
    transition_indices: np.ndarray,
    complex_bohr_frequencies: np.ndarray,
    *,
    apply_doppler: bool,
    doppler_broadener: DopplerBroadener,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the existing list-and-sum matrix response route."""
    omega = 2 * np.pi * CONSTANTS.C * 1e2 * wavenumber
    rho_after_int = mu_int @ rho - rho @ mu_int

    responses: list[np.ndarray] = []
    for transition in transition_indices.T:
        i, j = tuple(transition)
        transition_response = (
            -1j
            / CONSTANTS.HBAR
            * mu_det[i, j]
            * rho_after_int[j, i]
            / (1j * (omega + complex_bohr_frequencies[i, j]))
        )

        if apply_doppler:
            omega_trans = float(np.real(complex_bohr_frequencies[i, j]))
            transition_response = doppler_broadener(
                omega, transition_response, omega_trans
            )

        responses.append(transition_response)

    response_sum = cast(np.ndarray, np.sum(responses, axis=0))
    return omega, response_sum


def calculate_loop_response(
    rho: np.ndarray,
    wavenumber: np.ndarray,
    mu_int: np.ndarray,
    mu_det: np.ndarray,
    transition_indices: np.ndarray,
    complex_bohr_frequencies: np.ndarray,
    *,
    apply_doppler: bool,
    doppler_broadener: DopplerBroadener,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the existing in-place accumulation response route."""
    omega = 2 * np.pi * CONSTANTS.C * 1e2 * wavenumber
    rho_after_int = mu_int @ rho - rho @ mu_int
    response_sum = np.zeros(len(wavenumber), dtype=np.complex128)

    for transition in transition_indices.T:
        i, j = tuple(transition)
        transition_response = (
            -1j
            / CONSTANTS.HBAR
            * mu_det[i, j]
            * rho_after_int[j, i]
            / (1j * (omega + complex_bohr_frequencies[i, j]))
        )

        if apply_doppler:
            omega_trans = float(np.real(complex_bohr_frequencies[i, j]))
            transition_response = doppler_broadener(
                omega, transition_response, omega_trans
            )

        response_sum += transition_response

    return omega, response_sum
