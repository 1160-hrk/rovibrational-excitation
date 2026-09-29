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


def sparse_commutator(mu_int: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Return the existing CSR-computed interaction commutator as dense."""
    from scipy.sparse import csr_matrix

    mu_int_sparse = csr_matrix(mu_int)
    rho_sparse = csr_matrix(rho)
    commutator_sparse = mu_int_sparse @ rho_sparse - rho_sparse @ mu_int_sparse
    return np.asarray(commutator_sparse.toarray())


def select_response_entries(
    commutator: np.ndarray,
    mu_det_support: np.ndarray,
    relative_threshold: float | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Select exact or explicit approximate response entries and loss."""
    response_relevant = ~np.eye(commutator.shape[0], dtype=bool)
    response_relevant &= mu_det_support.T
    nonzero = response_relevant & (commutator != 0.0)

    if relative_threshold is None or not np.any(nonzero):
        i_indices, j_indices = np.where(nonzero)
        return i_indices, j_indices, 0.0

    magnitudes = np.abs(commutator)
    scale = float(np.max(magnitudes[nonzero]))
    retained = nonzero & (magnitudes >= relative_threshold * scale)
    discarded = nonzero & ~retained
    total_norm = float(np.linalg.norm(commutator[nonzero]))
    discarded_norm = float(np.linalg.norm(commutator[discarded]))
    discarded_fraction = discarded_norm / total_norm if total_norm > 0.0 else 0.0
    i_indices, j_indices = np.where(retained)
    return i_indices, j_indices, discarded_fraction


def calculate_chunked_response(
    commutator: np.ndarray,
    wavenumber: np.ndarray,
    i_indices: np.ndarray,
    j_indices: np.ndarray,
    mu_det: np.ndarray,
    complex_bohr_frequencies: np.ndarray,
    *,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the existing chunked response with fixed entry order."""
    response_sum = np.zeros(len(wavenumber), dtype=complex)

    for start_idx in range(0, len(wavenumber), chunk_size):
        end_idx = min(start_idx + chunk_size, len(wavenumber))
        omega_chunk = 2 * np.pi * CONSTANTS.C * wavenumber[start_idx:end_idx] * 100
        response_chunk = np.zeros(len(omega_chunk), dtype=complex)

        for _idx, (i, j) in enumerate(zip(i_indices, j_indices)):
            if i != j:
                omega_ij = complex_bohr_frequencies[j, i]
                mu_det_ij = mu_det[j, i]
                rho1_ij = commutator[i, j]
                denominator = 1j * (omega_chunk + omega_ij)
                kernel = -1.0 / denominator
                response_chunk += (1j / CONSTANTS.HBAR) * mu_det_ij * rho1_ij * kernel

        response_sum[start_idx:end_idx] = response_chunk

    omega = 2 * np.pi * CONSTANTS.C * wavenumber * 100
    return omega, response_sum
