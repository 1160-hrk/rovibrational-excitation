#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Absorbance spectrum calculator (mOD) – relocated from analysis.absorbance.

This module is the canonical location for absorbance calculations under
rovibrational_excitation.spectroscopy.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np
from numpy.typing import NDArray
from .utils import (
    wavenumber_cm_to_omega as util_wavenumber_cm_to_omega,
    to_energy_eigenbasis as util_to_energy_eigenbasis,
    select_transitions as util_select_transitions,
    compute_response as util_compute_response,
)


# --- Physical constants (SI) ---
HBAR: float = 1.054_571_817e-34      # [J·s]
EPS0: float = 8.854_187_8128e-12     # [F/m]
C: float = 299_792_458.0             # [m/s]
LN10: float = float(np.log(10.0))


def wavenumber_cm_to_omega(nu_tilde_cm: NDArray[np.floating]) -> NDArray[np.floating]:
    """(deprecated) Use utils.wavenumber_cm_to_omega. Kept for local usage."""
    return util_wavenumber_cm_to_omega(nu_tilde_cm, c=C)


def _validate_inputs(
    T2: float,
    number_density: float,
    path_length: float,
    rho: NDArray[np.complexfloating],
    mu_int: NDArray[np.complexfloating],
    H0: NDArray[np.complexfloating],
    mu_det: NDArray[np.complexfloating] | None,
    nu_tilde_cm: NDArray[np.floating],
) -> None:
    """Basic sanity checks for scalar inputs. Arrays are validated downstream."""
    if T2 <= 0.0:
        raise ValueError("T2 must be positive (seconds)")
    if path_length < 0.0:
        raise ValueError("path_length must be non-negative")
    if number_density < 0.0:
        raise ValueError("number_density must be non-negative")


def _to_energy_eigenbasis(
    H0: NDArray[np.complexfloating],
    rho: NDArray[np.complexfloating],
    mu_int: NDArray[np.complexfloating],
    mu_det: NDArray[np.complexfloating] | None,
    *,
    diag_atol: float = 0.0,
) -> Tuple[NDArray[np.floating], NDArray[np.complexfloating], NDArray[np.complexfloating], NDArray[np.complexfloating]]:
    return util_to_energy_eigenbasis(H0, rho, mu_int, mu_det, diag_atol=diag_atol)


def _select_transitions(
    mu_det_e: NDArray[np.complexfloating],
    rho1: NDArray[np.complexfloating],
    Omega: NDArray[np.complexfloating],
    *,
    use_sparse_transitions: bool,
    mu_nz_threshold: float,
) -> tuple[NDArray[np.complexfloating], NDArray[np.complexfloating], NDArray[np.complexfloating]]:
    return util_select_transitions(
        mu_det_e, rho1, Omega,
        use_sparse_transitions=use_sparse_transitions,
        mu_nz_threshold=mu_nz_threshold,
    )


def _compute_response(
    mu_det_nz: NDArray[np.complexfloating],
    rho1_nz: NDArray[np.complexfloating],
    Omega_nz: NDArray[np.complexfloating],
    omega: NDArray[np.floating],
    *,
    hbar: float,
    omega_chunk_size: int | None,
) -> NDArray[np.complexfloating]:
    return util_compute_response(
        mu_det_nz, rho1_nz, Omega_nz, omega, hbar=hbar, omega_chunk_size=omega_chunk_size
    )


def _resp_to_absorbance(
    resp: NDArray[np.complexfloating],
    omega: NDArray[np.floating],
    number_density: float,
    path_length: float,
    *,
    local_field: bool,
    eps0: float,
    c: float,
) -> NDArray[np.floating]:
    """Convert response to absorbance A(ν~) in milli-optical-density."""
    if local_field:
        x = (number_density / (3.0 * eps0)) * resp
        imag_sqrt = np.imag(np.sqrt(1.0 + x))
        A_mOD = (2.0 * path_length * omega / c) * imag_sqrt * (1000.0 / LN10)
    else:
        A_mOD = (
            (path_length * omega / c)
            * (number_density / (3.0 * eps0))
            * np.imag(resp)
            * (1000.0 / LN10)
        )
    return A_mOD


def compute_absorbance_spectrum(
    rho: NDArray[np.complexfloating],
    mu_int: NDArray[np.complexfloating],
    H0: NDArray[np.complexfloating],
    nu_tilde_cm: NDArray[np.floating],
    T2: float,
    number_density: float,
    path_length: float,
    mu_det: NDArray[np.complexfloating] | None = None,
    *,
    local_field: bool = True,
    return_resp: bool = False,
    # performance knobs
    use_sparse_transitions: bool = True,
    mu_nz_threshold: float = 0.0,
    omega_chunk_size: int | None = None,
    # model knobs
    eps0: float = EPS0,
    c: float = C,
    hbar: float = HBAR,
    diag_atol: float = 0.0,
):
    """Compute absorbance A(ν~) in mOD from external rho, μ, H0.

    Mathematics:
      Ω_ij = (E_i - E_j)/ħ - i/T2
      K_ij(ω) = -1 / [ i ( ω + Ω_ij ) ]
      ρ^(1) = [ μ_int , ρ ] = μ_int ρ - ρ μ_int
      resp(ω) = (i/ħ) * Σ_ij (μ_det)_{ji} * (ρ^(1))_{ij} * K_ij(ω)
      A(ω) [mOD] = (2 L ω / c) * Im{ sqrt( 1 + n*resp/(3ε0) ) } * 1e3 / ln(10)

    Notes:
      - mu_int should include probe polarization projection (μ · e).
      - mu_det should include detection polarization projection (μ · e*). If omitted, mu_det = mu_int.
      - If H0 is not diagonal, it is diagonalized; rho and μ are transformed to the energy eigenbasis.
    """
    # Basic checks
    rho = np.asarray(rho, dtype=complex)
    mu_int = np.asarray(mu_int, dtype=complex)
    H0 = np.asarray(H0, dtype=complex)
    nu_tilde_cm = np.asarray(nu_tilde_cm, dtype=float)
    _validate_inputs(T2, number_density, path_length, rho, mu_int, H0, mu_det, nu_tilde_cm)

    # Transform to energy eigenbasis (or reuse if diagonal)
    E, rho_e, mu_int_e, mu_det_e = _to_energy_eigenbasis(
        H0, rho, mu_int, mu_det, diag_atol=diag_atol
    )

    # Angular frequency grid
    omega = wavenumber_cm_to_omega(nu_tilde_cm)

    # Complex Bohr frequencies Ω_ij with dephasing (broadcast (N,N))
    E_over_hbar = (E / hbar).astype(complex)
    Omega = (E_over_hbar[:, None] - E_over_hbar[None, :]) - 1j * (1.0 / float(T2))

    # First-order density after probe interaction: commutator [μ_int, ρ]
    rho1 = mu_int_e @ rho_e - rho_e @ mu_int_e

    # Select transitions
    mu_det_nz, rho1_nz, Omega_nz = _select_transitions(
        mu_det_e, rho1, Omega,
        use_sparse_transitions=use_sparse_transitions,
        mu_nz_threshold=mu_nz_threshold,
    )

    # Compute response with chunking
    resp = _compute_response(
        mu_det_nz, rho1_nz, Omega_nz, omega, hbar=hbar, omega_chunk_size=omega_chunk_size
    )

    # Convert to absorbance [mOD]
    A_mOD = _resp_to_absorbance(
        resp, omega, number_density, path_length,
        local_field=local_field, eps0=eps0, c=c,
    )

    out: dict = {
        "nu_tilde_cm": nu_tilde_cm,
        "omega": omega,
        "A_mOD": A_mOD,
    }
    if return_resp:
        out["resp"] = resp
    return out


