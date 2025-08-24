#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spectroscopy utilities (shared helpers for spectrum calculations).

Note: Internal module – no public API guarantees.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np
from numpy.typing import NDArray


def wavenumber_cm_to_omega(nu_tilde_cm: NDArray[np.floating], *, c: float = 299_792_458.0) -> NDArray[np.floating]:
    """Convert wavenumber [cm^-1] to angular frequency [rad/s].

    omega = 2π c (100 * nu~)
    """
    return 2.0 * np.pi * c * (100.0 * np.asarray(nu_tilde_cm, dtype=float))


def to_energy_eigenbasis(
    H0: NDArray[np.complexfloating],
    rho: NDArray[np.complexfloating],
    mu_int: NDArray[np.complexfloating],
    mu_det: NDArray[np.complexfloating] | None,
    *,
    diag_atol: float = 0.0,
) -> Tuple[NDArray[np.floating], NDArray[np.complexfloating], NDArray[np.complexfloating], NDArray[np.complexfloating]]:
    """Diagonalize H0 if needed and transform operators to the energy eigenbasis."""
    H0 = np.asarray(H0, dtype=complex)
    rho = np.asarray(rho, dtype=complex)
    mu_int = np.asarray(mu_int, dtype=complex)
    mu_det_arr = None if mu_det is None else np.asarray(mu_det, dtype=complex)

    if diag_atol > 0.0 and np.allclose(H0, np.diag(np.diag(H0)), atol=diag_atol):
        E = np.asarray(np.diag(H0).real)
        rho_e = rho
        mu_int_e = mu_int
        mu_det_e = mu_int if mu_det_arr is None else mu_det_arr
        return E, rho_e, mu_int_e, mu_det_e

    E, U = np.linalg.eigh(H0)
    Udag = U.conj().T
    rho_e = Udag @ rho @ U
    mu_int_e = Udag @ mu_int @ U
    mu_det_e = mu_int_e if mu_det_arr is None else (Udag @ mu_det_arr @ U)
    E = E.astype(float, copy=False)
    return E, rho_e, mu_int_e, mu_det_e


def select_transitions(
    mu_det_e: NDArray[np.complexfloating],
    rho1: NDArray[np.complexfloating],
    Omega: NDArray[np.complexfloating],
    *,
    use_sparse_transitions: bool,
    mu_nz_threshold: float,
) -> tuple[NDArray[np.complexfloating], NDArray[np.complexfloating], NDArray[np.complexfloating]]:
    """Select contributing transitions according to sparsity policy."""
    if use_sparse_transitions:
        mask = (np.abs(mu_det_e) > mu_nz_threshold) if mu_nz_threshold > 0.0 else (mu_det_e != 0)
        i_idx, j_idx = np.nonzero(mask)
        if i_idx.size == 0:
            i_idx, j_idx = np.nonzero(np.ones_like(mu_det_e, dtype=bool))
        return mu_det_e[i_idx, j_idx], rho1[i_idx, j_idx], Omega[i_idx, j_idx]
    return mu_det_e.ravel(), rho1.ravel(), Omega.ravel()


def compute_response(
    mu_det_nz: NDArray[np.complexfloating],
    rho1_nz: NDArray[np.complexfloating],
    Omega_nz: NDArray[np.complexfloating],
    omega: NDArray[np.floating],
    *,
    hbar: float,
    omega_chunk_size: int | None,
) -> NDArray[np.complexfloating]:
    """Compute susceptibility-like response on omega grid with chunking."""
    W = int(omega.size)
    resp = np.zeros(W, dtype=complex)
    if omega_chunk_size is None or omega_chunk_size <= 0:
        omega_chunk_size = W
    for k0 in range(0, W, omega_chunk_size):
        wk = omega[k0 : k0 + omega_chunk_size]
        K = -1.0 / (1j * (wk[None, :] + Omega_nz[:, None]))
        resp[k0 : k0 + omega_chunk_size] += (1j / hbar) * np.einsum(
            "m,m,mw->w", mu_det_nz, rho1_nz, K, optimize=True
        )
    return resp


