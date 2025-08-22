#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Absorbance spectrum calculator (mOD) from density matrix, dipole operators and Hamiltonian.

Key features:
- Works with external rho, mu_int, mu_det, H0 and a wavenumber grid
- Diagonalizes H0 if needed and transforms operators to the energy eigenbasis
- Supports sparse transition selection and omega chunking to save memory
- Optional Lorentz local-field correction

All inputs are SI unless otherwise noted.

Functions:
- compute_absorbance_spectrum(...): main API
"""

from __future__ import annotations

from typing import Tuple
import numpy as np
from numpy.typing import NDArray


# --- Physical constants (SI) ---
HBAR: float = 1.054_571_817e-34      # [J·s]
EPS0: float = 8.854_187_8128e-12     # [F/m]
C: float = 299_792_458.0             # [m/s]
LN10: float = float(np.log(10.0))


def wavenumber_cm_to_omega(nu_tilde_cm: NDArray[np.floating]) -> NDArray[np.floating]:
    """Convert wavenumber [cm^-1] to angular frequency [rad/s]:
        omega = 2π c (100 * nu_tilde_cm)
    """
    return 2.0 * np.pi * C * (100.0 * np.asarray(nu_tilde_cm, dtype=float))


def _to_energy_eigenbasis(
    H0: NDArray[np.complexfloating],
    rho: NDArray[np.complexfloating],
    mu_int: NDArray[np.complexfloating],
    mu_det: NDArray[np.complexfloating] | None,
    *,
    diag_atol: float = 0.0,
) -> Tuple[NDArray[np.floating], NDArray[np.complexfloating], NDArray[np.complexfloating], NDArray[np.complexfloating]]:
    """If H0 is (approximately) diagonal within tolerance, skip eigendecomposition.
    Otherwise, diagonalize and transform rho, mu_int, mu_det to the energy eigenbasis.

    Returns (E, rho_e, mu_int_e, mu_det_e), where E are eigenvalues [J].
    """
    H0 = np.asarray(H0, dtype=complex)
    rho = np.asarray(rho, dtype=complex)
    mu_int = np.asarray(mu_int, dtype=complex)
    if mu_det is not None:
        mu_det_arr = np.asarray(mu_det, dtype=complex)
    else:
        mu_det_arr = None

    if diag_atol > 0.0:
        # If already diagonal within tolerance, skip eigh
        is_diag = np.allclose(H0, np.diag(np.diag(H0)), atol=diag_atol)
    else:
        is_diag = False

    if is_diag:
        E = np.asarray(np.diag(H0).real)
        rho_e = rho
        mu_int_e = mu_int
        mu_det_e = mu_int if mu_det_arr is None else mu_det_arr
        return E, rho_e, mu_int_e, mu_det_e

    # Hermitian eigendecomposition
    E, U = np.linalg.eigh(H0)
    Udag = U.conj().T
    rho_e = Udag @ rho @ U
    mu_int_e = Udag @ mu_int @ U
    mu_det_e = mu_int_e if mu_det_arr is None else (Udag @ mu_det_arr @ U)
    return E.real, rho_e, mu_int_e, mu_det_e


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
    if T2 <= 0.0:
        raise ValueError("T2 must be positive (seconds)")
    if path_length < 0.0:
        raise ValueError("path_length must be non-negative")
    if number_density < 0.0:
        raise ValueError("number_density must be non-negative")

    rho = np.asarray(rho, dtype=complex)
    mu_int = np.asarray(mu_int, dtype=complex)
    H0 = np.asarray(H0, dtype=complex)
    nu_tilde_cm = np.asarray(nu_tilde_cm, dtype=float)

    # Transform to energy eigenbasis (or reuse if diagonal)
    E, rho_e, mu_int_e, mu_det_e = _to_energy_eigenbasis(
        H0, rho, mu_int, mu_det, diag_atol=diag_atol
    )

    # Angular frequency grid
    omega = wavenumber_cm_to_omega(nu_tilde_cm)  # (W,)
    W = int(omega.size)

    # Complex Bohr frequencies Ω_ij with dephasing (broadcast (N,N))
    E_over_hbar = (E / hbar).astype(complex)  # [rad/s]
    Omega = (E_over_hbar[:, None] - E_over_hbar[None, :]) - 1j * (1.0 / float(T2))

    # First-order density after probe interaction: commutator [μ_int, ρ]
    rho1 = mu_int_e @ rho_e - rho_e @ mu_int_e

    # Select transitions
    if use_sparse_transitions:
        # Use detection operator to select contributing transitions
        if mu_nz_threshold > 0.0:
            mask = np.abs(mu_det_e) > mu_nz_threshold
        else:
            mask = mu_det_e != 0
        i_idx, j_idx = np.nonzero(mask)
        # If nothing selected, fall back to all
        if i_idx.size == 0:
            i_idx, j_idx = np.nonzero(np.ones_like(mu_det_e, dtype=bool))
        mu_det_nz = mu_det_e[i_idx, j_idx]
        rho1_nz = rho1[i_idx, j_idx]
        Omega_nz = Omega[i_idx, j_idx]
        M = int(mu_det_nz.size)
    else:
        # Dense path: flatten all i,j
        N = int(H0.shape[0])
        i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        i_idx = i_grid.ravel()
        j_idx = j_grid.ravel()
        mu_det_nz = mu_det_e.ravel()
        rho1_nz = rho1.ravel()
        Omega_nz = Omega.ravel()
        M = mu_det_nz.size

    # Compute response with omega chunking to control memory
    resp = np.zeros(W, dtype=complex)
    if omega_chunk_size is None or omega_chunk_size <= 0:
        omega_chunk_size = W  # single chunk

    for k0 in range(0, W, omega_chunk_size):
        wk = omega[k0 : k0 + omega_chunk_size]  # (Wc,)
        # Kernel K_m(ω) for each selected transition m and ω in the chunk
        # Shape: (M, Wc)
        K = -1.0 / (1j * (wk[None, :] + Omega_nz[:, None]))
        # resp_chunk(ω) = (i/ħ) Σ_m mu_det(m) * rho1(m) * K_m(ω)
        resp_chunk = (1j / hbar) * np.einsum(
            "m,m,mw->w", mu_det_nz, rho1_nz, K, optimize=True
        )
        resp[k0 : k0 + omega_chunk_size] += resp_chunk

    # Convert to absorbance [mOD]
    if local_field:
        # Lorentz local-field form
        x = (number_density / (3.0 * eps0)) * resp
        imag_sqrt = np.imag(np.sqrt(1.0 + x))
        A_mOD = (2.0 * path_length * omega / c) * imag_sqrt * (1000.0 / LN10)
    else:
        # Thin-gas approximation: Im{sqrt(1+x)} ≈ Im{x}/2
        A_mOD = (
            (path_length * omega / c)
            * (number_density / (3.0 * eps0))
            * np.imag(resp)
            * (1000.0 / LN10)
        )

    out: dict = {
        "nu_tilde_cm": nu_tilde_cm,
        "omega": omega,
        "A_mOD": A_mOD,
    }
    if return_resp:
        out["resp"] = resp
    return out


if __name__ == "__main__":
    # Minimal demo with a toy 3-level system
    N = 3
    E_levels_cm = np.array([0.0, 1000.0, 2000.0])  # [cm^-1]
    # Convert to Joule via E = h c ν~, using ħ and ω equivalence:
    # ω = 2π c 100 ν~, E = ħ ω
    omega_levels = wavenumber_cm_to_omega(E_levels_cm)
    E_levels_J = HBAR * omega_levels
    H0 = np.diag(E_levels_J.astype(complex))

    rho = np.diag(np.array([1.0, 0.3, 0.05])).astype(complex)
    rho /= rho.trace().real

    mu = np.zeros((N, N), dtype=complex)
    mu[0, 1] = mu[1, 0] = 1.0e-30
    mu[1, 2] = mu[2, 1] = 0.8e-30

    nu_grid = np.linspace(800.0, 2200.0, 4001)

    res = compute_absorbance_spectrum(
        rho=rho,
        mu_int=mu,
        mu_det=mu,
        H0=H0,
        nu_tilde_cm=nu_grid,
        T2=500e-15,
        number_density=1.0e22,
        path_length=0.1,
        use_sparse_transitions=True,
        mu_nz_threshold=0.0,
        omega_chunk_size=2000,
        local_field=True,
        return_resp=False,
        diag_atol=0.0,
    )

    print("Demo peak A [mOD] =", float(np.max(res["A_mOD"])))


