#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Official spectroscopy API entrypoints (trusted path).

This module delegates the absorbance spectrum calculation to the
well-tested implementation in analysis.absorbance, providing a
stable public import location under rovibrational_excitation.spectroscopy.
"""

from __future__ import annotations

# Re-export the trusted implementation (now canonical location)
from .absorbance import compute_absorbance_spectrum as compute_absorbance_spectrum
from .broadening import (
    convolution_w_doppler,
    convolution_w_sinc,
    convolution_w_sinc_square,
)
from typing import Literal

__all__ = [
    "compute_absorbance_spectrum",
    "convolve_absorbance_spectrum",
    "compute_absorbance_spectrum_broadened",
]

def convolve_absorbance_spectrum(
    nu_tilde_cm,
    A_mOD,
    *,
    method: Literal["doppler", "sinc", "sinc_square"],
    **kwargs,
):
    """Convolve absorbance A(ν~) with a wavenumber-domain broadening kernel.
    Always returns the spectrum on the original input grid (same length).

    Parameters (common)
    -------------------
    nu_tilde_cm : array_like
        Wavenumber grid [cm^-1]
    A_mOD : array_like
        Absorbance spectrum on the same grid [mOD]
    method : {"doppler", "sinc", "sinc_square"}
        Broadening kernel to use.

    Method-specific kwargs
    ----------------------
    doppler:
        k0: float       – center wavenumber [cm^-1]
        temp_K: float   – temperature [K]
        mass_kg: float  – molecular mass [kg]
    sinc:
        dk: float       – resolution width [cm^-1]
    sinc_square:
        dk: float       – resolution width [cm^-1]

    Returns
    -------
    (nu_tilde_cm, A_conv)
        Grid (passthrough) and convolved absorbance [mOD].
    """
    import numpy as np

    x = np.asarray(nu_tilde_cm)
    y = np.asarray(A_mOD)
    if method == "doppler":
        k0 = kwargs.get("k0")
        temp_K = kwargs.get("temp_K")
        mass_kg = kwargs.get("mass_kg")
        if k0 is None or temp_K is None or mass_kg is None:
            raise ValueError("doppler requires k0, temp_K, mass_kg")
        x_out, y_out = convolution_w_doppler(x, y, k0, temp_K, mass_kg)
        if (len(x_out) != len(x)) or (not np.array_equal(x_out, x)):
            y_out = np.interp(x, x_out, y_out)
        return x, y_out
    if method == "sinc":
        dk = kwargs.get("dk")
        if dk is None:
            raise ValueError("sinc requires dk")
        x_out, y_out = convolution_w_sinc(x, y, dk)
        if (len(x_out) != len(x)) or (not np.array_equal(x_out, x)):
            y_out = np.interp(x, x_out, y_out)
        return x, y_out
    if method == "sinc_square":
        dk = kwargs.get("dk")
        if dk is None:
            raise ValueError("sinc_square requires dk")
        x_out, y_out = convolution_w_sinc_square(x, y, dk)
        if (len(x_out) != len(x)) or (not np.array_equal(x_out, x)):
            y_out = np.interp(x, x_out, y_out)
        return x, y_out
    raise ValueError(f"Unknown method: {method}")


def compute_absorbance_spectrum_broadened(
    *,
    rho,
    mu_int,
    H0,
    nu_tilde_cm,
    T2,
    number_density,
    path_length,
    mu_det=None,
    local_field: bool = True,
    return_resp: bool = False,
    use_sparse_transitions: bool = True,
    mu_nz_threshold: float = 0.0,
    omega_chunk_size: int | None = None,
    eps0=None,
    c=None,
    hbar=None,
    diag_atol: float = 0.0,
    # broadening
    broadening_method: Literal["doppler", "sinc", "sinc_square"] | None = None,
    **broadening_kwargs,
):
    """Compute absorbance and (optionally) convolve with a broadening kernel.

    Returns a dict like compute_absorbance_spectrum; when broadening is
    requested, the field "A_mOD" is the broadened result and the pre-broadened
    curve is returned as "A_mOD_raw".
    """
    res = compute_absorbance_spectrum(
        rho=rho,
        mu_int=mu_int,
        mu_det=mu_det,
        H0=H0,
        nu_tilde_cm=nu_tilde_cm,
        T2=T2,
        number_density=number_density,
        path_length=path_length,
        local_field=local_field,
        return_resp=return_resp,
        use_sparse_transitions=use_sparse_transitions,
        mu_nz_threshold=mu_nz_threshold,
        omega_chunk_size=omega_chunk_size,
        eps0=eps0 if eps0 is not None else None,
        c=c if c is not None else None,
        hbar=hbar if hbar is not None else None,
        diag_atol=diag_atol,
    )

    if broadening_method is None:
        return res

    x_out, y_out = convolve_absorbance_spectrum(
        res["nu_tilde_cm"], res["A_mOD"], method=broadening_method, **broadening_kwargs
    )
    res_out = dict(res)
    res_out["A_mOD_raw"] = res["A_mOD"]
    res_out["nu_tilde_cm"] = x_out
    res_out["A_mOD"] = y_out
    return res_out