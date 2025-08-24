#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spectroscopy module for rovibrational excitation calculations.

This module provides linear response theory calculations for rovibrational
spectroscopy, including absorption, PFID, and radiation spectra calculations.

The module offers both modern class-based API and legacy function-based API
for backward compatibility.
"""

# Internal constants are not re-exported; use core.units.constants in external code

from .broadening import (
    doppler,
    sinc,
    sinc_square,
    convolution_w_doppler,
    convolution_w_sinc,
    convolution_w_sinc_square
)

__all__ = [
    # Broadening functions
    'doppler',
    'sinc',
    'sinc_square',
    'convolution_w_doppler',
    'convolution_w_sinc',
    'convolution_w_sinc_square',
] 

# ------------------------------------------------------------------
# Public absorbance and broadening API
# ------------------------------------------------------------------
try:
    from .api import (
        compute_absorbance_spectrum,
        convolve_absorbance_spectrum,
        compute_absorbance_spectrum_broadened,
    )  # type: ignore
    __all__.append('compute_absorbance_spectrum')
    __all__.append('convolve_absorbance_spectrum')
    __all__.append('compute_absorbance_spectrum_broadened')
except Exception:
    # Fallback: do not break imports if api module is unavailable
    pass