#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spectroscopy module for rovibrational excitation calculations.

This module provides linear response theory calculations for rovibrational
spectroscopy, including absorption, PFID, and radiation spectra calculations.

The module offers both modern class-based API and legacy function-based API
for backward compatibility.
"""

from .constants import (
    h_dirac, ee, c, eps, kb, m_CO2
)

from .linear_response import (
    LinearResponseCalculator,
    SpectroscopyParameters,
    MolecularParameters,
    prepare_variables,
    get_calculator,
    get_basis,
    get_dipole
)

from .spectra import (
    # Modern API
    calculate_absorption_spectrum,
    calculate_pfid_spectrum,
    calculate_emission_spectrum,
    calculate_absorption_from_hamiltonian,
    
    # Legacy API (for backward compatibility)
    absorbance_spectrum,
    absorbance_spectrum_for_loop,
    absorbance_spectrum_w_doppler_broadening,
    PFID_spectrum_for_loop,
    radiation_spectrum_for_loop,
    absorbance_spectrum_from_rho_and_mu
)

from .broadening import (
    doppler,
    sinc,
    sinc_square,
    convolution_w_doppler,
    convolution_w_sinc,
    convolution_w_sinc_square
)

__all__ = [
    # Constants
    'h_dirac', 'ee', 'c', 'eps', 'kb', 'm_CO2',
    
    # Modern API classes and parameters
    'LinearResponseCalculator',
    'SpectroscopyParameters',
    'MolecularParameters',
    
    # Setup and utilities
    'prepare_variables', 'get_calculator', 'get_basis', 'get_dipole',
    
    # Modern spectrum calculation functions
    'calculate_absorption_spectrum',
    'calculate_pfid_spectrum',
    'calculate_emission_spectrum',
    'calculate_absorption_from_hamiltonian',
    
    # Legacy spectrum calculation functions (backward compatibility)
    'absorbance_spectrum',
    'absorbance_spectrum_for_loop',
    'absorbance_spectrum_w_doppler_broadening',
    'PFID_spectrum_for_loop',
    'radiation_spectrum_for_loop',
    'absorbance_spectrum_from_rho_and_mu',
    
    # Broadening functions
    'doppler',
    'sinc',
    'sinc_square',
    'convolution_w_doppler',
    'convolution_w_sinc',
    'convolution_w_sinc_square',
] 