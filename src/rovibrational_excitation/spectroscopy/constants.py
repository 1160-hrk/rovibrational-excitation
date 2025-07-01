#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physical constants for spectroscopy calculations.

This module contains fundamental physical constants used in 
rovibrational spectroscopy calculations.

All values are based on the 2018 CODATA recommended values 
(Committee on Data for Science and Technology).

References:
- CODATA 2018: https://physics.nist.gov/cuu/Constants/
- NIST Physical Constants: https://www.nist.gov/pml/fundamental-physical-constants
"""

import numpy as np

# ========================================================================
# FUNDAMENTAL PHYSICAL CONSTANTS (2018 CODATA)
# ========================================================================

# Planck constant and related
h_planck = 6.62607015e-34        # Planck constant [J⋅s] (exact, defining constant)
h_dirac = 1.054571817e-34        # Reduced Planck constant ℏ = h/(2π) [J⋅s]

# Elementary charge
ee = 1.602176634e-19             # Elementary charge [C] (exact, defining constant)
e = ee                           # Alias for elementary charge

# Speed of light
c = 299792458                    # Speed of light in vacuum [m/s] (exact, defining constant)

# Vacuum properties
eps_0 = 8.8541878128e-12         # Vacuum permittivity ε₀ [F/m]
eps = eps_0                      # Alias for vacuum permittivity
mu_0 = 1.25663706212e-6          # Vacuum permeability μ₀ [H/m]

# Thermal physics
kb = 1.380649e-23                # Boltzmann constant k_B [J/K] (exact, defining constant)
k_B = kb                         # Alias for Boltzmann constant

# Avogadro constant
N_A = 6.02214076e23              # Avogadro constant [mol⁻¹] (exact, defining constant)

# Mass and energy units
u = 1.66053906660e-27            # Atomic mass unit [kg]
m_e = 9.1093837015e-31           # Electron mass [kg]
m_p = 1.67262192369e-27          # Proton mass [kg]

# Fine structure constant
alpha = 7.2973525693e-3          # Fine structure constant (dimensionless)

# Other useful constants
R = 8.314462618                  # Gas constant [J/(mol⋅K)] (exact)

# ========================================================================
# UNIT CONVERSION FACTORS
# ========================================================================

# Energy conversions
eV_to_J = ee                     # 1 eV = ee J
hartree_to_J = 4.3597447222071e-18  # 1 hartree = 4.359744... × 10⁻¹⁸ J
cm_inv_to_J = h_planck * c * 1e2 # 1 cm⁻¹ = hc × 100 J
cm_inv_to_Hz = c * 1e2           # 1 cm⁻¹ = c × 100 Hz

# Length conversions
bohr_to_m = 5.29177210903e-11    # Bohr radius [m]
angstrom_to_m = 1e-10            # 1 Ångström = 10⁻¹⁰ m

# Time conversions
fs_to_s = 1e-15                  # 1 femtosecond = 10⁻¹⁵ s
ps_to_s = 1e-12                  # 1 picosecond = 10⁻¹² s

# ========================================================================
# MOLECULAR CONSTANTS
# ========================================================================

# CO₂ molecule (carbon dioxide)
# Molecular weight: 12.011 + 2×15.999 = 44.009 g/mol (IUPAC 2016)
m_CO2 = 44.009e-3 / N_A          # Mass of CO₂ molecule [kg]

# H₂O molecule (water)  
# Molecular weight: 2×1.008 + 15.999 = 18.015 g/mol
m_H2O = 18.015e-3 / N_A          # Mass of H₂O molecule [kg]

# N₂ molecule (nitrogen)
# Molecular weight: 2×14.007 = 28.014 g/mol  
m_N2 = 28.014e-3 / N_A           # Mass of N₂ molecule [kg]

# O₂ molecule (oxygen)
# Molecular weight: 2×15.999 = 31.998 g/mol
m_O2 = 31.998e-3 / N_A           # Mass of O₂ molecule [kg]

# ========================================================================
# SPECTROSCOPY-SPECIFIC CONSTANTS
# ========================================================================

# Typical spectroscopic parameters for CO₂
# These are approximate values for the ν₃ antisymmetric stretch mode

# Vibrational frequency (ν₃ mode of CO₂)
# Fundamental frequency: ~2349.1 cm⁻¹
nu3_CO2_cm = 2349.1              # CO₂ ν₃ frequency [cm⁻¹]
omega3_CO2_rad_s = nu3_CO2_cm * cm_inv_to_Hz * 2 * np.pi  # [rad/s]
omega3_CO2_rad_fs = omega3_CO2_rad_s * fs_to_s    # [rad/fs]

# Rotational constant for CO₂
# B₀ ≈ 0.39021 cm⁻¹
B0_CO2_cm = 0.39021              # CO₂ rotational constant [cm⁻¹]
B0_CO2_rad_s = B0_CO2_cm * cm_inv_to_Hz * 2 * np.pi  # [rad/s]
B0_CO2_rad_fs = B0_CO2_rad_s * fs_to_s        # [rad/fs]

# Anharmonicity constant for CO₂ ν₃ mode
# x₃₃ ≈ 12.3 cm⁻¹
x33_CO2_cm = 12.3                # CO₂ anharmonicity [cm⁻¹]
x33_CO2_rad_fs = x33_CO2_cm * cm_inv_to_Hz * 2 * np.pi * fs_to_s  # [rad/fs]

# Vibration-rotation coupling
# α₃ ≈ 0.0032 cm⁻¹ (typical value)
alpha3_CO2_cm = 0.0032           # CO₂ α constant [cm⁻¹]
alpha3_CO2_rad_fs = alpha3_CO2_cm * cm_inv_to_Hz * 2 * np.pi * fs_to_s  # [rad/fs]

# ========================================================================
# CONVENIENCE DICTIONARIES
# ========================================================================

# Physical constants dictionary
PHYSICAL_CONSTANTS = {
    'h': h_planck,
    'hbar': h_dirac,
    'c': c,
    'e': ee,
    'eps0': eps_0,
    'mu0': mu_0,
    'kB': kb,
    'NA': N_A,
    'u': u,
    'me': m_e,
    'mp': m_p,
    'alpha': alpha,
    'R': R
}

# Unit conversion factors
UNIT_CONVERSIONS = {
    'eV_to_J': eV_to_J,
    'hartree_to_J': hartree_to_J,
    'cm_inv_to_J': cm_inv_to_J,
    'cm_inv_to_Hz': cm_inv_to_Hz,
    'bohr_to_m': bohr_to_m,
    'angstrom_to_m': angstrom_to_m,
    'fs_to_s': fs_to_s,
    'ps_to_s': ps_to_s
}

# Molecular masses
MOLECULAR_MASSES = {
    'CO2': m_CO2,
    'H2O': m_H2O,
    'N2': m_N2,
    'O2': m_O2
}

# CO₂ spectroscopic parameters
CO2_SPECTROSCOPY = {
    'nu3_cm': nu3_CO2_cm,
    'omega3_rad_s': omega3_CO2_rad_s,
    'omega3_rad_fs': omega3_CO2_rad_fs,
    'B0_cm': B0_CO2_cm,
    'B0_rad_s': B0_CO2_rad_s,
    'B0_rad_fs': B0_CO2_rad_fs,
    'x33_cm': x33_CO2_cm,
    'x33_rad_fs': x33_CO2_rad_fs,
    'alpha3_cm': alpha3_CO2_cm,
    'alpha3_rad_fs': alpha3_CO2_rad_fs
}

# ========================================================================
# LEGACY ALIASES (for backward compatibility)
# ========================================================================

# These maintain compatibility with existing code
h_dirac = h_dirac                # Already defined above
ee = ee                          # Already defined above  
c = c                            # Already defined above
eps = eps_0                      # Already defined above
kb = kb                          # Already defined above
m_CO2 = m_CO2                    # Already defined above

# ========================================================================
# VALIDATION FUNCTIONS
# ========================================================================

def validate_constants():
    """
    定数の一貫性をチェックする関数。
    
    基本的な物理関係式が満たされているかを確認します。
    """
    
    print("Physical Constants Validation")
    print("=" * 40)
    
    # Check: c = 1/√(ε₀μ₀)
    c_calc = 1 / np.sqrt(eps_0 * mu_0)
    print(f"Speed of light check:")
    print(f"  Defined: {c:.0f} m/s")
    print(f"  Calculated from ε₀μ₀: {c_calc:.0f} m/s")
    print(f"  Relative error: {abs(c - c_calc)/c * 100:.2e}%")
    print()
    
    # Check: ℏ = h/(2π)
    hbar_calc = h_planck / (2 * np.pi)
    print(f"Reduced Planck constant check:")
    print(f"  Defined: {h_dirac:.10e} J⋅s")
    print(f"  Calculated from h: {hbar_calc:.10e} J⋅s")
    print(f"  Relative error: {abs(h_dirac - hbar_calc)/h_dirac * 100:.2e}%")
    print()
    
    # Check: R = kB × NA
    R_calc = kb * N_A
    print(f"Gas constant check:")
    print(f"  Defined: {R:.9f} J/(mol⋅K)")
    print(f"  Calculated from kB×NA: {R_calc:.9f} J/(mol⋅K)")
    print(f"  Relative error: {abs(R - R_calc)/R * 100:.2e}%")
    print()

if __name__ == "__main__":
    validate_constants() 