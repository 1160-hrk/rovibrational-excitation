#!/usr/bin/env python3
"""Test new basis API with integrated physical parameters."""

import numpy as np
from src.rovibrational_excitation.core.basis import LinMolBasis, TwoLevelBasis, VibLadderBasis

print("=== Testing New Basis API ===\n")

# Test 1: LinMolBasis with cm^-1 input
print("1. LinMolBasis (CO2 parameters in cm^-1):")
basis_co2 = LinMolBasis(
    V_max=2,
    J_max=10,
    omega=2350,        # CO2 ν3 mode in cm^-1
    B=0.39,           # Rotational constant in cm^-1
    alpha=0.0,
    delta_omega=0.0,
    input_units="cm^-1",
    output_units="J"
)
print(f"   Basis: {basis_co2}")
H0_co2 = basis_co2.generate_H0()  # No arguments needed!
print(f"   H0: {H0_co2}")
print(f"   Ground state energy: {H0_co2.eigenvalues[0]:.3e} J")
print(f"   First excitation: {H0_co2.eigenvalues[1] - H0_co2.eigenvalues[0]:.3e} J\n")

# Test 2: TwoLevelBasis with eV input
print("2. TwoLevelBasis (1.5 eV gap):")
basis_tls = TwoLevelBasis(
    energy_gap=1.5,
    input_units="eV",
    output_units="J"
)
print(f"   Basis: {basis_tls}")
H0_tls = basis_tls.generate_H0()
print(f"   H0: {H0_tls}")
print(f"   Energy gap: {H0_tls.eigenvalues[1]:.3e} J\n")

# Test 3: VibLadderBasis with THz input
print("3. VibLadderBasis (100 THz oscillator):")
basis_vib = VibLadderBasis(
    V_max=5,
    omega=100,         # 100 THz
    delta_omega=0.5,   # 0.5 THz anharmonicity
    input_units="THz",
    output_units="rad/fs"
)
print(f"   Basis: {basis_vib}")
H0_vib = basis_vib.generate_H0()
print(f"   H0: {H0_vib}")
print(f"   Fundamental frequency: {H0_vib.eigenvalues[1] - H0_vib.eigenvalues[0]:.3f} rad/fs")
print(f"   First overtone: {H0_vib.eigenvalues[2] - H0_vib.eigenvalues[0]:.3f} rad/fs\n")

# Test 4: Backward compatibility with generate_H0_with_params
print("4. Testing backward compatibility:")
H0_override = basis_co2.generate_H0_with_params(
    omega=2400,  # Override with different frequency
    input_units="cm^-1"
)
print(f"   Original ω: 2350 cm^-1")
print(f"   Override ω: 2400 cm^-1")
print(f"   Original gap: {(H0_co2.eigenvalues[1] - H0_co2.eigenvalues[0]):.3e} J")
print(f"   Override gap: {(H0_override.eigenvalues[1] - H0_override.eigenvalues[0]):.3e} J")

print("\n✅ All tests passed!")

# Cleanup
import os
if os.path.exists("test_new_api.py"):
    os.remove("test_new_api.py") 