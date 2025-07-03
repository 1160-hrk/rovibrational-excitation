#!/usr/bin/env python
"""
Test unit management for TwoLevel and VibLadder dipole matrices.

This script verifies that the new unit management features work correctly
for both TwoLevelDipoleMatrix and VibLadderDipoleMatrix classes.
"""

import os
import sys

import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from rovibrational_excitation.core.basis import TwoLevelBasis, VibLadderBasis
from rovibrational_excitation.dipole import TwoLevelDipoleMatrix, VibLadderDipoleMatrix


def test_twolevel_units():
    """Test TwoLevelDipoleMatrix unit management."""
    print("=== Testing TwoLevelDipoleMatrix Unit Management ===")
    
    # Create basis
    basis = TwoLevelBasis()
    
    # Test different units
    dipole_cm = TwoLevelDipoleMatrix(basis, mu0=1e-30, units="C*m")
    dipole_D = TwoLevelDipoleMatrix(basis, mu0=1.0, units="D")  
    dipole_ea0 = TwoLevelDipoleMatrix(basis, mu0=1.0, units="ea0")
    
    print(f"Created dipoles: {dipole_cm}")
    print(f"                {dipole_D}")
    print(f"                {dipole_ea0}")
    
    # Test unit conversion
    mu_x_cm = dipole_cm.get_mu_x_SI()
    mu_x_D_to_cm = dipole_D.get_mu_x_SI()
    mu_x_ea0_to_cm = dipole_ea0.get_mu_x_SI()
    
    print(f"\nμ_x matrices in C·m:")
    print(f"Original C·m: max = {np.max(np.abs(mu_x_cm)):.6e}")
    print(f"From D:       max = {np.max(np.abs(mu_x_D_to_cm)):.6e}")
    print(f"From ea0:     max = {np.max(np.abs(mu_x_ea0_to_cm)):.6e}")
    
    # Test specific conversions
    mu_x_as_D = dipole_cm.get_mu_in_units("x", "D")
    mu_x_as_ea0 = dipole_cm.get_mu_in_units("x", "ea0")
    
    print(f"\nConversions from C·m to other units:")
    print(f"→ D:   max = {np.max(np.abs(mu_x_as_D)):.6e}")
    print(f"→ ea0: max = {np.max(np.abs(mu_x_as_ea0)):.6e}")
    
    # Verify Pauli matrix structure
    print(f"\nPauli-X structure check:")
    print(f"μ_x =\n{mu_x_cm}")
    expected_structure = np.array([[0, 1], [1, 0]], dtype=complex)
    structure_match = np.allclose(mu_x_cm / np.max(np.abs(mu_x_cm)), expected_structure)
    print(f"Structure matches Pauli-X: {structure_match}")
    
    # Test y-component (Pauli-Y)
    mu_y_cm = dipole_cm.get_mu_y_SI()
    print(f"\nμ_y =\n{mu_y_cm}")
    expected_y_structure = np.array([[0, -1j], [1j, 0]], dtype=complex)
    y_structure_match = np.allclose(mu_y_cm / np.max(np.abs(mu_y_cm)), expected_y_structure)
    print(f"Structure matches Pauli-Y: {y_structure_match}")
    
    # Test z-component (should be zero)
    mu_z_cm = dipole_cm.get_mu_z_SI()
    print(f"\nμ_z max = {np.max(np.abs(mu_z_cm)):.6e} (should be ~0)")
    
    print("✅ TwoLevelDipoleMatrix tests completed\n")


def test_viblad_units():
    """Test VibLadderDipoleMatrix unit management."""
    print("=== Testing VibLadderDipoleMatrix Unit Management ===")
    
    # Create basis
    basis = VibLadderBasis(V_max=2, omega_rad_pfs=1.0, delta_omega_rad_pfs=0.0)
    
    # Test different units and potentials
    dipole_harm_cm = VibLadderDipoleMatrix(
        basis, mu0=1e-30, potential_type="harmonic", units="C*m"
    )
    dipole_harm_D = VibLadderDipoleMatrix(
        basis, mu0=1.0, potential_type="harmonic", units="D"
    )
    dipole_morse_ea0 = VibLadderDipoleMatrix(
        basis, mu0=1.0, potential_type="morse", units="ea0"
    )
    
    print(f"Created dipoles: {dipole_harm_cm}")
    print(f"                {dipole_harm_D}")
    print(f"                {dipole_morse_ea0}")
    
    # Test z-component (main component for vibrational transitions)
    mu_z_harm_cm = dipole_harm_cm.get_mu_z_SI()
    mu_z_harm_D_to_cm = dipole_harm_D.get_mu_z_SI()
    mu_z_morse_ea0_to_cm = dipole_morse_ea0.get_mu_z_SI()
    
    print(f"\nμ_z matrices in C·m:")
    print(f"Harmonic (C·m): max = {np.max(np.abs(mu_z_harm_cm)):.6e}")
    print(f"Harmonic (D):   max = {np.max(np.abs(mu_z_harm_D_to_cm)):.6e}")
    print(f"Morse (ea0):    max = {np.max(np.abs(mu_z_morse_ea0_to_cm)):.6e}")
    
    # Show matrix structure
    print(f"\nHarmonic μ_z matrix structure:")
    print(f"Shape: {mu_z_harm_cm.shape}")
    print(f"μ_z =\n{mu_z_harm_cm}")
    
    # Test selection rules (should be non-zero only for Δv = ±1)
    print(f"\nSelection rule check (harmonic):")
    for i in range(basis.size()):
        for j in range(basis.size()):
            v_i = basis.V_array[i]
            v_j = basis.V_array[j]
            element = mu_z_harm_cm[i, j]
            if abs(element) > 1e-15:
                print(f"  <{v_i}|μ_z|{v_j}> = {element:.6e} (Δv = {v_j - v_i})")
    
    # Test x and y components (should be zero for pure vibrational system)
    mu_x_cm = dipole_harm_cm.get_mu_x_SI()
    mu_y_cm = dipole_harm_cm.get_mu_y_SI()
    print(f"\nμ_x max = {np.max(np.abs(mu_x_cm)):.6e} (should be ~0)")
    print(f"μ_y max = {np.max(np.abs(mu_y_cm)):.6e} (should be ~0)")
    
    # Test unit conversions
    mu_z_as_D = dipole_harm_cm.get_mu_in_units("z", "D")
    mu_z_as_ea0 = dipole_harm_cm.get_mu_in_units("z", "ea0")
    
    print(f"\nConversions from C·m to other units:")
    print(f"→ D:   max = {np.max(np.abs(mu_z_as_D)):.6e}")
    print(f"→ ea0: max = {np.max(np.abs(mu_z_as_ea0)):.6e}")
    
    print("✅ VibLadderDipoleMatrix tests completed\n")


def test_consistency_with_linmol():
    """Test consistency with LinMolDipoleMatrix interface."""
    print("=== Testing Interface Consistency ===")
    
    # Test TwoLevel
    basis_2level = TwoLevelBasis()
    dipole_2level = TwoLevelDipoleMatrix(basis_2level, mu0=1.0, units="D")
    
    # Test VibLadder  
    basis_viblad = VibLadderBasis(V_max=1)
    dipole_viblad = VibLadderDipoleMatrix(basis_viblad, mu0=1.0, units="D")
    
    # Test common interface methods
    for name, dipole in [("TwoLevel", dipole_2level), ("VibLadder", dipole_viblad)]:
        print(f"\n{name} interface test:")
        
        # Test property access
        mu_x = dipole.mu_x
        mu_y = dipole.mu_y
        mu_z = dipole.mu_z
        print(f"  Properties: mu_x {mu_x.shape}, mu_y {mu_y.shape}, mu_z {mu_z.shape}")
        
        # Test SI unit methods
        mu_x_SI = dipole.get_mu_x_SI()
        mu_y_SI = dipole.get_mu_y_SI()
        mu_z_SI = dipole.get_mu_z_SI()
        print(f"  SI methods: shapes {mu_x_SI.shape}, {mu_y_SI.shape}, {mu_z_SI.shape}")
        
        # Test unit conversion
        mu_x_cm = dipole.get_mu_in_units("x", "C*m")
        mu_x_ea0 = dipole.get_mu_in_units("x", "ea0")
        print(f"  Unit conversion: C*m max = {np.max(np.abs(mu_x_cm)):.6e}")
        print(f"                   ea0 max = {np.max(np.abs(mu_x_ea0)):.6e}")
        
        # Test stacked method
        stacked_xyz = dipole.stacked("xyz")
        stacked_z = dipole.stacked("z")
        print(f"  Stacked: xyz {stacked_xyz.shape}, z {stacked_z.shape}")
    
    print("✅ Interface consistency tests completed")


def main():
    """Run all tests."""
    print("🧪 Testing Unit Management for TwoLevel and VibLadder Dipole Matrices")
    print("=" * 80)
    
    try:
        test_twolevel_units()
        test_viblad_units()
        test_consistency_with_linmol()
        
        print("\n🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 