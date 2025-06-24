#!/usr/bin/env python
"""
Check script for new basis classes and dipole matrices.
Interactive exploration and validation of the new quantum systems.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpy as np
from rovibrational_excitation.core.basis import LinMolBasis, TwoLevelBasis, VibLadderBasis
from rovibrational_excitation.dipole.twolevel import TwoLevelDipoleMatrix
from rovibrational_excitation.dipole.viblad import VibLadderDipoleMatrix
from rovibrational_excitation.dipole.linmol import LinMolDipoleMatrix


def test_two_level_system():
    """Test two-level system."""
    print("=== Testing Two-Level System ===")
    
    # Create basis
    basis = TwoLevelBasis()
    print(f"Basis: {basis}")
    print(f"Size: {basis.size()}")
    print(f"States: {[basis.get_state(i) for i in range(basis.size())]}")
    
    # Test Hamiltonian
    H0 = basis.generate_H0(energy_gap=2.0)
    print(f"Hamiltonian:\n{H0}")
    
    # Test dipole matrix
    dipole = TwoLevelDipoleMatrix(basis, mu0=0.5)
    print(f"Dipole matrix (x): \n{dipole.mu_x}")
    print(f"Dipole matrix (y): \n{dipole.mu_y}")
    print(f"Dipole matrix (z): \n{dipole.mu_z}")
    print()


def test_vib_ladder_system():
    """Test vibrational ladder system."""
    print("=== Testing Vibrational Ladder System ===")
    
    # Create basis
    basis = VibLadderBasis(V_max=3, omega_rad_phz=1.0, delta_omega_rad_phz=0.01)
    print(f"Basis: {basis}")
    print(f"Size: {basis.size()}")
    print(f"States: {[basis.get_state(i) for i in range(basis.size())]}")
    
    # Test Hamiltonian
    H0 = basis.generate_H0()
    print(f"Hamiltonian diagonal: {np.diag(H0)}")
    
    # Test dipole matrix (harmonic)
    dipole_harm = VibLadderDipoleMatrix(basis, mu0=1.0, potential_type="harmonic")
    print(f"Harmonic dipole matrix (z):\n{dipole_harm.mu_z}")
    
    # Test dipole matrix (morse)
    dipole_morse = VibLadderDipoleMatrix(basis, mu0=1.0, potential_type="morse")
    print(f"Morse dipole matrix (z):\n{dipole_morse.mu_z}")
    print()


def test_linmol_system():
    """Test linear molecule system (backward compatibility)."""
    print("=== Testing LinMol System (New Interface) ===")
    
    # Create basis
    basis = LinMolBasis(V_max=2, J_max=2)
    print(f"Basis: {basis}")
    print(f"Size: {basis.size()}")
    
    # Test new generate_H0 method
    H0 = basis.generate_H0(omega_rad_phz=1.0, B_rad_phz=0.1)
    print(f"Hamiltonian diagonal: {np.diag(H0)}")
    
    # Test dipole matrix
    dipole = LinMolDipoleMatrix(basis, mu0=0.3, potential_type="harmonic", dense=True)
    print(f"LinMol dipole matrix shape: {dipole.mu_x.shape}")
    print(f"LinMol dipole matrix nnz: {np.count_nonzero(dipole.mu_x)}")
    print()


def test_backward_compatibility():
    """Test backward compatibility with old imports."""
    print("=== Testing Backward Compatibility ===")
    
    # This should work but show deprecation warning
    try:
        from rovibrational_excitation.core.basis import LinMolBasis as OldLinMolBasis
        from rovibrational_excitation.core.hamiltonian import generate_H0_LinMol
        
        basis = OldLinMolBasis(V_max=1, J_max=1)
        H0_old = generate_H0_LinMol(basis)
        H0_new = basis.generate_H0()
        
        print(f"Old method result: {np.diag(H0_old)}")
        print(f"New method result: {np.diag(H0_new)}")
        print(f"Results match: {np.allclose(H0_old, H0_new)}")
        
    except Exception as e:
        print(f"Backward compatibility test failed: {e}")
    
    print()


if __name__ == "__main__":
    print("Testing new basis classes and dipole matrices...")
    print()
    
    test_two_level_system()
    test_vib_ladder_system()
    test_linmol_system()
    test_backward_compatibility()
    
    print("All tests completed!") 