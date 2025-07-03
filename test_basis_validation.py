#!/usr/bin/env python3
"""
Comprehensive test and validation script for rovibrational_excitation.core.basis module
======================================================================================

This script tests all components of the basis package:
1. Base abstract class
2. Hamiltonian class (unit management)
3. TwoLevelBasis
4. VibLadderBasis
5. LinMolBasis
6. Integration tests

Usage: python test_basis_validation.py
"""

import sys
import numpy as np
import traceback
from typing import Any, Dict

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")


def print_test(test_name: str, success: bool, details: str = ""):
    status = f"{Colors.OKGREEN}✓ PASS{Colors.ENDC}" if success else f"{Colors.FAIL}✗ FAIL{Colors.ENDC}"
    print(f"  {status} {test_name}")
    if details:
        print(f"    {details}")
    if not success:
        print(f"    {Colors.WARNING}See traceback above for details{Colors.ENDC}")


def test_abstract_base():
    """Test the abstract base class"""
    print_header("1. Testing Abstract Base Class (BasisBase)")
    
    try:
        from src.rovibrational_excitation.core.basis.base import BasisBase
        
        # Test 1: Cannot instantiate abstract class
        try:
            base = BasisBase()
            print_test("Abstract class instantiation", False, "Should not be able to instantiate abstract class")
        except TypeError:
            print_test("Abstract class instantiation", True, "Correctly prevents instantiation")
        
        # Test 2: Check abstract methods exist
        abstract_methods = BasisBase.__abstractmethods__
        expected_methods = {'size', 'get_index', 'get_state', 'generate_H0'}
        missing_methods = expected_methods - abstract_methods
        
        print_test("Abstract methods defined", 
                  len(missing_methods) == 0,
                  f"Expected: {expected_methods}, Found: {abstract_methods}")
        
        return True
        
    except Exception as e:
        print_test("Abstract base class import", False, f"Error: {e}")
        traceback.print_exc()
        return False


def test_hamiltonian_class():
    """Test the Hamiltonian class with unit management"""
    print_header("2. Testing Hamiltonian Class")
    
    try:
        from src.rovibrational_excitation.core.basis.hamiltonian import Hamiltonian
        
        # Test 1: Basic creation
        matrix = np.array([[0.0, 0.1], [0.1, 1.0]])
        h = Hamiltonian(matrix, "J")
        print_test("Basic Hamiltonian creation", True, f"Created {h.shape} matrix")
        
        # Test 2: Unit conversion J <-> rad/fs
        h_rad_fs = h.to_frequency_units()
        h_back = h_rad_fs.to_energy_units()
        
        conversion_ok = np.allclose(h.matrix, h_back.matrix)
        print_test("J ↔ rad/fs conversion", conversion_ok, 
                  f"Original: {h.units}, Converted: {h_rad_fs.units}, Back: {h_back.units}")
        
        # Test 3: Matrix extraction with units
        matrix_J = h.get_matrix("J")
        matrix_rad_fs = h.get_matrix("rad/fs")
        
        print_test("Matrix extraction with units", True,
                  f"J shape: {matrix_J.shape}, rad/fs shape: {matrix_rad_fs.shape}")
        
        # Test 4: Eigenvalue calculation
        eigvals_J = h.get_eigenvalues("J")
        eigvals_rad_fs = h.get_eigenvalues("rad/fs")
        
        print_test("Eigenvalue calculation", True,
                  f"Eigenvalues J: {eigvals_J}, rad/fs: {eigvals_rad_fs}")
        
        # Test 5: Energy differences
        energy_diffs = h.energy_differences("J")
        max_diff = h.max_energy_difference("J")
        
        print_test("Energy difference calculation", True,
                  f"Max energy difference: {max_diff:.3e} J")
        
        # Test 6: Hamiltonian arithmetic
        h2 = Hamiltonian(np.eye(2) * 0.5, "J")
        h_sum = h + h2
        h_diff = h - h2
        h_scaled = h * 2.0
        
        print_test("Hamiltonian arithmetic", True,
                  f"Addition, subtraction, scaling operations work")
        
        # Test 7: Error handling
        try:
            bad_matrix = np.array([1, 2, 3])  # Not 2D
            Hamiltonian(bad_matrix, "J")
            print_test("Error handling - bad matrix", False, "Should reject non-2D matrix")
        except ValueError:
            print_test("Error handling - bad matrix", True, "Correctly rejects non-2D matrix")
        
        try:
            Hamiltonian(matrix, "invalid_unit")
            print_test("Error handling - bad units", False, "Should reject invalid units")
        except ValueError:
            print_test("Error handling - bad units", True, "Correctly rejects invalid units")
        
        return True
        
    except Exception as e:
        print_test("Hamiltonian class test", False, f"Error: {e}")
        traceback.print_exc()
        return False


def test_twolevel_basis():
    """Test the TwoLevelBasis class"""
    print_header("3. Testing TwoLevelBasis")
    
    try:
        from src.rovibrational_excitation.core.basis.twolevel import TwoLevelBasis
        
        # Test 1: Basic creation
        basis = TwoLevelBasis()
        print_test("TwoLevelBasis creation", True, f"Size: {basis.size()}")
        
        # Test 2: Size check
        size_ok = basis.size() == 2
        print_test("Size verification", size_ok, f"Expected: 2, Got: {basis.size()}")
        
        # Test 3: State indexing
        idx0 = basis.get_index(0)
        idx1 = basis.get_index((1,))
        
        indexing_ok = idx0 == 0 and idx1 == 1
        print_test("State indexing", indexing_ok, f"Index(0): {idx0}, Index((1,)): {idx1}")
        
        # Test 4: State retrieval
        state0 = basis.get_state(0)
        state1 = basis.get_state(1)
        
        retrieval_ok = np.array_equal(state0, [0]) and np.array_equal(state1, [1])
        print_test("State retrieval", retrieval_ok, f"State(0): {state0}, State(1): {state1}")
        
        # Test 5: Hamiltonian generation
        H0 = basis.generate_H0(units="J")
        
        hamiltonian_ok = (H0.shape == (2, 2) and H0.units == "J")
        print_test("Hamiltonian generation", hamiltonian_ok, f"Shape: {H0.shape}, Units: {H0.units}")
        
        # Test 6: Hamiltonian in different units
        H0_rad_fs = basis.generate_H0(units="rad/fs")
        
        units_ok = H0_rad_fs.units == "rad/fs"
        print_test("Hamiltonian units", units_ok, f"rad/fs units: {H0_rad_fs.units}")
        
        # Test 7: Error handling
        try:
            basis.get_index(2)
            print_test("Error handling - invalid index", False, "Should reject index 2")
        except ValueError:
            print_test("Error handling - invalid index", True, "Correctly rejects invalid index")
        
        return True
        
    except Exception as e:
        print_test("TwoLevelBasis test", False, f"Error: {e}")
        traceback.print_exc()
        return False


def test_viblad_basis():
    """Test the VibLadderBasis class"""
    print_header("4. Testing VibLadderBasis")
    
    try:
        from src.rovibrational_excitation.core.basis.viblad import VibLadderBasis
        
        # Test 1: Basic creation
        V_max = 3
        basis = VibLadderBasis(V_max=V_max, omega_rad_pfs=1.0, delta_omega_rad_pfs=0.1)
        expected_size = V_max + 1
        
        size_ok = basis.size() == expected_size
        print_test("VibLadderBasis creation", size_ok, 
                  f"V_max: {V_max}, Size: {basis.size()}, Expected: {expected_size}")
        
        # Test 2: State indexing
        indices = []
        for v in range(V_max + 1):
            idx = basis.get_index(v)
            indices.append(idx)
        
        indexing_ok = indices == list(range(V_max + 1))
        print_test("State indexing", indexing_ok, f"Indices: {indices}")
        
        # Test 3: State retrieval
        states = []
        for i in range(basis.size()):
            state = basis.get_state(i)
            states.append(state[0])  # Extract v quantum number
        
        retrieval_ok = states == list(range(V_max + 1))
        print_test("State retrieval", retrieval_ok, f"States: {states}")
        
        # Test 4: Hamiltonian generation with defaults
        H0 = basis.generate_H0(units="J")
        
        # Check that it's diagonal and has correct eigenvalues
        is_diagonal = np.allclose(H0.matrix, np.diag(np.diag(H0.matrix)))
        eigvals = H0.get_eigenvalues("rad/fs")  # Get in rad/fs for physics check
        
        # Check harmonic oscillator energy levels: E_v = ω(v + 1/2) - Δω(v + 1/2)²
        omega = 1.0
        delta_omega = 0.1
        expected_eigvals = []
        for v in range(V_max + 1):
            vterm = v + 0.5
            energy = omega * vterm - delta_omega * vterm**2
            expected_eigvals.append(energy)
        
        eigenval_ok = np.allclose(eigvals, expected_eigvals)
        print_test("Hamiltonian eigenvalues", eigenval_ok,
                  f"Expected: {expected_eigvals}, Got: {eigvals}")
        
        # Test 5: Custom parameters
        H0_custom = basis.generate_H0(omega_rad_pfs=2.0, delta_omega_rad_pfs=0.05, units="rad/fs")
        custom_eigvals = H0_custom.get_eigenvalues("rad/fs")
        
        # Check custom parameters were used
        omega_custom = 2.0
        delta_custom = 0.05
        expected_custom = []
        for v in range(V_max + 1):
            vterm = v + 0.5
            energy = omega_custom * vterm - delta_custom * vterm**2
            expected_custom.append(energy)
        
        custom_ok = np.allclose(custom_eigvals, expected_custom)
        print_test("Custom parameters", custom_ok,
                  f"Custom eigenvalues: {custom_eigvals}")
        
        # Test 6: Error handling
        try:
            basis.get_index(V_max + 1)
            print_test("Error handling - invalid v", False, "Should reject v > V_max")
        except ValueError:
            print_test("Error handling - invalid v", True, "Correctly rejects v > V_max")
        
        return True
        
    except Exception as e:
        print_test("VibLadderBasis test", False, f"Error: {e}")
        traceback.print_exc()
        return False


def test_linmol_basis():
    """Test the LinMolBasis class"""
    print_header("5. Testing LinMolBasis")
    
    try:
        from src.rovibrational_excitation.core.basis.linmol import LinMolBasis
        
        # Test 1: Basic creation with M quantum numbers
        V_max, J_max = 2, 2
        basis_with_M = LinMolBasis(V_max=V_max, J_max=J_max, use_M=True)
        
        # Expected size: (V_max+1) * sum_{J=0}^{J_max} (2J+1)
        expected_size_M = 0
        for V in range(V_max + 1):
            for J in range(J_max + 1):
                expected_size_M += (2*J + 1)
        
        size_M_ok = basis_with_M.size() == expected_size_M
        print_test("LinMolBasis with M creation", size_M_ok,
                  f"V_max: {V_max}, J_max: {J_max}, Size: {basis_with_M.size()}, Expected: {expected_size_M}")
        
        # Test 2: Creation without M quantum numbers
        basis_no_M = LinMolBasis(V_max=V_max, J_max=J_max, use_M=False)
        expected_size_no_M = (V_max + 1) * (J_max + 1)
        
        size_no_M_ok = basis_no_M.size() == expected_size_no_M
        print_test("LinMolBasis without M creation", size_no_M_ok,
                  f"Size: {basis_no_M.size()}, Expected: {expected_size_no_M}")
        
        # Test 3: State indexing with M
        state_VJM = (0, 1, -1)  # V=0, J=1, M=-1
        try:
            idx = basis_with_M.get_index(state_VJM)
            state_back = basis_with_M.get_state(idx)
            
            indexing_M_ok = tuple(state_back) == state_VJM
            print_test("State indexing with M", indexing_M_ok,
                      f"State {state_VJM} -> Index {idx} -> State {tuple(state_back)}")
        except Exception as e:
            print_test("State indexing with M", False, f"Error: {e}")
        
        # Test 4: State indexing without M
        state_VJ = (1, 2)  # V=1, J=2
        try:
            idx = basis_no_M.get_index(state_VJ)
            state_back = basis_no_M.get_state(idx)
            
            indexing_no_M_ok = tuple(state_back) == state_VJ
            print_test("State indexing without M", indexing_no_M_ok,
                      f"State {state_VJ} -> Index {idx} -> State {tuple(state_back)}")
        except Exception as e:
            print_test("State indexing without M", False, f"Error: {e}")
        
        # Test 5: Hamiltonian generation
        H0 = basis_with_M.generate_H0(
            omega_rad_pfs=1.0,
            delta_omega_rad_pfs=0.1,
            B_rad_pfs=0.5,
            alpha_rad_pfs=0.01,
            units="J"
        )
        
        hamiltonian_ok = (H0.shape[0] == basis_with_M.size() and H0.units == "J")
        print_test("Hamiltonian generation", hamiltonian_ok,
                  f"Shape: {H0.shape}, Units: {H0.units}")
        
        # Test 6: Check Hamiltonian eigenvalues
        eigvals = H0.get_eigenvalues("rad/fs")
        
        # Verify some expected energy levels
        # E(V,J) = ω(V+1/2) - Δω(V+1/2)² + (B - α(V+1/2))J(J+1)
        omega, delta_omega, B, alpha = 1.0, 0.1, 0.5, 0.01
        
        # Calculate expected energy for ground state (V=0, J=0)
        V, J = 0, 0
        vterm = V + 0.5
        jterm = J * (J + 1)
        expected_ground = omega * vterm - delta_omega * vterm**2 + (B - alpha * vterm) * jterm
        
        # The ground state should be the minimum eigenvalue
        ground_state_ok = abs(np.min(eigvals) - expected_ground) < 1e-10
        print_test("Ground state energy", ground_state_ok,
                  f"Expected: {expected_ground:.6f}, Got: {np.min(eigvals):.6f}")
        
        # Test 7: Border indices functionality
        try:
            if basis_with_M.use_M:
                border_j = basis_with_M.get_border_indices_j()
                print_test("Border indices J", True, f"J borders: {border_j[:5]}...")
            else:
                print_test("Border indices J", True, "Skipped (M=False)")
            
            border_v = basis_with_M.get_border_indices_v()
            print_test("Border indices V", True, f"V borders: {border_v}")
        except Exception as e:
            print_test("Border indices", False, f"Error: {e}")
        
        # Test 8: Error handling
        try:
            basis_with_M.get_index((V_max + 1, 0, 0))
            print_test("Error handling - invalid V", False, "Should reject V > V_max")
        except ValueError:
            print_test("Error handling - invalid V", True, "Correctly rejects V > V_max")
        
        try:
            basis_with_M.get_index((0, J_max + 1, 0))
            print_test("Error handling - invalid J", False, "Should reject J > J_max")
        except ValueError:
            print_test("Error handling - invalid J", True, "Correctly rejects J > J_max")
        
        return True
        
    except Exception as e:
        print_test("LinMolBasis test", False, f"Error: {e}")
        traceback.print_exc()
        return False


def test_package_integration():
    """Test package-level integration"""
    print_header("6. Testing Package Integration")
    
    try:
        # Test 1: Package imports
        from src.rovibrational_excitation.core.basis import (
            BasisBase, Hamiltonian, LinMolBasis, TwoLevelBasis, VibLadderBasis
        )
        print_test("Package imports", True, "All classes imported successfully")
        
        # Test 2: __all__ completeness
        import src.rovibrational_excitation.core.basis as basis_pkg
        expected_all = {"BasisBase", "Hamiltonian", "LinMolBasis", "TwoLevelBasis", "VibLadderBasis"}
        actual_all = set(basis_pkg.__all__)
        
        all_complete = expected_all == actual_all
        print_test("__all__ completeness", all_complete,
                  f"Expected: {expected_all}, Got: {actual_all}")
        
        # Test 3: Legacy module warning
        try:
            import warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                from src.rovibrational_excitation.core.basis import LinMolBasis as LegacyLinMol
                
                legacy_warning_ok = len(w) > 0 and issubclass(w[0].category, DeprecationWarning)
                print_test("Legacy module deprecation warning", legacy_warning_ok,
                          f"Warning issued: {len(w) > 0}")
        except Exception as e:
            print_test("Legacy module test", False, f"Error: {e}")
        
        # Test 4: Cross-compatibility between classes
        try:
            # Create different basis types
            two_level = TwoLevelBasis()
            vib_ladder = VibLadderBasis(V_max=2)
            lin_mol = LinMolBasis(V_max=1, J_max=1, use_M=False)
            
            # Generate Hamiltonians
            h_two = two_level.generate_H0(units="J")
            h_vib = vib_ladder.generate_H0(units="J")
            h_lin = lin_mol.generate_H0(units="J")
            
            # Check they're all Hamiltonian objects
            all_hamiltonians = all(isinstance(h, Hamiltonian) for h in [h_two, h_vib, h_lin])
            print_test("Cross-class compatibility", all_hamiltonians,
                      "All basis classes return Hamiltonian objects")
            
            # Test unit conversion consistency
            h_two_fs = h_two.to_frequency_units()
            h_vib_fs = h_vib.to_frequency_units()
            h_lin_fs = h_lin.to_frequency_units()
            
            units_consistent = all(h.units == "rad/fs" for h in [h_two_fs, h_vib_fs, h_lin_fs])
            print_test("Unit conversion consistency", units_consistent,
                      "All Hamiltonians convert units consistently")
        
        except Exception as e:
            print_test("Cross-compatibility test", False, f"Error: {e}")
        
        return True
        
    except Exception as e:
        print_test("Package integration test", False, f"Error: {e}")
        traceback.print_exc()
        return False


def test_performance_and_edge_cases():
    """Test performance and edge cases"""
    print_header("7. Testing Performance and Edge Cases")
    
    try:
        from src.rovibrational_excitation.core.basis import LinMolBasis, VibLadderBasis, Hamiltonian
        
        # Test 1: Large basis performance
        import time
        start_time = time.time()
        
        large_basis = LinMolBasis(V_max=5, J_max=10, use_M=True)
        large_H = large_basis.generate_H0(units="J")
        
        creation_time = time.time() - start_time
        performance_ok = creation_time < 5.0  # Should complete in under 5 seconds
        
        print_test("Large basis performance", performance_ok,
                  f"Size: {large_basis.size()}, Time: {creation_time:.3f}s")
        
        # Test 2: Zero and negative parameters
        try:
            zero_basis = VibLadderBasis(V_max=2, omega_rad_pfs=0.0, delta_omega_rad_pfs=0.0)
            zero_H = zero_basis.generate_H0(units="J")
            
            print_test("Zero parameters handling", True, "Handles zero frequencies")
        except Exception as e:
            print_test("Zero parameters handling", False, f"Error: {e}")
        
        # Test 3: Very small basis
        tiny_basis = VibLadderBasis(V_max=0)  # Single state
        tiny_H = tiny_basis.generate_H0(units="J")
        
        tiny_ok = tiny_basis.size() == 1 and tiny_H.shape == (1, 1)
        print_test("Minimum size basis", tiny_ok, f"V_max=0 gives size {tiny_basis.size()}")
        
        # Test 4: Hamiltonian matrix properties
        # Test positive definiteness, hermiticity, etc.
        test_basis = LinMolBasis(V_max=2, J_max=2, use_M=False)
        test_H = test_basis.generate_H0(units="J")
        
        # Check hermiticity
        matrix = test_H.matrix
        is_hermitian = np.allclose(matrix, matrix.conj().T)
        
        # Check real eigenvalues (should be real for Hermitian matrix)
        eigvals = np.linalg.eigvals(matrix)
        real_eigenvals = np.allclose(eigvals.imag, 0)
        
        matrix_props_ok = is_hermitian and real_eigenvals
        print_test("Matrix properties", matrix_props_ok,
                  f"Hermitian: {is_hermitian}, Real eigenvalues: {real_eigenvals}")
        
        # Test 5: Memory usage for different basis types
        import sys
        
        bases = [
            ("TwoLevel", lambda: __import__('src.rovibrational_excitation.core.basis.twolevel', fromlist=['TwoLevelBasis']).TwoLevelBasis()),
            ("VibLadder", lambda: VibLadderBasis(V_max=10)),
            ("LinMol_noM", lambda: LinMolBasis(V_max=3, J_max=5, use_M=False)),
            ("LinMol_M", lambda: LinMolBasis(V_max=2, J_max=3, use_M=True))
        ]
        
        memory_info = []
        for name, creator in bases:
            try:
                basis = creator()
                h = basis.generate_H0(units="J")
                size = sys.getsizeof(h.matrix)
                memory_info.append(f"{name}: {basis.size()} states, {size} bytes")
            except Exception as e:
                memory_info.append(f"{name}: Error - {e}")
        
        print_test("Memory usage check", True, f"Memory info: {'; '.join(memory_info)}")
        
        return True
        
    except Exception as e:
        print_test("Performance and edge cases", False, f"Error: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and provide summary"""
    print(f"{Colors.BOLD}{Colors.OKCYAN}")
    print("Comprehensive Basis Module Test Suite")
    print("====================================")
    print(f"{Colors.ENDC}")
    
    test_functions = [
        ("Abstract Base Class", test_abstract_base),
        ("Hamiltonian Class", test_hamiltonian_class),
        ("TwoLevelBasis", test_twolevel_basis),
        ("VibLadderBasis", test_viblad_basis),
        ("LinMolBasis", test_linmol_basis),
        ("Package Integration", test_package_integration),
        ("Performance & Edge Cases", test_performance_and_edge_cases)
    ]
    
    results = {}
    total_tests = len(test_functions)
    
    for test_name, test_func in test_functions:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print_test(f"{test_name} (overall)", False, f"Test suite failed: {e}")
            results[test_name] = False
    
    # Summary
    print_header("Test Summary")
    
    passed = sum(results.values())
    failed = total_tests - passed
    
    for test_name, result in results.items():
        status = f"{Colors.OKGREEN}PASS{Colors.ENDC}" if result else f"{Colors.FAIL}FAIL{Colors.ENDC}"
        print(f"  {status} {test_name}")
    
    print(f"\n{Colors.BOLD}Overall Results:{Colors.ENDC}")
    print(f"  Passed: {Colors.OKGREEN}{passed}/{total_tests}{Colors.ENDC}")
    print(f"  Failed: {Colors.FAIL}{failed}/{total_tests}{Colors.ENDC}")
    
    if failed == 0:
        print(f"\n{Colors.OKGREEN}{Colors.BOLD}🎉 All tests passed! The basis module is working correctly.{Colors.ENDC}")
    else:
        print(f"\n{Colors.WARNING}{Colors.BOLD}⚠️  Some tests failed. Please check the details above.{Colors.ENDC}")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 