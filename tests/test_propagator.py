import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
import numpy as np
from rovibrational_excitation.core.propagator import schrodinger_propagation, mixed_state_propagation, liouville_propagation
from rovibrational_excitation.core.electric_field import ElectricField
from rovibrational_excitation.core.basis import LinMolBasis
import pytest

class DummyDipole:
    def __init__(self):
        self.mu_x = np.eye(2, dtype=np.complex128)
        self.mu_y = np.zeros((2,2), dtype=np.complex128)
        self.mu_z = np.zeros((2,2), dtype=np.complex128)

def test_schrodinger_propagation():
    tlist = np.linspace(0, 1, 3)
    ef = ElectricField(tlist)
    ef.Efield[:,0] = 1.0
    basis = LinMolBasis(V_max=0, J_max=1, use_M=False)
    H0 = np.diag([0.0, 1.0])
    dip = DummyDipole()
    psi0 = np.array([1.0, 0.0], dtype=np.complex128)
    result = schrodinger_propagation(H0, ef, dip, psi0)
    assert result.shape[-1] == 2 or result[1].shape[-1] == 2

def test_mixed_state_propagation():
    tlist = np.linspace(0, 1, 3)
    ef = ElectricField(tlist)
    ef.Efield[:,0] = 1.0
    basis = LinMolBasis(V_max=0, J_max=1, use_M=False)
    H0 = np.diag([0.0, 1.0])
    dip = DummyDipole()
    psi0s = [np.array([1.0, 0.0], dtype=np.complex128), np.array([0.0, 1.0], dtype=np.complex128)]
    result = mixed_state_propagation(H0, ef, psi0s, dip)
    assert result.shape[-1] == 2 or result[1].shape[-1] == 2

def test_liouville_propagation():
    tlist = np.linspace(0, 1, 3)
    ef = ElectricField(tlist)
    ef.Efield[:,0] = 1.0
    basis = LinMolBasis(V_max=0, J_max=1, use_M=False)
    H0 = np.diag([0.0, 1.0])
    dip = DummyDipole()
    rho0 = np.eye(2, dtype=np.complex128)
    result = liouville_propagation(H0, ef, dip, rho0)
    assert result.shape[-1] == 2 or result[1].shape[-1] == 2 