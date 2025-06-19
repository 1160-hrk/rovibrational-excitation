import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
import numpy as np
from rovibrational_excitation.core.basis import LinMolBasis
from rovibrational_excitation.core.hamiltonian import generate_H0_LinMol

def test_generate_H0_LinMol_shape_and_value():
    basis = LinMolBasis(V_max=1, J_max=1, use_M=False)
    H0 = generate_H0_LinMol(basis, omega_rad_phz=2.0, delta_omega_rad_phz=0.0, B_rad_phz=1.0, alpha_rad_phz=0.0)
    # H0は対角行列
    assert H0.shape[0] == H0.shape[1]
    # 対角要素の値を計算
    vj = basis.basis
    expected = []
    for v, j in vj:
        energy = 2.0 * (v + 0.5) + 1.0 * (j * (j + 1))
        expected.append(energy)
    np.testing.assert_allclose(np.diag(H0), expected) 