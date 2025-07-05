import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
import numpy as np

from rovibrational_excitation.core.basis import LinMolBasis


def test_generate_H0_LinMol_shape_and_value():
    basis = LinMolBasis(V_max=1, J_max=1, use_M=False)
    H0 = basis.generate_H0(
        omega_rad_pfs=2.0, delta_omega_rad_pfs=0.0, B_rad_pfs=1.0, alpha_rad_pfs=0.0,
        units="rad/fs"
    )
    
    # Hamiltonianオブジェクトから行列を取得
    H0_matrix = H0.matrix
    
    # H0は対角行列
    assert H0_matrix.shape[0] == H0_matrix.shape[1]
    # 対角要素の値を計算
    vj = basis.basis
    expected = []
    for v, j in vj:
        energy = 2.0 * (v + 0.5) + 1.0 * (j * (j + 1))
        expected.append(energy)
    np.testing.assert_allclose(np.diag(H0_matrix), expected)
