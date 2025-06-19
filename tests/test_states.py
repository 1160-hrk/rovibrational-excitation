import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
import numpy as np
import pytest
from rovibrational_excitation.core.basis import LinMolBasis
from rovibrational_excitation.core.states import StateVector, DensityMatrix

def test_statevector_basic():
    basis = LinMolBasis(V_max=0, J_max=1, use_M=False)
    sv = StateVector(basis)
    # 初期状態はゼロ
    assert np.all(sv.data == 0)
    # 状態設定
    sv.set_state([0,0])
    assert np.isclose(sv.norm(), 1.0)
    # 正規化
    sv.data *= 2
    sv.normalize()
    assert np.isclose(sv.norm(), 1.0)
    # コピー
    sv2 = sv.copy()
    assert np.all(sv2.data == sv.data)
    # repr
    assert "StateVector" in repr(sv)
    # 存在しない状態
    with pytest.raises(ValueError):
        sv.set_state([9,9])

def test_densitymatrix_basic():
    basis = LinMolBasis(V_max=0, J_max=1, use_M=False)
    sv = StateVector(basis)
    sv.set_state([0,0])
    dm = DensityMatrix(basis)
    # 対角設定
    dm.set_diagonal([1,0])
    assert np.isclose(dm.trace(), 1.0)
    # 純粋状態設定
    dm.set_pure_state(sv)
    assert np.isclose(dm.trace(), 1.0)
    # 正規化
    dm.data *= 2
    dm.normalize()
    assert np.isclose(dm.trace(), 1.0)
    # コピー
    dm2 = dm.copy()
    assert np.all(dm2.data == dm.data)
    # repr
    assert "DensityMatrix" in repr(dm)
    # 例外
    with pytest.raises(ValueError):
        dm.set_diagonal([1,2,3]) 