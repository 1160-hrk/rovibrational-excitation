import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
import numpy as np
import pytest
from rovibrational_excitation.core.basis import LinMolBasis

def test_basis_generate_and_size():
    basis = LinMolBasis(V_max=1, J_max=1, use_M=True)
    # V=0,1; J=0,1; M=-J..J → (1+3)*2=8個
    assert basis.size() == 8
    # basis内容の形状
    assert basis.basis.shape[1] == 3

def test_basis_get_index_and_state():
    basis = LinMolBasis(V_max=1, J_max=1, use_M=True)
    idx = basis.get_index([0,1,0])
    assert idx is not None
    state = basis.get_state(idx)
    assert np.all(state == [0,1,0])
    # 存在しない状態
    assert basis.get_index([9,9,9]) is None

def test_basis_repr():
    basis = LinMolBasis(V_max=1, J_max=1, use_M=True)
    s = repr(basis)
    assert "VJMBasis" in s

def test_basis_border_indices():
    basis = LinMolBasis(V_max=1, J_max=1, use_M=True)
    inds_j = basis.get_border_indices_j()
    inds_v = basis.get_border_indices_v()
    assert isinstance(inds_j, np.ndarray)
    assert isinstance(inds_v, np.ndarray)
    # use_M=False時の例外
    basis2 = LinMolBasis(V_max=1, J_max=1, use_M=False)
    # get_border_indices_jは例外、get_border_indices_vは正常
    with pytest.raises(ValueError):
        basis2.get_border_indices_j()
    inds_v2 = basis2.get_border_indices_v()
    assert isinstance(inds_v2, np.ndarray) 