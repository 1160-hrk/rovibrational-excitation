import os
import sys

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
    idx = basis.get_index([0, 1, 0])
    assert idx is not None
    state = basis.get_state(idx)
    assert np.all(state == [0, 1, 0])
    # 存在しない状態でエラーが発生
    with pytest.raises(ValueError):
        basis.get_index([9, 9, 9])


def test_basis_repr():
    basis = LinMolBasis(V_max=1, J_max=1, use_M=True)
    s = repr(basis)
    assert "LinMolBasis" in s


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


def test_linmol_initialization_parameters():
    """初期化パラメータのテスト"""
    basis = LinMolBasis(
        V_max=2, J_max=2, use_M=True, omega_rad_phz=2.0, delta_omega_rad_phz=0.1
    )

    assert basis.V_max == 2
    assert basis.J_max == 2
    assert basis.use_M
    assert basis.omega_rad_phz == 2.0
    assert basis.delta_omega_rad_phz == 0.1


def test_linmol_no_M_basis():
    """use_M=False時の基底テスト"""
    basis = LinMolBasis(V_max=1, J_max=1, use_M=False)

    # サイズは(V_max+1)*(J_max+1) = 2*2 = 4
    assert basis.size() == 4
    assert basis.basis.shape[1] == 2  # [V, J]のみ

    # 期待される基底状態
    expected_basis = [[0, 0], [0, 1], [1, 0], [1, 1]]
    np.testing.assert_array_equal(basis.basis, expected_basis)


def test_linmol_with_M_basis():
    """use_M=True時の基底テスト"""
    basis = LinMolBasis(V_max=1, J_max=1, use_M=True)

    # V=0,1; J=0,1; M=-J..J
    # V=0,J=0,M=0: 1個
    # V=0,J=1,M=-1,0,1: 3個
    # V=1,J=0,M=0: 1個
    # V=1,J=1,M=-1,0,1: 3個
    # 合計: 8個
    assert basis.size() == 8
    assert basis.basis.shape[1] == 3  # [V, J, M]


def test_linmol_generate_H0():
    """ハミルトニアン生成の詳細テスト"""
    basis = LinMolBasis(V_max=1, J_max=1, use_M=False)

    # デフォルトパラメータでのテスト
    H0 = basis.generate_H0(
        omega_rad_phz=1.0, B_rad_phz=0.5, delta_omega_rad_phz=0.0, alpha_rad_phz=0.0
    )

    # 期待されるエネルギー
    # [V=0,J=0]: 1.0*0.5 + 0.5*0 = 0.5
    # [V=0,J=1]: 1.0*0.5 + 0.5*2 = 1.5
    # [V=1,J=0]: 1.0*1.5 + 0.5*0 = 1.5
    # [V=1,J=1]: 1.0*1.5 + 0.5*2 = 2.5
    expected_energies = [0.5, 1.5, 1.5, 2.5]
    expected = np.diag(expected_energies)

    np.testing.assert_array_almost_equal(H0, expected)


def test_linmol_generate_H0_anharmonic():
    """非調和性を含むハミルトニアンのテスト"""
    basis = LinMolBasis(V_max=1, J_max=0, use_M=False)

    H0 = basis.generate_H0(
        omega_rad_phz=1.0, delta_omega_rad_phz=0.1, B_rad_phz=0.0, alpha_rad_phz=0.0
    )

    # E = ω*(V+1/2) - Δω*(V+1/2)^2
    # V=0: 1.0*0.5 - 0.1*0.25 = 0.475
    # V=1: 1.0*1.5 - 0.1*2.25 = 1.275
    expected_energies = [0.475, 1.275]
    expected = np.diag(expected_energies)

    np.testing.assert_array_almost_equal(H0, expected)


def test_linmol_generate_H0_vibrot_coupling():
    """振動-回転結合のテスト"""
    basis = LinMolBasis(V_max=1, J_max=1, use_M=False)

    H0 = basis.generate_H0(
        omega_rad_phz=1.0, B_rad_phz=0.5, delta_omega_rad_phz=0.0, alpha_rad_phz=0.1
    )

    # E = ω*(V+1/2) + (B - α*(V+1/2))*J*(J+1)
    # [V=0,J=0]: 0.5 + (0.5-0.1*0.5)*0 = 0.5
    # [V=0,J=1]: 0.5 + (0.5-0.1*0.5)*2 = 0.5 + 0.45*2 = 1.4
    # [V=1,J=0]: 1.5 + (0.5-0.1*1.5)*0 = 1.5
    # [V=1,J=1]: 1.5 + (0.5-0.1*1.5)*2 = 1.5 + 0.35*2 = 2.2
    expected_energies = [0.5, 1.4, 1.5, 2.2]
    expected = np.diag(expected_energies)

    np.testing.assert_array_almost_equal(H0, expected)


def test_linmol_hamiltonian_properties():
    """ハミルトニアンの性質テスト"""
    basis = LinMolBasis(V_max=2, J_max=2, use_M=False)
    H0 = basis.generate_H0()

    # エルミート性
    np.testing.assert_array_equal(H0, H0.conj().T)

    # 対角性
    assert np.allclose(H0 - np.diag(np.diag(H0)), 0)

    # 実数性
    assert np.allclose(H0.imag, 0)


def test_linmol_get_index_various_inputs():
    """様々な入力形式でのget_indexテスト"""
    basis = LinMolBasis(V_max=1, J_max=1, use_M=True)

    # リスト入力
    idx1 = basis.get_index([0, 0, 0])

    # タプル入力
    idx2 = basis.get_index((0, 0, 0))

    # numpy配列入力
    idx3 = basis.get_index(np.array([0, 0, 0]))

    assert idx1 == idx2 == idx3


def test_linmol_edge_cases():
    """エッジケースのテスト"""
    # 最小基底
    basis_min = LinMolBasis(V_max=0, J_max=0, use_M=True)
    assert basis_min.size() == 1

    # V_max=0, J_max=0の場合
    basis_single = LinMolBasis(V_max=0, J_max=0, use_M=False)
    assert basis_single.size() == 1

    # 大きなJ_maxの場合
    basis_large_j = LinMolBasis(V_max=0, J_max=5, use_M=False)
    assert basis_large_j.size() == 6  # J=0,1,2,3,4,5


def test_linmol_arrays_consistency():
    """配列の一貫性テスト"""
    basis = LinMolBasis(V_max=2, J_max=2, use_M=True)

    # V_array, J_arrayの長さが基底数と一致
    assert len(basis.V_array) == basis.size()
    assert len(basis.J_array) == basis.size()
    assert len(basis.M_array) == basis.size()

    # 配列値が基底と一致
    for i in range(basis.size()):
        state = basis.get_state(i)
        assert basis.V_array[i] == state[0]
        assert basis.J_array[i] == state[1]
        assert basis.M_array[i] == state[2]
