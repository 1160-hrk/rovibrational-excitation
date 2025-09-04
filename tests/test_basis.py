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
        V_max=2, J_max=2, use_M=True, omega_rad_pfs=2.0, delta_omega_rad_pfs=0.1
    )

    assert basis.V_max == 2
    assert basis.J_max == 2
    assert basis.use_M
    assert basis.omega_rad_pfs == 2.0
    assert basis.delta_omega_rad_pfs == 0.1


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

    # デフォルトパラメータでのテスト（周波数単位で出力）
    H0 = basis.generate_H0(
        omega_rad_pfs=1.0, B_rad_pfs=0.5, delta_omega_rad_pfs=0.0, alpha_rad_pfs=0.0,
        units="rad/fs"  # 周波数単位で返す
    )

    # 期待されるエネルギー（周波数単位）
    # [V=0,J=0]: 1.0*0.5 + 0.5*0 = 0.5
    # [V=0,J=1]: 1.0*0.5 + 0.5*2 = 1.5
    # [V=1,J=0]: 1.0*1.5 + 0.5*0 = 1.5
    # [V=1,J=1]: 1.0*1.5 + 0.5*2 = 2.5
    expected_energies = [0.5, 1.5, 1.5, 2.5]
    expected = np.diag(expected_energies)

    # Hamiltonianオブジェクトから行列を取得して比較
    np.testing.assert_array_almost_equal(H0.matrix, expected)


def test_linmol_generate_H0_anharmonic():
    """非調和性を含むハミルトニアンのテスト"""
    basis = LinMolBasis(V_max=1, J_max=0, use_M=False)

    H0 = basis.generate_H0(
        omega_rad_pfs=1.0, delta_omega_rad_pfs=0.1, B_rad_pfs=0.0, alpha_rad_pfs=0.0,
        units="rad/fs"  # 周波数単位で返す
    )

    # 現行仕様: E = (ω+Δω)*(V+1/2) - (Δω/2)*(V+1/2)^2
    # V=0: (1.0+0.1)*0.5 - 0.05*0.25 = 0.5375
    # V=1: (1.0+0.1)*1.5 - 0.05*2.25 = 1.5375
    expected_energies = [0.5375, 1.5375]
    expected = np.diag(expected_energies)

    # Hamiltonianオブジェクトから行列を取得して比較
    np.testing.assert_array_almost_equal(H0.matrix, expected)


def test_linmol_generate_H0_vibrot_coupling():
    """振動-回転結合のテスト"""
    basis = LinMolBasis(V_max=1, J_max=1, use_M=False)

    H0 = basis.generate_H0(
        omega_rad_pfs=1.0, B_rad_pfs=0.5, delta_omega_rad_pfs=0.0, alpha_rad_pfs=0.1,
        units="rad/fs"  # 周波数単位で返す
    )

    # E = ω*(V+1/2) + (B - α*(V+1/2))*J*(J+1)
    # [V=0,J=0]: 0.5 + (0.5-0.1*0.5)*0 = 0.5
    # [V=0,J=1]: 0.5 + (0.5-0.1*0.5)*2 = 0.5 + 0.45*2 = 1.4
    # [V=1,J=0]: 1.5 + (0.5-0.1*1.5)*0 = 1.5
    # [V=1,J=1]: 1.5 + (0.5-0.1*1.5)*2 = 1.5 + 0.35*2 = 2.2
    expected_energies = [0.5, 1.4, 1.5, 2.2]
    expected = np.diag(expected_energies)

    # Hamiltonianオブジェクトから行列を取得して比較
    np.testing.assert_array_almost_equal(H0.matrix, expected)


def test_linmol_hamiltonian_properties():
    """ハミルトニアンの性質テスト"""
    basis = LinMolBasis(V_max=2, J_max=2, use_M=False)
    H0 = basis.generate_H0()

    # Hamiltonianオブジェクトから行列を取得
    H0_matrix = H0.matrix

    # エルミート性
    np.testing.assert_array_equal(H0_matrix, H0_matrix.conj().T)

    # 対角性
    assert np.allclose(H0_matrix - np.diag(np.diag(H0_matrix)), 0)

    # 実数性
    assert np.allclose(H0_matrix.imag, 0)


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

    # 各状態について
    for i in range(basis.size()):
        state = basis.get_state(i)
        assert state[0] == basis.V_array[i]
        assert state[1] == basis.J_array[i]
        assert state[2] == basis.M_array[i]


def test_linmol_hamiltonian_units():
    """ハミルトニアンの単位変換テスト"""
    basis = LinMolBasis(V_max=1, J_max=0, use_M=False)

    # 周波数単位でのハミルトニアン
    H0_freq = basis.generate_H0(omega_rad_pfs=1000.0, units="rad/fs")
    assert H0_freq.units == "rad/fs"

    # エネルギー単位でのハミルトニアン
    H0_energy = basis.generate_H0(omega_rad_pfs=1000.0, units="J")
    assert H0_energy.units == "J"

    # 単位変換が正しく行われているかチェック
    # E = hbar * omega なので、エネルギー値は周波数値にhbarを掛けたものになる
    # hbar ≈ 1.055e-34 J·s = 1.055e-19 J·fs なので、エネルギー値は非常に小さくなる
    assert H0_energy.units == "J"
    assert H0_freq.units == "rad/fs"
    
    # 単位変換のテスト - 相対的な比較
    freq_eigenvals = H0_freq.eigenvalues
    energy_eigenvals = H0_energy.eigenvalues
    
    # エネルギー差の比率が周波数差の比率と一致することを確認
    freq_ratio = freq_eigenvals[1] / freq_eigenvals[0] if freq_eigenvals[0] != 0 else float('inf')
    energy_ratio = energy_eigenvals[1] / energy_eigenvals[0] if energy_eigenvals[0] != 0 else float('inf')
    
    np.testing.assert_almost_equal(freq_ratio, energy_ratio, decimal=10)
