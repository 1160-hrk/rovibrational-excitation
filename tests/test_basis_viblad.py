import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
import numpy as np
import pytest

from rovibrational_excitation.core.basis import VibLadderBasis


def test_viblad_basic():
    """基本的な機能のテスト"""
    basis = VibLadderBasis(V_max=2)

    # サイズはV_max+1
    assert basis.size() == 3

    # 基底状態の形状確認
    assert basis.basis.shape == (3, 1)
    np.testing.assert_array_equal(basis.basis, [[0], [1], [2]])


def test_viblad_initialization():
    """初期化パラメータのテスト"""
    basis = VibLadderBasis(V_max=3, omega=2.0, delta_omega=0.1, input_units="rad/fs")

    assert basis.V_max == 3
    assert basis.omega_rad_pfs == 2.0
    assert basis.delta_omega_rad_pfs == 0.1
    assert basis.size() == 4

    # V_arrayの確認
    np.testing.assert_array_equal(basis.V_array, [0, 1, 2, 3])


def test_viblad_get_index():
    """インデックス取得のテスト"""
    basis = VibLadderBasis(V_max=2)

    # 整数での指定
    assert basis.get_index(0) == 0
    assert basis.get_index(1) == 1
    assert basis.get_index(2) == 2

    # タプルでの指定
    assert basis.get_index((0,)) == 0
    assert basis.get_index((1,)) == 1
    assert basis.get_index((2,)) == 2

    # リストでの指定
    assert basis.get_index([0]) == 0
    assert basis.get_index([1]) == 1
    assert basis.get_index([2]) == 2

    # numpy integerでの指定
    assert basis.get_index(np.int32(0)) == 0
    assert basis.get_index(np.int64(1)) == 1


def test_viblad_get_index_errors():
    """get_indexのエラーケースのテスト"""
    basis = VibLadderBasis(V_max=2)

    # 範囲外の値
    with pytest.raises(ValueError):
        basis.get_index(3)

    with pytest.raises(ValueError):
        basis.get_index(-1)

    # 無効なタプル
    with pytest.raises(ValueError):
        basis.get_index((3,))

    # 無効な長さのタプル
    with pytest.raises(ValueError):
        basis.get_index((0, 1))

    # 無効なタイプ
    with pytest.raises(ValueError):
        basis.get_index("invalid")


def test_viblad_get_state():
    """状態取得のテスト"""
    basis = VibLadderBasis(V_max=2)

    state0 = basis.get_state(0)
    state1 = basis.get_state(1)
    state2 = basis.get_state(2)

    np.testing.assert_array_equal(state0, [0])
    np.testing.assert_array_equal(state1, [1])
    np.testing.assert_array_equal(state2, [2])


def test_viblad_get_state_errors():
    """get_stateのエラーケースのテスト"""
    basis = VibLadderBasis(V_max=2)

    with pytest.raises(ValueError):
        basis.get_state(3)

    with pytest.raises(ValueError):
        basis.get_state(-1)


def test_viblad_generate_H0_default():
    """デフォルトパラメータでのハミルトニアン生成テスト"""
    basis = VibLadderBasis(V_max=2, omega=1.0, delta_omega=0.0, input_units="rad/fs", output_units="rad/fs")
    H0 = basis.generate_H0()

    # E = ω*(v+1/2)
    expected_energies = [0.5, 1.5, 2.5]  # v=0,1,2
    expected = np.diag(expected_energies)

    np.testing.assert_array_almost_equal(H0.matrix, expected)


def test_viblad_generate_H0_custom():
    """カスタムパラメータでのハミルトニアン生成テスト"""
    basis = VibLadderBasis(V_max=2)
    H0 = basis.generate_H0_with_params(omega=2.0, delta_omega=0.1, input_units="rad/fs", units="rad/fs")

    # E = ω*(v+1/2) - Δω*(v+1/2)^2
    # v=0: 2.0*0.5 - 0.1*0.25 = 1.0 - 0.025 = 0.975
    # v=1: 2.0*1.5 - 0.1*2.25 = 3.0 - 0.225 = 2.775
    # v=2: 2.0*2.5 - 0.1*6.25 = 5.0 - 0.625 = 4.375
    expected_energies = [0.975, 2.775, 4.375]
    expected = np.diag(expected_energies)

    np.testing.assert_array_almost_equal(H0.matrix, expected)


def test_viblad_generate_H0_anharmonic():
    """非調和性を含むハミルトニアン生成テスト"""
    basis = VibLadderBasis(V_max=1, omega=1.0, delta_omega=0.1, input_units="rad/fs", output_units="rad/fs")
    H0 = basis.generate_H0()

    # 現行仕様: E = (ω+Δω)*(v+1/2) - (Δω/2)*(v+1/2)^2
    # v=0: (1.0+0.1)*0.5 - 0.05*0.25 = 0.5375
    # v=1: (1.0+0.1)*1.5 - 0.05*2.25 = 1.5375
    expected_energies = [0.5375, 1.5375]
    expected = np.diag(expected_energies)

    np.testing.assert_array_almost_equal(H0.matrix, expected)


def test_viblad_generate_H0_override():
    """パラメータ上書きテスト"""
    basis = VibLadderBasis(V_max=1, omega=1.0, delta_omega=0.1, input_units="rad/fs", output_units="rad/fs")

    # インスタンスパラメータを使用
    H0_instance = basis.generate_H0()

    # パラメータを上書き
    H0_override = basis.generate_H0_with_params(omega=2.0, delta_omega=0.0, input_units="rad/fs", units="rad/fs")

    # 結果が異なること
    assert not np.allclose(H0_instance.matrix, H0_override.matrix)

    # 上書き結果の確認
    expected_override = np.diag([1.0, 3.0])  # 2.0*(v+0.5)
    np.testing.assert_array_almost_equal(H0_override.matrix, expected_override)


def test_viblad_hamiltonian_properties():
    """ハミルトニアンの性質のテスト"""
    basis = VibLadderBasis(V_max=3, omega=2.5, delta_omega=0.05, input_units="rad/fs")
    H0 = basis.generate_H0()

    # Hamiltonianオブジェクトから行列を取得
    H0_matrix = H0.matrix

    # エルミート性
    np.testing.assert_array_equal(H0_matrix, H0_matrix.conj().T)

    # 対角性
    assert np.allclose(H0_matrix - np.diag(np.diag(H0_matrix)), 0)

    # 実数性
    assert np.allclose(H0_matrix.imag, 0)

    # エネルギー順序（調和項が支配的な場合は単調増加）
    energies = np.diag(H0_matrix)
    assert np.all(energies[1:] > energies[:-1])


def test_viblad_repr():
    """文字列表現のテスト"""
    basis = VibLadderBasis(V_max=3)
    repr_str = repr(basis)

    assert "VibLadderBasis" in repr_str
    assert "V_max=3" in repr_str
    assert "size=4" in repr_str


def test_viblad_index_map_consistency():
    """index_mapの一貫性テスト"""
    basis = VibLadderBasis(V_max=3)

    for i in range(basis.size()):
        state = basis.get_state(i)
        recovered_index = basis.get_index(tuple(state))
        assert recovered_index == i


def test_viblad_edge_cases():
    """エッジケースのテスト"""
    # V_max=0の場合
    basis_min = VibLadderBasis(V_max=0)
    assert basis_min.size() == 1
    assert basis_min.get_index(0) == 0

    # 大きなV_maxの場合
    basis_large = VibLadderBasis(V_max=100)
    assert basis_large.size() == 101
    assert basis_large.get_index(50) == 50
    assert basis_large.get_index(100) == 100


def test_viblad_multiple_instances():
    """複数インスタンスの独立性テスト"""
    basis1 = VibLadderBasis(V_max=2, omega=1.0, input_units="rad/fs")
    basis2 = VibLadderBasis(V_max=2, omega=2.0, input_units="rad/fs")

    # 異なるパラメータで初期化されていること
    assert basis1.omega_rad_pfs != basis2.omega_rad_pfs

    # 同じサイズを持つこと
    assert basis1.size() == basis2.size()

    # 独立したオブジェクトであること
    assert basis1 is not basis2
    assert basis1.basis is not basis2.basis

    # 異なるパラメータで異なるハミルトニアンを生成すること（rad/fs単位で比較）
    H0_1 = basis1.generate_H0_with_params(units="rad/fs")
    H0_2 = basis2.generate_H0_with_params(units="rad/fs")
    assert not np.allclose(H0_1.matrix, H0_2.matrix)


def test_viblad_hamiltonian_units():
    """ハミルトニアンの単位変換テスト"""
    basis = VibLadderBasis(V_max=1, omega=1000.0, input_units="rad/fs")

    # 周波数単位でのハミルトニアン
    H0_freq = basis.generate_H0_with_params(units="rad/fs")
    assert H0_freq.units == "rad/fs"

    # エネルギー単位でのハミルトニアン
    H0_energy = basis.generate_H0_with_params(units="J")
    assert H0_energy.units == "J"

    # 単位変換のテスト - 相対的な比較
    freq_eigenvals = H0_freq.eigenvalues
    energy_eigenvals = H0_energy.eigenvalues
    
    # エネルギー差の比率が周波数差の比率と一致することを確認
    freq_ratio = freq_eigenvals[1] / freq_eigenvals[0] if freq_eigenvals[0] != 0 else float('inf')
    energy_ratio = energy_eigenvals[1] / energy_eigenvals[0] if energy_eigenvals[0] != 0 else float('inf')
    
    np.testing.assert_almost_equal(freq_ratio, energy_ratio, decimal=10)
