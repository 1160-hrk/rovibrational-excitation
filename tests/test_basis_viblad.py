import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
import numpy as np
import pytest

from rovibrational_excitation.core.basis import VibLadderBasis


def test_viblad_basic():
    """基本的な機能のテスト"""
    basis = VibLadderBasis(V_max=2)

    # サイズ = V_max + 1
    assert basis.size() == 3

    # 基底状態の形状確認
    assert basis.basis.shape == (3, 1)
    np.testing.assert_array_equal(basis.basis, [[0], [1], [2]])


def test_viblad_initialization():
    """初期化パラメータのテスト"""
    basis = VibLadderBasis(V_max=3, omega_rad_phz=2.0, delta_omega_rad_phz=0.1)

    assert basis.V_max == 3
    assert basis.omega_rad_phz == 2.0
    assert basis.delta_omega_rad_phz == 0.1
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
    basis = VibLadderBasis(V_max=2, omega_rad_phz=1.0, delta_omega_rad_phz=0.0)
    H0 = basis.generate_H0()

    # E = ω*(v+1/2)
    expected_energies = [0.5, 1.5, 2.5]  # v=0,1,2
    expected = np.diag(expected_energies)

    np.testing.assert_array_almost_equal(H0, expected)


def test_viblad_generate_H0_custom():
    """カスタムパラメータでのハミルトニアン生成テスト"""
    basis = VibLadderBasis(V_max=2)
    H0 = basis.generate_H0(omega_rad_phz=2.0, delta_omega_rad_phz=0.1)

    # E = ω*(v+1/2) - Δω*(v+1/2)^2
    # v=0: 2.0*0.5 - 0.1*0.25 = 1.0 - 0.025 = 0.975
    # v=1: 2.0*1.5 - 0.1*2.25 = 3.0 - 0.225 = 2.775
    # v=2: 2.0*2.5 - 0.1*6.25 = 5.0 - 0.625 = 4.375
    expected_energies = [0.975, 2.775, 4.375]
    expected = np.diag(expected_energies)

    np.testing.assert_array_almost_equal(H0, expected)


def test_viblad_generate_H0_anharmonic():
    """非調和性を含むハミルトニアン生成テスト"""
    basis = VibLadderBasis(V_max=1, omega_rad_phz=1.0, delta_omega_rad_phz=0.1)
    H0 = basis.generate_H0()

    # E = ω*(v+1/2) - Δω*(v+1/2)^2
    # v=0: 1.0*0.5 - 0.1*0.25 = 0.475
    # v=1: 1.0*1.5 - 0.1*2.25 = 1.275
    expected_energies = [0.475, 1.275]
    expected = np.diag(expected_energies)

    np.testing.assert_array_almost_equal(H0, expected)


def test_viblad_generate_H0_override():
    """パラメータ上書きテスト"""
    basis = VibLadderBasis(V_max=1, omega_rad_phz=1.0, delta_omega_rad_phz=0.1)

    # インスタンスパラメータを使用
    H0_instance = basis.generate_H0()

    # パラメータを上書き
    H0_override = basis.generate_H0(omega_rad_phz=2.0, delta_omega_rad_phz=0.0)

    # 結果が異なること
    assert not np.allclose(H0_instance, H0_override)

    # 上書き結果の確認
    expected_override = np.diag([1.0, 3.0])  # 2.0*(v+0.5)
    np.testing.assert_array_almost_equal(H0_override, expected_override)


def test_viblad_hamiltonian_properties():
    """ハミルトニアンの性質のテスト"""
    basis = VibLadderBasis(V_max=3, omega_rad_phz=2.5, delta_omega_rad_phz=0.05)
    H0 = basis.generate_H0()

    # エルミート性
    np.testing.assert_array_equal(H0, H0.conj().T)

    # 対角性
    assert np.allclose(H0 - np.diag(np.diag(H0)), 0)

    # 実数性
    assert np.allclose(H0.imag, 0)

    # エネルギー順序（調和項が支配的な場合は単調増加）
    energies = np.diag(H0)
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
    basis1 = VibLadderBasis(V_max=2, omega_rad_phz=1.0)
    basis2 = VibLadderBasis(V_max=2, omega_rad_phz=2.0)

    # 異なるパラメータで異なるハミルトニアン
    H0_1 = basis1.generate_H0()
    H0_2 = basis2.generate_H0()
    assert not np.allclose(H0_1, H0_2)

    # 独立したオブジェクトであること
    assert basis1 is not basis2
    assert basis1.basis is not basis2.basis
