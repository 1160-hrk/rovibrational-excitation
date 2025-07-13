import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
import numpy as np
import pytest

from rovibrational_excitation.core.basis import TwoLevelBasis


def test_twolevel_basic():
    """基本的な機能のテスト"""
    basis = TwoLevelBasis()

    # サイズは常に2
    assert basis.size() == 2

    # 基底状態の形状確認
    assert basis.basis.shape == (2, 1)
    np.testing.assert_array_equal(basis.basis, [[0], [1]])


def test_twolevel_get_index():
    """インデックス取得のテスト"""
    basis = TwoLevelBasis()

    # 整数での指定
    assert basis.get_index(0) == 0
    assert basis.get_index(1) == 1

    # タプルでの指定
    assert basis.get_index((0,)) == 0
    assert basis.get_index((1,)) == 1

    # リストでの指定
    assert basis.get_index([0]) == 0
    assert basis.get_index([1]) == 1

    # numpy integerでの指定
    assert basis.get_index(np.int32(0)) == 0
    assert basis.get_index(np.int64(1)) == 1


def test_twolevel_get_index_errors():
    """get_indexのエラーケースのテスト"""
    basis = TwoLevelBasis()

    # 無効な整数
    with pytest.raises(ValueError):
        basis.get_index(2)

    with pytest.raises(ValueError):
        basis.get_index(-1)

    # 無効なタプル
    with pytest.raises(ValueError):
        basis.get_index((2,))

    # 無効な長さのタプル
    with pytest.raises(ValueError):
        basis.get_index((0, 1))

    # 無効なタイプ
    with pytest.raises(ValueError):
        basis.get_index("invalid")


def test_twolevel_get_state():
    """状態取得のテスト"""
    basis = TwoLevelBasis()

    state0 = basis.get_state(0)
    state1 = basis.get_state(1)

    np.testing.assert_array_equal(state0, [0])
    np.testing.assert_array_equal(state1, [1])


def test_twolevel_get_state_errors():
    """get_stateのエラーケースのテスト"""
    basis = TwoLevelBasis()

    with pytest.raises(ValueError):
        basis.get_state(2)

    with pytest.raises(ValueError):
        basis.get_state(-1)


def test_twolevel_generate_H0():
    """ハミルトニアン生成のテスト"""
    # デフォルトパラメータ（1.0 rad/fsのギャップ）
    basis = TwoLevelBasis()
    H0 = basis.generate_H0()
    
    # デフォルトではJ単位で出力
    assert H0.units == "J"
    
    # 周波数単位で出力する場合
    basis_freq = TwoLevelBasis(energy_gap=1.0, input_units="rad/fs", output_units="rad/fs")
    H0_freq = basis_freq.generate_H0()
    expected = np.diag([0.0, 1.0])
    np.testing.assert_array_equal(H0_freq.matrix, expected)
    assert H0_freq.units == "rad/fs"


def test_twolevel_hamiltonian_properties():
    """ハミルトニアンの性質のテスト"""
    basis = TwoLevelBasis()
    H0 = basis.generate_H0()

    # Hamiltonianオブジェクトから行列を取得
    H0_matrix = H0.matrix

    # エルミート性
    np.testing.assert_array_equal(H0_matrix, H0_matrix.conj().T)

    # 対角性
    assert np.allclose(H0_matrix - np.diag(np.diag(H0_matrix)), 0)

    # 実数性
    assert np.allclose(H0_matrix.imag, 0)


def test_twolevel_repr():
    """文字列表現のテスト"""
    basis = TwoLevelBasis()
    repr_str = repr(basis)

    assert "TwoLevelBasis" in repr_str
    assert "|0⟩" in repr_str
    assert "|1⟩" in repr_str


def test_twolevel_index_map_consistency():
    """index_mapの一貫性テスト"""
    basis = TwoLevelBasis()

    for i in range(basis.size()):
        state = basis.get_state(i)
        recovered_index = basis.get_index(tuple(state))
        assert recovered_index == i


def test_twolevel_multiple_instances():
    """複数インスタンスの独立性テスト"""
    basis1 = TwoLevelBasis()
    basis2 = TwoLevelBasis()

    # 同じ結果を返すこと
    assert basis1.size() == basis2.size()
    np.testing.assert_array_equal(basis1.basis, basis2.basis)

    # 独立したオブジェクトであること
    assert basis1 is not basis2
    assert basis1.basis is not basis2.basis
