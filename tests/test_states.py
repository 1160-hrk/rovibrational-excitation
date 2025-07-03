import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
import numpy as np
import pytest

from rovibrational_excitation.core.basis import LinMolBasis, StateVector, DensityMatrix


def test_statevector_basic():
    basis = LinMolBasis(V_max=0, J_max=1, use_M=False)
    sv = StateVector(basis)
    # 初期状態はゼロ
    assert np.all(sv.data == 0)
    # 状態設定
    sv.set_state([0, 0])
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
        sv.set_state([9, 9])


def test_densitymatrix_basic():
    basis = LinMolBasis(V_max=0, J_max=1, use_M=False)
    sv = StateVector(basis)
    sv.set_state([0, 0])
    dm = DensityMatrix(basis)
    # 対角設定
    dm.set_diagonal([1, 0])
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
        dm.set_diagonal([1, 2, 3])


def test_statevector_advanced():
    """StateVectorの高度なテスト"""
    basis = LinMolBasis(V_max=1, J_max=1, use_M=False)
    sv = StateVector(basis)

    # 重ね合わせ状態の作成
    sv.data[0] = 1.0 / np.sqrt(2)
    sv.data[1] = 1.0 / np.sqrt(2)
    assert np.isclose(sv.norm(), 1.0)

    # 位相を持つ状態
    sv.data[0] = 1.0 / np.sqrt(2)
    sv.data[1] = 1j / np.sqrt(2)
    assert np.isclose(sv.norm(), 1.0)

    # 非正規化状態からの正規化
    sv.data[0] = 3.0
    sv.data[1] = 4.0
    sv.normalize()
    expected_norm = 1.0
    assert np.isclose(sv.norm(), expected_norm)

    # ゼロ状態の正規化（エラーにならない）
    sv.data[:] = 0
    sv.normalize()  # ゼロ除算回避
    assert sv.norm() == 0


def test_statevector_different_basis():
    """異なる基底でのStateVectorテスト"""
    # use_M=True
    basis_with_m = LinMolBasis(V_max=1, J_max=1, use_M=True)
    sv_with_m = StateVector(basis_with_m)
    sv_with_m.set_state([0, 0, 0])  # V=0, J=0, M=0
    assert np.isclose(sv_with_m.norm(), 1.0)

    # use_M=False
    basis_no_m = LinMolBasis(V_max=1, J_max=1, use_M=False)
    sv_no_m = StateVector(basis_no_m)
    sv_no_m.set_state([0, 0])  # V=0, J=0
    assert np.isclose(sv_no_m.norm(), 1.0)

    # サイズが異なる
    assert sv_with_m.data.size != sv_no_m.data.size


def test_densitymatrix_advanced():
    """DensityMatrixの高度なテスト"""
    basis = LinMolBasis(V_max=1, J_max=1, use_M=False)
    dm = DensityMatrix(basis)

    # 統計混合状態
    populations = [0.6, 0.3, 0.05, 0.05]
    dm.set_diagonal(populations)
    assert np.isclose(dm.trace(), 1.0)

    # エルミート性確認
    assert np.allclose(dm.data, dm.data.conj().T)

    # 対角成分が実数
    assert np.allclose(dm.data.diagonal().imag, 0)


def test_densitymatrix_from_pure_states():
    """純粋状態からの密度行列生成テスト"""
    basis = LinMolBasis(V_max=1, J_max=0, use_M=False)  # 2状態系

    # 基底状態
    sv1 = StateVector(basis)
    sv1.set_state([0, 0])
    dm1 = DensityMatrix(basis)
    dm1.set_pure_state(sv1)

    # 密度行列の要素確認
    expected_dm1 = np.array([[1, 0], [0, 0]])
    np.testing.assert_array_almost_equal(dm1.data, expected_dm1)

    # 励起状態
    sv2 = StateVector(basis)
    sv2.set_state([1, 0])
    dm2 = DensityMatrix(basis)
    dm2.set_pure_state(sv2)

    expected_dm2 = np.array([[0, 0], [0, 1]])
    np.testing.assert_array_almost_equal(dm2.data, expected_dm2)

    # 重ね合わせ状態
    sv3 = StateVector(basis)
    sv3.data[0] = 1.0 / np.sqrt(2)
    sv3.data[1] = 1.0 / np.sqrt(2)
    dm3 = DensityMatrix(basis)
    dm3.set_pure_state(sv3)

    expected_dm3 = np.array([[0.5, 0.5], [0.5, 0.5]])
    np.testing.assert_array_almost_equal(dm3.data, expected_dm3)


def test_densitymatrix_coherence():
    """密度行列のコヒーレンスのテスト"""
    basis = LinMolBasis(V_max=1, J_max=0, use_M=False)

    # 位相付き重ね合わせ状態
    sv = StateVector(basis)
    sv.data[0] = 1.0 / np.sqrt(2)
    sv.data[1] = 1j / np.sqrt(2)  # 位相π/2

    dm = DensityMatrix(basis)
    dm.set_pure_state(sv)

    # 非対角成分（コヒーレンス）の確認
    assert np.isclose(dm.data[0, 1], -1j / 2)
    assert np.isclose(dm.data[1, 0], 1j / 2)

    # エルミート性確認
    assert np.allclose(dm.data, dm.data.conj().T)


def test_state_operations():
    """状態操作のテスト"""
    basis = LinMolBasis(V_max=1, J_max=1, use_M=False)

    # 複数状態の重ね合わせ
    sv = StateVector(basis)
    # 手動で重ね合わせを作成
    sv.data[0] = 0.5  # [0,0]
    sv.data[1] = 0.6  # [0,1]
    sv.data[2] = 0.3  # [1,0]
    sv.data[3] = 0.5  # [1,1]
    sv.normalize()

    original_norm = sv.norm()
    assert np.isclose(original_norm, 1.0)

    # コピーの独立性
    sv_copy = sv.copy()
    sv_copy.data[0] = 0
    assert not np.allclose(sv.data, sv_copy.data)


def test_density_matrix_mixed_states():
    """混合状態の密度行列テスト"""
    basis = LinMolBasis(V_max=1, J_max=0, use_M=False)

    # 完全混合状態（最大エントロピー）
    dm_mixed = DensityMatrix(basis)
    dm_mixed.set_diagonal([0.5, 0.5])

    # 純粋状態
    sv_pure = StateVector(basis)
    sv_pure.set_state([0, 0])
    dm_pure = DensityMatrix(basis)
    dm_pure.set_pure_state(sv_pure)

    # 純度の計算（Tr(ρ²)）
    purity_mixed = np.trace(dm_mixed.data @ dm_mixed.data).real
    purity_pure = np.trace(dm_pure.data @ dm_pure.data).real

    # 純粋状態の純度は1、混合状態は1未満
    assert np.isclose(purity_pure, 1.0)
    assert purity_mixed < 1.0
    assert np.isclose(purity_mixed, 0.5)  # 2状態の完全混合


def test_error_handling():
    """エラーハンドリングのテスト"""
    basis = LinMolBasis(V_max=1, J_max=0, use_M=False)

    # StateVectorエラー
    sv = StateVector(basis)
    with pytest.raises(ValueError):
        sv.set_state([2, 0])  # 存在しない状態

    # DensityMatrixエラー
    dm = DensityMatrix(basis)
    with pytest.raises(ValueError):
        dm.set_diagonal([1, 2, 3])  # 長さ不一致


def test_large_system():
    """大きなシステムでのテスト"""
    basis = LinMolBasis(V_max=3, J_max=3, use_M=False)

    sv = StateVector(basis)
    dm = DensityMatrix(basis)

    # サイズ確認
    expected_size = 4 * 4  # (V_max+1) * (J_max+1)
    assert sv.data.size == expected_size
    assert dm.data.shape == (expected_size, expected_size)

    # ランダム状態の設定
    sv.set_state([2, 1])  # V=2, J=1
    assert np.isclose(sv.norm(), 1.0)

    # 統計分布設定
    populations = np.random.random(expected_size)
    populations /= populations.sum()  # 正規化
    dm.set_diagonal(populations)
    assert np.isclose(dm.trace(), 1.0)


def test_state_vector_complex_coefficients():
    """複素係数を持つ状態ベクトルのテスト"""
    basis = LinMolBasis(V_max=1, J_max=1, use_M=False)
    sv = StateVector(basis)

    # 複素係数で状態を手動設定
    sv.data[0] = 0.6 + 0.2j
    sv.data[1] = 0.3 - 0.1j
    sv.data[2] = -0.4 + 0.5j
    sv.data[3] = 0.1 + 0.7j
    sv.normalize()

    # ノルムは実数で1
    norm = sv.norm()
    assert np.isclose(norm, 1.0)
    assert np.isreal(norm)

    # 密度行列への変換
    dm = DensityMatrix(basis)
    dm.set_pure_state(sv)

    # トレースは1
    assert np.isclose(dm.trace(), 1.0)

    # エルミート性
    assert np.allclose(dm.data, dm.data.conj().T)
