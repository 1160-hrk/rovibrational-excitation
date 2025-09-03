"""
TwoLevel Dipole Matrixの包括的テスト
===============================
現在0%カバレッジの完全未テスト領域を網羅：
- 基本初期化とパラメータ設定
- 双極子行列の生成と物理的妥当性
- エラーハンドリング
- 異なるmu0値での動作
- 物理的対称性の確認
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpy as np

from rovibrational_excitation.core.basis import TwoLevelBasis
from rovibrational_excitation.dipole import TwoLevelDipoleMatrix


class TestTwoLevelDipoleMatrix:
    """TwoLevel Dipole Matrixの詳細テスト"""

    def test_basic_initialization(self):
        """基本初期化テスト"""
        basis = TwoLevelBasis()
        dipole = TwoLevelDipoleMatrix(basis, mu0=1.5)

        assert dipole.basis is basis
        assert dipole.mu0 == 1.5
        assert dipole.basis.size() == 2

    def test_mu_x_matrix_structure(self):
        """μx行列の構造テスト"""
        basis = TwoLevelBasis()
        dipole = TwoLevelDipoleMatrix(basis, mu0=2.0)

        mu_x = dipole.mu_x

        # 形状確認
        assert mu_x.shape == (2, 2)
        assert mu_x.dtype == np.complex128

        # 期待される構造: [[0, mu0], [mu0, 0]]
        expected = np.array([[0, 2.0], [2.0, 0]], dtype=np.complex128)
        np.testing.assert_array_equal(mu_x, expected)

    def test_mu_y_matrix_structure(self):
        """μy行列の構造テスト"""
        basis = TwoLevelBasis()
        dipole = TwoLevelDipoleMatrix(basis, mu0=1.0)

        mu_y = dipole.mu_y

        # 形状確認
        assert mu_y.shape == (2, 2)
        assert mu_y.dtype == np.complex128

        # 期待される構造: [[0, -i*mu0], [i*mu0, 0]]
        expected = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        np.testing.assert_array_equal(mu_y, expected)

    def test_mu_z_matrix_structure(self):
        """μz行列の構造テスト"""
        basis = TwoLevelBasis()
        dipole = TwoLevelDipoleMatrix(basis, mu0=3.0)

        mu_z = dipole.mu_z

        # 形状確認
        assert mu_z.shape == (2, 2)
        assert mu_z.dtype == np.complex128

        # 期待される構造: ゼロ行列（電子双極子遷移では通常ゼロ）
        expected = np.array([[0, 0], [0, 0]], dtype=np.complex128)
        np.testing.assert_array_equal(mu_z, expected)

    def test_hermiticity_properties(self):
        """エルミート性の確認"""
        basis = TwoLevelBasis()
        dipole = TwoLevelDipoleMatrix(basis, mu0=1.0)

        mu_x = dipole.mu_x
        mu_y = dipole.mu_y
        mu_z = dipole.mu_z

        # μx, μy, μzはすべてエルミート
        np.testing.assert_array_almost_equal(mu_x, mu_x.conj().T)
        np.testing.assert_array_almost_equal(mu_y, mu_y.conj().T)
        np.testing.assert_array_almost_equal(mu_z, mu_z.conj().T)

    def test_pauli_matrix_relations(self):
        """Pauli行列との関係確認"""
        basis = TwoLevelBasis()
        dipole = TwoLevelDipoleMatrix(basis, mu0=1.0)

        # Pauli行列
        sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        sigma_z = np.array([[0, 0], [0, 0]], dtype=np.complex128)  # ゼロ行列

        # 双極子行列はPauli行列のスケール版
        np.testing.assert_array_almost_equal(dipole.mu_x, sigma_x)
        np.testing.assert_array_almost_equal(dipole.mu_y, sigma_y)
        np.testing.assert_array_almost_equal(dipole.mu_z, sigma_z)

    def test_mu0_scaling(self):
        """mu0スケーリングの確認"""
        basis = TwoLevelBasis()

        dipole1 = TwoLevelDipoleMatrix(basis, mu0=1.0)
        dipole2 = TwoLevelDipoleMatrix(basis, mu0=3.5)

        # スケーリング比較
        scale = 3.5
        np.testing.assert_array_almost_equal(dipole2.mu_x, scale * dipole1.mu_x)
        np.testing.assert_array_almost_equal(dipole2.mu_y, scale * dipole1.mu_y)
        np.testing.assert_array_almost_equal(dipole2.mu_z, scale * dipole1.mu_z)

    def test_zero_mu0_behavior(self):
        """mu0=0での動作"""
        basis = TwoLevelBasis()
        dipole = TwoLevelDipoleMatrix(basis, mu0=0.0)

        # すべてがゼロ行列
        expected_x = np.array([[0, 0], [0, 0]], dtype=np.complex128)
        expected_y = np.array([[0, 0], [0, 0]], dtype=np.complex128)
        expected_z = np.array([[0, 0], [0, 0]], dtype=np.complex128)

        np.testing.assert_array_equal(dipole.mu_x, expected_x)
        np.testing.assert_array_equal(dipole.mu_y, expected_y)
        np.testing.assert_array_equal(dipole.mu_z, expected_z)

    def test_commutation_relations(self):
        """交換関係の確認"""
        basis = TwoLevelBasis()
        dipole = TwoLevelDipoleMatrix(basis, mu0=1.0)

        mu_x = dipole.mu_x
        mu_y = dipole.mu_y
        dipole.mu_z

        # [μx, μy] = 2i μz (ただしμz=0なので結果も特殊)
        comm_xy = mu_x @ mu_y - mu_y @ mu_x
        # 実際の計算: [[0,1],[1,0]] @ [[0,-i],[i,0]] - [[0,-i],[i,0]] @ [[0,1],[1,0]]
        # = [[i,0],[0,-i]] - [[-i,0],[0,i]] = [[2i,0],[0,-2i]]
        expected_xy = 2j * np.array([[1, 0], [0, -1]], dtype=np.complex128)
        np.testing.assert_array_almost_equal(comm_xy, expected_xy)

    def test_transition_matrix_elements(self):
        """遷移行列要素の物理的妥当性"""
        basis = TwoLevelBasis()
        dipole = TwoLevelDipoleMatrix(basis, mu0=1.0)

        # 基底状態|0⟩から励起状態|1⟩への遷移
        # ⟨0|μx|1⟩ = 1, ⟨1|μx|0⟩ = 1
        assert dipole.mu_x[0, 1] == 1.0
        assert dipole.mu_x[1, 0] == 1.0

        # ⟨0|μy|1⟩ = -i, ⟨1|μy|0⟩ = i
        assert dipole.mu_y[0, 1] == -1j
        assert dipole.mu_y[1, 0] == 1j

        # ⟨0|μz|0⟩ = 0, ⟨1|μz|1⟩ = 0 (ゼロ行列)
        assert dipole.mu_z[0, 0] == 0.0
        assert dipole.mu_z[1, 1] == 0.0
        assert dipole.mu_z[0, 1] == 0.0
        assert dipole.mu_z[1, 0] == 0.0
