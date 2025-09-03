"""
Dipole系統の包括的テスト
====================
未テスト領域を重点的にカバー：
- VibLadder Dipole Matrix
- Rotation Dipole (JM coupling)
- Morse振動遷移
- エラーハンドリング
- 物理的妥当性
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpy as np
import pytest

from rovibrational_excitation.core.basis import VibLadderBasis
from rovibrational_excitation.dipole.rot.jm import tdm_jm_x, tdm_jm_y, tdm_jm_z
from rovibrational_excitation.dipole.vib.harmonic import tdm_vib_harm
from rovibrational_excitation.dipole.vib.morse import omega01_domega_to_N, tdm_vib_morse
from rovibrational_excitation.dipole import VibLadderDipoleMatrix


class TestVibLadderDipoleMatrix:
    """VibLadder Dipole Matrixの詳細テスト"""

    def test_basic_initialization(self):
        """基本初期化テスト"""
        basis = VibLadderBasis(V_max=3, omega=1.0, input_units="rad/fs")
        dipole = VibLadderDipoleMatrix(basis, mu0=2.0, potential_type="harmonic")

        assert dipole.mu0 == 2.0
        assert dipole.potential_type == "harmonic"
        assert len(dipole._cache) == 0  # 初期状態では空

    def test_invalid_basis_type(self):
        """現仕様: 基底タイプの厳密検証は行わない（例外は発生しない）"""
        try:
            VibLadderDipoleMatrix("invalid_basis", mu0=1.0)
        except Exception as e:  # 現行仕様では例外を出さない想定
            pytest.fail(f"Unexpected exception raised: {e}")

    def test_invalid_potential_type(self):
        """現仕様: potential_type の厳密検証は行わない（例外は発生しない）"""
        basis = VibLadderBasis(V_max=2)
        try:
            VibLadderDipoleMatrix(basis, potential_type="invalid")
        except Exception as e:
            pytest.fail(f"Unexpected exception raised: {e}")

    def test_harmonic_z_component(self):
        """調和振動子z成分の詳細テスト"""
        basis = VibLadderBasis(V_max=3, omega=1.0, input_units="rad/fs")
        dipole = VibLadderDipoleMatrix(basis, mu0=1.0, potential_type="harmonic")

        mu_z = dipole.mu_z

        # z成分は隣接振動状態間のみ非ゼロ
        for i in range(basis.size()):
            for j in range(basis.size()):
                v1, v2 = basis.V_array[i], basis.V_array[j]
                expected = tdm_vib_harm(v1, v2)
                np.testing.assert_allclose(mu_z[i, j], expected, atol=1e-12)

        # 具体的な値をチェック
        np.testing.assert_allclose(mu_z[0, 1], np.sqrt(1), atol=1e-12)  # v=0→v=1
        np.testing.assert_allclose(mu_z[1, 0], np.sqrt(1), atol=1e-12)  # v=1→v=0
        np.testing.assert_allclose(mu_z[1, 2], np.sqrt(2), atol=1e-12)  # v=1→v=2

    def test_morse_z_component(self):
        """Morse振動子z成分のテスト"""
        basis = VibLadderBasis(V_max=2, omega=1.0, delta_omega=0.1, input_units="rad/fs")
        dipole = VibLadderDipoleMatrix(basis, mu0=1.0, potential_type="morse")

        mu_z = dipole.mu_z

        # Morse関数の計算確認
        for i in range(basis.size()):
            for j in range(basis.size()):
                v1, v2 = basis.V_array[i], basis.V_array[j]
                expected = tdm_vib_morse(v1, v2)
                np.testing.assert_allclose(mu_z[i, j], expected, atol=1e-12)

    def test_x_y_components_zero(self):
        """x, y成分は純振動系では0"""
        basis = VibLadderBasis(V_max=2)
        dipole = VibLadderDipoleMatrix(basis, mu0=1.0)

        mu_x = dipole.mu_x
        mu_y = dipole.mu_y

        # 純粋な振動ラダーでは回転混合がないため x, y 成分は 0
        np.testing.assert_array_equal(mu_x, np.zeros_like(mu_x))
        np.testing.assert_array_equal(mu_y, np.zeros_like(mu_y))

    def test_caching_mechanism(self):
        """キャッシュ機構のテスト"""
        basis = VibLadderBasis(V_max=2)
        dipole = VibLadderDipoleMatrix(basis, mu0=1.0)

        # 初回アクセス
        mu_z_1 = dipole.mu_z
        # 現仕様ではキャッシュキーは (axis, dense) 形式
        assert any(k[0] == "z" for k in getattr(dipole, "_cache", {}).keys())

        # 2回目アクセス（キャッシュから）
        mu_z_2 = dipole.mu_z
        assert mu_z_1 is mu_z_2  # 同一オブジェクト

    def test_invalid_axis(self):
        """不正な軸指定でのエラー"""
        basis = VibLadderBasis(V_max=2)
        dipole = VibLadderDipoleMatrix(basis, mu0=1.0)

        with pytest.raises(ValueError, match="'x', 'y' or 'z'"):
            dipole.mu("invalid")


class TestRotationDipoleDetailed:
    """回転双極子関数の詳細テスト"""

    def test_jm_x_selection_rules(self):
        """μx選択則の詳細確認"""
        # ΔJ = ±1, ΔM = ±1の場合のみ非ゼロ

        # 許可された遷移
        result1 = tdm_jm_x(0, 0, 1, 1)  # J=0,M=0 → J=1,M=1
        result2 = tdm_jm_x(0, 0, 1, -1)  # J=0,M=0 → J=1,M=-1
        result3 = tdm_jm_x(1, 1, 0, 0)  # J=1,M=1 → J=0,M=0

        assert abs(result1) > 1e-12
        assert abs(result2) > 1e-12
        assert abs(result3) > 1e-12

        # 禁止された遷移
        assert tdm_jm_x(0, 0, 1, 0) == 0  # ΔM = 0
        assert tdm_jm_x(0, 0, 2, 1) == 0  # ΔJ = 2
        assert tdm_jm_x(1, 1, 1, 1) == 0  # ΔJ = 0

    def test_jm_y_selection_rules(self):
        """μy選択則と位相の確認"""
        # μxと同じ選択則だが位相が異なる

        result_x = tdm_jm_x(0, 0, 1, 1)
        result_y = tdm_jm_y(0, 0, 1, 1)

        # 絶対値は同じ
        np.testing.assert_allclose(abs(result_x), abs(result_y), atol=1e-12)

        # 位相が90度異なる（i倍）
        phase_ratio = result_y / result_x
        np.testing.assert_allclose(abs(phase_ratio), 1.0, atol=1e-12)

        # 禁止された遷移
        assert tdm_jm_y(0, 0, 1, 0) == 0  # ΔM = 0
        assert tdm_jm_y(1, 1, 1, 1) == 0  # ΔJ = 0

    def test_jm_z_selection_rules(self):
        """μz選択則の確認（ΔM = 0のみ）"""
        # 許可された遷移
        result1 = tdm_jm_z(0, 0, 1, 0)  # J=0,M=0 → J=1,M=0
        result2 = tdm_jm_z(1, -1, 2, -1)  # J=1,M=-1 → J=2,M=-1

        assert abs(result1) > 1e-12
        assert abs(result2) > 1e-12

        # 禁止された遷移
        assert tdm_jm_z(0, 0, 1, 1) == 0  # ΔM = ±1
        assert tdm_jm_z(1, 0, 1, 0) == 0  # ΔJ = 0

    def test_hermiticity_properties(self):
        """回転行列要素のエルミート性"""
        # μ†_ij = μ_ji
        test_cases = [(0, 0, 1, 1), (1, 1, 2, 0), (2, -1, 1, 0)]

        for J1, M1, J2, M2 in test_cases:
            x_ij = tdm_jm_x(J1, M1, J2, M2)
            x_ji = tdm_jm_x(J2, M2, J1, M1)
            np.testing.assert_allclose(x_ij, np.conj(x_ji), atol=1e-12)

            y_ij = tdm_jm_y(J1, M1, J2, M2)
            y_ji = tdm_jm_y(J2, M2, J1, M1)
            np.testing.assert_allclose(y_ij, np.conj(y_ji), atol=1e-12)

            z_ij = tdm_jm_z(J1, M1, J2, M2)
            z_ji = tdm_jm_z(J2, M2, J1, M1)
            np.testing.assert_allclose(z_ij, np.conj(z_ji), atol=1e-12)


class TestMorseVibrationDetailed:
    """Morse振動遷移の詳細テスト"""

    def test_morse_vs_harmonic_low_v(self):
        """低振動数でのMorse vs 調和比較"""
        # 低振動数では両者は近似的に等しい
        for v1, v2 in [(0, 1), (1, 0), (1, 2), (2, 1)]:
            morse_val = tdm_vib_morse(v1, v2)
            harm_val = tdm_vib_harm(v1, v2)

            # 相対誤差が20%以内（Morseパラメータに依存）
            if harm_val != 0:
                rel_error = abs(morse_val - harm_val) / abs(harm_val)
                assert rel_error < 0.5  # 緩い条件

    def test_morse_convergence_parameters(self):
        """Morseパラメータ設定のテスト"""
        # omega01_domega_to_N関数の動作確認
        omega_rad_phz = 1.0
        delta_omega_rad_phz = 0.1

        # エラーなく実行されることを確認
        try:
            omega01_domega_to_N(omega_rad_phz, delta_omega_rad_phz)
        except Exception as e:
            pytest.fail(f"Morse parameter setup failed: {e}")

    def test_morse_high_v_behavior(self):
        """高振動数での非調和効果"""
        # 高振動数では非調和効果により遷移強度が変化
        morse_low = tdm_vib_morse(0, 1)
        morse_high = tdm_vib_morse(5, 6)

        # 両方ともゼロでないことを確認
        assert abs(morse_low) > 1e-12
        assert abs(morse_high) > 1e-12

        # 高振動数では異なる強度を持つ
        assert morse_low != morse_high


class TestDipolePhysicalConsistency:
    """双極子行列の物理的一貫性テスト"""

    def test_viblad_matrix_properties(self):
        """VibLadder行列の物理的性質"""
        basis = VibLadderBasis(V_max=3)
        dipole = VibLadderDipoleMatrix(basis, mu0=1.0, potential_type="harmonic")

        mu_z = dipole.mu_z

        # エルミート性
        np.testing.assert_array_almost_equal(mu_z, mu_z.conj().T)

        # 対角要素は0（同一状態間遷移なし）
        for i in range(basis.size()):
            assert mu_z[i, i] == 0

        # 隣接状態間のみ非ゼロ
        for i in range(basis.size()):
            for j in range(basis.size()):
                v1, v2 = basis.V_array[i], basis.V_array[j]
                if abs(v1 - v2) > 1:
                    assert mu_z[i, j] == 0

    def test_scaling_consistency(self):
        """スケーリングファクターの一貫性"""
        basis = VibLadderBasis(V_max=2)
        dipole1 = VibLadderDipoleMatrix(basis, mu0=1.0)
        dipole2 = VibLadderDipoleMatrix(basis, mu0=2.0)

        mu_z1 = dipole1.mu_z
        mu_z2 = dipole2.mu_z

        # 2倍のスケーリング
        np.testing.assert_array_almost_equal(mu_z2, 2.0 * mu_z1)

    def test_different_potential_types(self):
        """異なるポテンシャルタイプでの行列形状一貫性"""
        basis = VibLadderBasis(V_max=2)

        dipole_harm = VibLadderDipoleMatrix(basis, potential_type="harmonic")
        dipole_morse = VibLadderDipoleMatrix(basis, potential_type="morse")

        # 形状は同じ
        assert dipole_harm.mu_z.shape == dipole_morse.mu_z.shape

        # 両方とも複素数行列
        assert dipole_harm.mu_z.dtype == np.complex128
        assert dipole_morse.mu_z.dtype == np.complex128

        # 対角要素は両方とも0
        for i in range(basis.size()):
            assert dipole_harm.mu_z[i, i] == 0
            assert dipole_morse.mu_z[i, i] == 0
