"""
低レベル伝播機の詳細テスト
========================
RK4 Schrödinger、RK4 Liouville-von Neumann、Split-Operatorの
数値精度と物理的妥当性を詳細に検証します。
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpy as np
import pytest

# 低レベル伝播機能をインポート
from rovibrational_excitation.core.propagation.algorithms.rk4.lvne import rk4_lvne, rk4_lvne_traj
from rovibrational_excitation.core.propagation.algorithms.rk4.schrodinger import rk4_schrodinger
from rovibrational_excitation.core.propagation.algorithms.split_operator.schrodinger import splitop_schrodinger

# CuPyが利用可能か判定
try:
    import cupy as cp  # noqa: F401

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


# =============================================================================
# テスト用ヘルパー関数
# =============================================================================


def create_two_level_system():
    """2レベルシステムのテストケースを生成"""
    H0 = np.diag([0.0, 1.0])
    mu_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    mu_y = np.zeros((2, 2), dtype=np.complex128)
    return H0, mu_x, mu_y


def create_harmonic_oscillator(n_levels=5):
    """調和振動子システムのテストケースを生成"""
    H0 = np.diag(np.arange(n_levels, dtype=float))

    # 昇降演算子
    a_dag = np.zeros((n_levels, n_levels), dtype=complex)
    a = np.zeros((n_levels, n_levels), dtype=complex)

    for i in range(n_levels - 1):
        a_dag[i, i + 1] = np.sqrt(i + 1)
        a[i + 1, i] = np.sqrt(i + 1)

    mu_x = (a_dag + a) / np.sqrt(2)
    mu_y = 1j * (a_dag - a) / np.sqrt(2)

    return H0, mu_x, mu_y


def create_electric_field(n_points, amplitude=0.1, frequency=1.0, phase=0.0):
    """テスト用電場を生成"""
    t = np.linspace(0, 2 * np.pi, n_points)
    Ex = amplitude * np.cos(frequency * t + phase)
    Ey = amplitude * np.sin(frequency * t + phase) * 0.5
    return Ex, Ey


def create_gaussian_pulse(n_points, amplitude=0.1, center=None, width=1.0):
    """ガウシアンパルス電場を生成"""
    if center is None:
        center = n_points // 2

    t = np.arange(n_points)
    pulse = amplitude * np.exp(-(((t - center) / width) ** 2))

    Ex = pulse
    Ey = pulse * 0.3

    return Ex, Ey


def compute_energy(psi, H):
    """エネルギー期待値を計算"""
    return np.real(np.conj(psi) @ H @ psi)


def compute_energy_density_matrix(rho, H):
    """密度行列からエネルギー期待値を計算"""
    return np.real(np.trace(rho @ H))


# =============================================================================
# RK4 Schrödinger詳細テスト
# =============================================================================


class TestRK4SchrodingerDetailed:
    """RK4 Schrödinger伝播の詳細テスト"""

    def test_norm_conservation_precision(self):
        """高精度ノルム保存の確認"""
        H0, mu_x, mu_y = create_two_level_system()
        Ex, Ey = create_electric_field(21, amplitude=0.1)

        psi0 = np.array([0.6, 0.8], dtype=complex)

        traj = rk4_schrodinger(H0, mu_x, mu_y, Ex, Ey, psi0, dt=0.05, return_traj=True)

        for i, psi in enumerate(traj):
            norm = np.linalg.norm(psi)
            np.testing.assert_allclose(
                norm, 1.0, atol=1e-12, err_msg=f"Step {i}: norm = {norm}"
            )

    def test_energy_conservation_free_evolution(self):
        """自由発展時のエネルギー保存"""
        H0, mu_x, mu_y = create_harmonic_oscillator(4)

        # ゼロ電場
        Ex = np.zeros(21)
        Ey = np.zeros(21)

        psi0 = np.array([0.5, 0.6, 0.6, 0], dtype=complex)
        psi0 /= np.linalg.norm(psi0)

        traj = rk4_schrodinger(H0, mu_x, mu_y, Ex, Ey, psi0, dt=0.1, return_traj=True)

        energies = [compute_energy(psi, H0) for psi in traj]

        energy_variation = np.max(energies) - np.min(energies)
        assert energy_variation < 1e-5, f"エネルギー変動: {energy_variation}"

    def test_time_reversal_symmetry(self):
        """時間可逆性の確認"""
        H0, mu_x, mu_y = create_two_level_system()
        Ex, Ey = create_gaussian_pulse(21, amplitude=0.1, width=3.0)

        psi0 = np.array([1, 0], dtype=complex)
        dt = 0.1

        # 前進伝播
        traj_forward = rk4_schrodinger(
            H0, mu_x, mu_y, Ex, Ey, psi0, dt, return_traj=True
        )

        # 後進伝播（電場と時間ステップを反転）
        Ex_rev = Ex[::-1]
        Ey_rev = Ey[::-1]
        psi_end = traj_forward[-1]

        traj_backward = rk4_schrodinger(
            H0, mu_x, mu_y, Ex_rev, Ey_rev, psi_end, -dt, return_traj=True
        )

        # 初期状態への復帰を確認
        psi_returned = traj_backward[-1]
        overlap = np.abs(np.vdot(psi0, psi_returned)) ** 2

        np.testing.assert_allclose(
            overlap, 1.0, atol=1e-6, err_msg=f"時間可逆性誤差: {1 - overlap}"
        )

    @pytest.mark.skipif(
        not HAS_CUPY, reason="CuPyがインストールされていないためスキップ"
    )
    def test_backend_consistency(self):
        """NumPy vs CuPyバックエンドの一貫性"""
        H0, mu_x, mu_y = create_two_level_system()
        Ex, Ey = create_electric_field(11, amplitude=0.05)

        psi0 = np.array([0.7, 0.7], dtype=complex)
        psi0 /= np.linalg.norm(psi0)

        # NumPy
        result_numpy = rk4_schrodinger(
            H0, mu_x, mu_y, Ex, Ey, psi0, dt=0.1, return_traj=False, backend="numpy"
        )

        # CuPy
        result_cupy = rk4_schrodinger(
            H0, mu_x, mu_y, Ex, Ey, psi0, dt=0.1, return_traj=False, backend="cupy"
        )

        np.testing.assert_allclose(result_numpy[0], result_cupy[0], atol=1e-12)

    def test_renormalization_option(self):
        """再正規化オプションの効果"""
        H0, mu_x, mu_y = create_two_level_system()
        Ex, Ey = create_electric_field(11, amplitude=0.3)

        psi0 = np.array([1, 0], dtype=complex)

        # 再正規化あり
        traj_renorm = rk4_schrodinger(
            H0, mu_x, mu_y, Ex, Ey, psi0, dt=0.1, return_traj=True, renorm=True
        )

        # 再正規化なし
        traj_no_renorm = rk4_schrodinger(
            H0, mu_x, mu_y, Ex, Ey, psi0, dt=0.1, return_traj=True, renorm=False
        )

        # 再正規化ありの場合、全時刻でノルムが1
        for psi in traj_renorm:
            norm = np.linalg.norm(psi)
            np.testing.assert_allclose(norm, 1.0, atol=1e-12)

        # 再正規化なしでも適度な電場では大きくずれない
        norms = [np.linalg.norm(psi) for psi in traj_no_renorm]
        max_deviation = max(abs(norm - 1.0) for norm in norms)
        assert max_deviation < 0.1


# =============================================================================
# RK4 Liouville-von Neumann詳細テスト
# =============================================================================


class TestRK4LiouvilleVonNeumannDetailed:
    """RK4 Liouville-von Neumann伝播の詳細テスト"""

    def test_trace_conservation_precision(self):
        """トレース保存の高精度確認"""
        H0, mu_x, mu_y = create_two_level_system()
        Ex, Ey = create_electric_field(21, amplitude=0.1)

        # 一般的な混合状態
        rho0 = np.array([[0.7, 0.1 + 0.2j], [0.1 - 0.2j, 0.3]], dtype=complex)

        traj = rk4_lvne_traj(
            H0, mu_x, mu_y, Ex, Ey, rho0, dt=0.05, steps=10, sample_stride=1
        )

        initial_trace = np.trace(rho0)

        for i, rho in enumerate(traj):
            trace = np.trace(rho)
            np.testing.assert_allclose(
                trace, initial_trace, atol=1e-12, err_msg=f"Step {i}: trace = {trace}"
            )

    def test_hermiticity_preservation(self):
        """エルミート性保存の詳細確認"""
        H0, mu_x, mu_y = create_harmonic_oscillator(3)
        Ex, Ey = create_electric_field(11, amplitude=0.1)

        # 初期エルミート密度行列
        rho0 = np.array(
            [
                [0.5, 0.1 + 0.2j, 0.0],
                [0.1 - 0.2j, 0.3, 0.05 + 0.1j],
                [0.0, 0.05 - 0.1j, 0.2],
            ],
            dtype=complex,
        )

        traj = rk4_lvne_traj(
            H0, mu_x, mu_y, Ex, Ey, rho0, dt=0.1, steps=5, sample_stride=1
        )

        for i, rho in enumerate(traj):
            hermiticity_error = np.max(np.abs(rho - rho.conj().T))
            error_msg = f"Step {i}: エルミート性誤差 = {hermiticity_error}"
            assert hermiticity_error < 1e-12, error_msg

    def test_trajectory_vs_final_consistency(self):
        """軌跡版と最終結果版の一貫性"""
        H0, mu_x, mu_y = create_two_level_system()
        Ex, Ey = create_electric_field(11, amplitude=0.08)

        rho0 = np.eye(2, dtype=complex) / 2
        dt = 0.1
        steps = 5

        # 軌跡版
        traj = rk4_lvne_traj(H0, mu_x, mu_y, Ex, Ey, rho0, dt, steps, sample_stride=1)

        # 最終結果版
        rho_final = rk4_lvne(H0, mu_x, mu_y, Ex, Ey, rho0, dt, steps)

        np.testing.assert_allclose(traj[-1], rho_final, atol=1e-14)


# =============================================================================
# Split-Operator詳細テスト
# =============================================================================


class TestSplitOperatorDetailed:
    """Split-Operator伝播の詳細テスト"""

    def test_norm_conservation(self):
        """ノルム保存の確認"""
        H0, mu_x, mu_y = create_two_level_system()
        pol = np.array([1.0, 0.0], dtype=np.float64)
        # 奇数長の電場配列を作成（split operatorの要求に合わせる）
        Efield = np.concatenate([np.zeros(5), create_gaussian_pulse(11, amplitude=0.1)[0], np.zeros(5)])

        psi0 = np.array([0.6, 0.8], dtype=complex)

        traj = splitop_schrodinger(H0, mu_x, mu_y, pol, Efield, psi0, dt=0.1, 
                                   return_traj=True, sample_stride=1)

        for i in range(traj.shape[0]):
            norm = np.linalg.norm(traj[i])
            np.testing.assert_allclose(norm, 1.0, atol=1e-12)

    def test_energy_conservation_free_evolution(self):
        """自由発展時のエネルギー保存"""
        H0, mu_x, mu_y = create_harmonic_oscillator(4)
        pol = np.array([0.0, 0.0], dtype=np.float64)  # ゼロ偏光
        Efield = np.zeros(21)  # 奇数長

        psi0 = np.array([0.5, 0.6, 0.6, 0], dtype=complex)
        psi0 /= np.linalg.norm(psi0)

        traj = splitop_schrodinger(H0, mu_x, mu_y, pol, Efield, psi0, dt=0.1,
                                   return_traj=True, sample_stride=1)

        energies = [compute_energy(traj[i], H0) for i in range(traj.shape[0])]

        energy_variation = np.max(energies) - np.min(energies)
        assert energy_variation < 1e-12

    @pytest.mark.skipif(
        not HAS_CUPY, reason="CuPyがインストールされていないためスキップ"
    )
    def test_cupy_backend_consistency(self):
        """CuPyバックエンドの一貫性"""
        H0, mu_x, mu_y = create_two_level_system()
        pol = np.array([1.0, 0.3], dtype=np.float64)
        # 奇数長の電場配列を作成
        Efield = np.concatenate([np.zeros(5), create_gaussian_pulse(11, amplitude=0.1)[0], np.zeros(5)])

        psi0 = np.array([0.7, 0.7], dtype=complex)
        psi0 /= np.linalg.norm(psi0)

        # NumPyバックエンド
        traj_numpy = splitop_schrodinger(
            H0, mu_x, mu_y, pol, Efield, psi0, dt=0.1, return_traj=True, sample_stride=1, backend="numpy"
        )

        # CuPyバックエンド
        traj_cupy = splitop_schrodinger(
            H0, mu_x, mu_y, pol, Efield, psi0, dt=0.1, return_traj=True, sample_stride=1, backend="cupy"
        )

        np.testing.assert_allclose(traj_numpy, traj_cupy, atol=1e-12)


# =============================================================================
# エラーハンドリングテスト
# =============================================================================


class TestErrorHandling:
    """エラーハンドリングのテスト"""

    def test_invalid_array_shapes(self):
        """不正な配列形状でのエラー"""
        H0 = np.diag([0, 1])
        mu_x = np.array([[0, 1], [1, 0]], dtype=complex)
        mu_y = np.zeros((2, 2), dtype=complex)

        # サイズ不一致
        psi0_wrong = np.array([1, 0, 0], dtype=complex)
        Ex, Ey = create_electric_field(11)

        with pytest.raises((ValueError, IndexError)):
            rk4_schrodinger(H0, mu_x, mu_y, Ex, Ey, psi0_wrong, dt=0.1)

    def test_field_length_validation(self):
        """電場長の検証"""
        H0, mu_x, mu_y = create_two_level_system()
        psi0 = np.array([1, 0], dtype=complex)

        # 空の電場配列でエラーが発生することを確認
        with pytest.raises((ValueError, IndexError)):
            rk4_schrodinger(
                H0, mu_x, mu_y, np.array([]), np.array([]), psi0, dt=0.1
            )


# =============================================================================
# アルゴリズム比較テスト
# =============================================================================


class TestAlgorithmComparison:
    """異なるアルゴリズム間の比較テスト"""

    def test_rk4_vs_splitop_weak_field(self):
        """弱電場でのRK4とSplit-Operatorの比較"""
        H0, mu_x, mu_y = create_two_level_system()

        # 弱い電場
        Ex, Ey = create_gaussian_pulse(21, amplitude=0.01)
        pol = np.array([1.0, 0.5], dtype=np.float64)
        Efield = Ex  # x成分のみ使用

        psi0 = np.array([1, 0], dtype=complex)
        dt = 0.1

        # RK4 Schrödinger
        traj_rk4 = rk4_schrodinger(H0, mu_x, mu_y, Ex, Ey, psi0, dt, return_traj=True)

        # Split-Operator
        traj_splitop = splitop_schrodinger(
            H0, mu_x, mu_y, pol, Efield, psi0, dt, return_traj=True, sample_stride=1
        )

        # 弱電場では結果が近いはず
        psi_rk4_final = traj_rk4[-1]
        psi_splitop_final = traj_splitop[-1]

        overlap = np.abs(np.vdot(psi_rk4_final, psi_splitop_final)) ** 2
        assert overlap > 0.95  # 95%以上の重なり


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
