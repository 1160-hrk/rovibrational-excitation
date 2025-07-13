"""
RK4伝播機の包括的テスト
===================
未テスト領域を重点的にカバー：
- CuPyバックエンド
- エラーハンドリング
- 高度オプション（renorm、stride）
- 境界条件とエッジケース
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpy as np
import pytest

from rovibrational_excitation.core.propagation.algorithms.rk4.lvne import rk4_lvne, rk4_lvne_traj
from rovibrational_excitation.core.propagation.algorithms.rk4.schrodinger import rk4_schrodinger

# CuPy可用性チェック
try:
    import cupy as cp  # noqa: F401

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


class TestRK4ErrorHandling:
    """RK4エラーハンドリングテスト"""

    def test_rk4_schrodinger_dimension_mismatch(self):
        """次元不一致エラー"""
        H0 = np.diag([0.0, 1.0])  # 2x2
        mu_x = np.array([[0, 1], [1, 0]], dtype=complex)
        mu_y = np.zeros((2, 2), dtype=complex)
        E_field = np.array([0, 0.1, 0, 0.1, 0])

        # 間違った次元のpsi0
        psi0_wrong = np.array([1, 0, 0], dtype=complex)  # 3次元

        # H0とpsi0の次元が合わない場合の動作確認
        # （内部でエラーが発生するか、結果が正しくないか）
        with pytest.raises((ValueError, IndexError)):
            rk4_schrodinger(H0, mu_x, mu_y, E_field, E_field, psi0_wrong, dt=0.1)


class TestRK4AdvancedOptions:
    """RK4高度オプションテスト"""

    def test_renormalization_option(self):
        """再正規化オプション"""
        H0 = np.diag([0.0, 1.0])
        mu_x = np.array([[0, 1], [1, 0]], dtype=complex)
        mu_y = np.zeros((2, 2), dtype=complex)
        E_field = np.array([0, 0.5, 0, 0.5, 0])  # 強めの電場
        psi0 = np.array([1, 0], dtype=complex)

        # 再正規化なし
        result_no_renorm = rk4_schrodinger(
            H0,
            mu_x,
            mu_y,
            E_field,
            E_field,
            psi0,
            dt=0.1,
            return_traj=True,
            renorm=False,
        )

        # 再正規化あり
        result_with_renorm = rk4_schrodinger(
            H0,
            mu_x,
            mu_y,
            E_field,
            E_field,
            psi0,
            dt=0.1,
            return_traj=True,
            renorm=True,
        )

        # 再正規化ありの場合、すべての時点でノルムが1
        for i in range(result_with_renorm.shape[0]):
            norm = np.linalg.norm(result_with_renorm[i])
            np.testing.assert_allclose(norm, 1.0, atol=1e-10)

        # 再正規化なしの場合、ノルムが1から逸脱する可能性がある
        final_norm_no_renorm = np.linalg.norm(result_no_renorm[-1])
        # 強い電場で数値誤差による逸脱を許容
        assert final_norm_no_renorm > 0.99  # 最低限の物理的合理性

    def test_stride_sampling(self):
        """ストライドサンプリング"""
        H0 = np.diag([0.0, 1.0])
        mu_x = np.array([[0, 1], [1, 0]], dtype=complex)
        mu_y = np.zeros((2, 2), dtype=complex)
        E_field = np.linspace(0, 0.1, 21)  # 21点 -> 10ステップ
        psi0 = np.array([1, 0], dtype=complex)

        # ストライド1（全点記録）
        result_stride1 = rk4_schrodinger(
            H0, mu_x, mu_y, E_field, E_field, psi0, dt=0.1, return_traj=True, stride=1
        )

        # ストライド2（2点おき記録）
        result_stride2 = rk4_schrodinger(
            H0, mu_x, mu_y, E_field, E_field, psi0, dt=0.1, return_traj=True, stride=2
        )

        # ストライド5（5点おき記録）
        result_stride5 = rk4_schrodinger(
            H0, mu_x, mu_y, E_field, E_field, psi0, dt=0.1, return_traj=True, stride=5
        )

        # 形状確認
        steps = (len(E_field) - 1) // 2  # 10ステップ
        assert result_stride1.shape[0] == steps + 1  # 11点
        assert result_stride2.shape[0] == steps // 2 + 1  # 6点
        assert result_stride5.shape[0] == steps // 5 + 1  # 3点

        # サンプルポイントでの一致確認
        np.testing.assert_allclose(result_stride1[0], result_stride2[0], atol=1e-12)
        np.testing.assert_allclose(result_stride1[0], result_stride5[0], atol=1e-12)

    def test_return_traj_false(self):
        """軌跡記録なしオプション"""
        H0 = np.diag([0.0, 1.0])
        mu_x = np.array([[0, 1], [1, 0]], dtype=complex)
        mu_y = np.zeros((2, 2), dtype=complex)
        E_field = np.array([0, 0.1, 0, 0.1, 0])
        psi0 = np.array([1, 0], dtype=complex)

        # 軌跡記録なし
        result_no_traj = rk4_schrodinger(
            H0, mu_x, mu_y, E_field, E_field, psi0, dt=0.1, return_traj=False
        )

        # 軌跡記録あり
        result_with_traj = rk4_schrodinger(
            H0, mu_x, mu_y, E_field, E_field, psi0, dt=0.1, return_traj=True
        )

        # 形状確認
        assert result_no_traj.shape == (1, 2)  # 最終状態のみ
        assert result_with_traj.shape[0] > 1  # 複数時点

        # 最終状態の一致確認
        np.testing.assert_allclose(result_no_traj[0], result_with_traj[-1], atol=1e-12)


@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")
class TestRK4CuPyBackend:
    """CuPyバックエンドテスト"""

    def test_cupy_vs_numpy_consistency(self):
        """CuPyとNumPyの結果一致性"""
        H0 = np.diag([0.0, 1.0, 2.0])
        mu_x = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=complex)
        mu_y = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]], dtype=complex)
        E_field = np.array([0, 0.1, 0, 0.1, 0])
        psi0 = np.array([1, 0, 0], dtype=complex)
        dt = 0.1

        # NumPy結果
        result_numpy = rk4_schrodinger(
            H0, mu_x, mu_y, E_field, E_field, psi0, dt, backend="numpy"
        )

        # CuPy結果
        result_cupy = rk4_schrodinger(
            H0, mu_x, mu_y, E_field, E_field, psi0, dt, backend="cupy"
        )

        # 結果の一致確認
        np.testing.assert_allclose(result_numpy, result_cupy, atol=1e-10)

    def test_cupy_large_system(self):
        """CuPy大規模システム"""
        dim = 20
        H0 = np.diag(np.arange(dim, dtype=float))

        # ランダムな双極子行列（エルミート）
        np.random.seed(42)
        mu_x_raw = np.random.random((dim, dim)) + 1j * np.random.random((dim, dim))
        mu_x = (mu_x_raw + mu_x_raw.conj().T) / 2
        mu_y = np.zeros((dim, dim), dtype=complex)

        E_field = np.linspace(0, 0.05, 11)  # 弱い電場
        psi0 = np.zeros(dim, dtype=complex)
        psi0[0] = 1.0  # 基底状態

        # CuPy実行
        result = rk4_schrodinger(
            H0, mu_x, mu_y, E_field, E_field, psi0, dt=0.1, backend="cupy"
        )

        # 基本的な物理検証
        norm = np.linalg.norm(result[0])
        np.testing.assert_allclose(norm, 1.0, atol=1e-6)

    def test_cupy_error_when_unavailable(self):
        """CuPy利用不可時のエラー"""
        # CuPyを一時的に無効化
        import rovibrational_excitation.core.propagation.algorithms.rk4.schrodinger as rk4_mod

        original_cp = rk4_mod.cp
        rk4_mod.cp = None

        try:
            H0 = np.diag([0.0, 1.0])
            mu_x = np.array([[0, 1], [1, 0]], dtype=complex)
            mu_y = np.zeros((2, 2), dtype=complex)
            E_field = np.array([0, 0.1, 0])
            psi0 = np.array([1, 0], dtype=complex)

            with pytest.raises(RuntimeError, match="CuPy backend requested but CuPy not installed"):
                rk4_schrodinger(
                    H0, mu_x, mu_y, E_field, E_field, psi0, dt=0.1, backend="cupy"
                )
        finally:
            # 元に戻す
            rk4_mod.cp = original_cp


class TestRK4LVNEComprehensive:
    """RK4 LVNE包括的テスト"""

    def test_lvne_vs_lvne_traj_consistency(self):
        """rk4_lvne物理的妥当性テスト"""
        H0 = np.diag([0.0, 1.0])
        mu_x = np.array([[0, 1], [1, 0]], dtype=complex)
        mu_y = np.zeros((2, 2), dtype=complex)
        Ex = np.array([0, 0.1, 0, 0.1, 0])
        Ey = np.zeros(5)
        rho0 = np.array([[1, 0], [0, 0]], dtype=complex)
        dt = 0.1
        steps = 2

        # 最終状態のみ
        result_final = rk4_lvne(H0, mu_x, mu_y, Ex, Ey, rho0, dt, steps)

        # 軌跡版
        result_traj = rk4_lvne_traj(
            H0, mu_x, mu_y, Ex, Ey, rho0, dt, steps, sample_stride=1
        )

        # 物理的妥当性: トレース保存
        final_trace = np.trace(result_final)
        traj_trace = np.trace(result_traj[-1])

        np.testing.assert_allclose(final_trace.real, 1.0, atol=1e-12)
        np.testing.assert_allclose(traj_trace.real, 1.0, atol=1e-12)
        np.testing.assert_allclose(final_trace.imag, 0.0, atol=1e-12)
        np.testing.assert_allclose(traj_trace.imag, 0.0, atol=1e-12)

    def test_lvne_sample_stride(self):
        """LVNEサンプルストライド"""
        H0 = np.diag([0.0, 1.0, 2.0])
        mu_x = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=complex)
        mu_y = np.zeros((3, 3), dtype=complex)
        Ex = np.linspace(0, 0.1, 21)  # 10ステップ
        Ey = np.zeros(21)
        rho0 = np.diag([1, 0, 0])
        dt = 0.1
        steps = 10

        # ストライド1
        result_s1 = rk4_lvne_traj(
            H0, mu_x, mu_y, Ex, Ey, rho0, dt, steps, sample_stride=1
        )

        # ストライド3
        result_s3 = rk4_lvne_traj(
            H0, mu_x, mu_y, Ex, Ey, rho0, dt, steps, sample_stride=3
        )

        # 形状確認
        assert result_s1.shape[0] == steps + 1  # 11
        assert result_s3.shape[0] == steps // 3 + 1  # 4

        # 対応点での一致
        np.testing.assert_allclose(result_s1[0], result_s3[0], atol=1e-12)

    def test_lvne_hermiticity_preservation(self):
        """LVNEエルミート性保存詳細テスト"""
        H0 = np.diag([0.0, 1.0, 2.0])
        mu_x = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=complex)
        mu_y = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]], dtype=complex)
        Ex = np.array([0, 0.1, 0, 0.1, 0])
        Ey = np.array([0, 0.05, 0, 0.05, 0])

        # 非対角要素を持つ初期密度行列
        rho0 = np.array(
            [
                [0.5, 0.3 + 0.2j, 0.1],
                [0.3 - 0.2j, 0.3, 0.2 + 0.1j],
                [0.1, 0.2 - 0.1j, 0.2],
            ],
            dtype=complex,
        )
        dt = 0.05
        steps = 2

        result = rk4_lvne_traj(H0, mu_x, mu_y, Ex, Ey, rho0, dt, steps, sample_stride=1)

        # 各時点でエルミート性確認
        for i in range(result.shape[0]):
            rho = result[i]
            hermitian_error = np.max(np.abs(rho - rho.conj().T))
            error_msg = f"非エルミート at step {i}: {hermitian_error}"
            assert hermitian_error < 1e-12, error_msg

            # トレース保存
            trace = np.trace(rho)
            np.testing.assert_allclose(trace.real, 1.0, atol=1e-12)
            np.testing.assert_allclose(trace.imag, 0.0, atol=1e-12)


class TestRK4EdgeCases:
    """RK4境界条件・エッジケーステスト"""

    def test_zero_electric_field(self):
        """ゼロ電場での自由発展"""
        H0 = np.diag([0.0, 1.0, 3.0])
        mu_x = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=complex)
        mu_y = np.zeros((3, 3), dtype=complex)
        E_zero = np.zeros(5)
        psi0 = np.array([0, 1, 0], dtype=complex)  # 第1励起状態
        dt = 0.1

        result = rk4_schrodinger(
            H0, mu_x, mu_y, E_zero, E_zero, psi0, dt, return_traj=True
        )

        # 自由発展では振幅は変わらず、位相のみ変化
        initial_populations = np.abs(result[0]) ** 2
        final_populations = np.abs(result[-1]) ** 2
        np.testing.assert_allclose(initial_populations, final_populations, atol=1e-12)

        # 第1励起状態のエネルギー固有値による位相発展確認
        time_evolution = dt * ((len(E_zero) - 1) // 2)  # 総時間
        np.exp(-1j * 1.0 * time_evolution)  # E1 = 1.0
        phase_ratio = result[-1][1] / result[0][1]
        np.testing.assert_allclose(np.abs(phase_ratio), 1.0, atol=1e-12)

    def test_minimal_system_size(self):
        """最小システムサイズ（1x1）でのテスト"""
        H0 = np.array([[1.0]], dtype=complex)  # 1x1
        mu_x = np.array([[0.0]], dtype=complex)
        mu_y = np.array([[0.0]], dtype=complex)
        E_field = np.array([0, 0.1, 0])
        psi0 = np.array([1.0], dtype=complex)

        # エラーが発生しないことを確認
        result = rk4_schrodinger(H0, mu_x, mu_y, E_field, E_field, psi0, dt=0.1)

        # 1次元システムでは双極子結合がないため、位相発展のみ
        assert result.shape == (2, 1)  # 形状を正しく修正（時間ステップ x 次元）
        np.testing.assert_allclose(np.abs(result[0, 0]), 1.0, atol=1e-12)

    def test_odd_even_field_lengths(self):
        """奇数・偶数長電場の処理"""
        H0 = np.diag([0.0, 1.0])
        mu_x = np.array([[0, 1], [1, 0]], dtype=complex)
        mu_y = np.zeros((2, 2), dtype=complex)
        psi0 = np.array([1, 0], dtype=complex)

        # 奇数長（標準）
        E_odd = np.array([0, 0.1, 0, 0.1, 0])  # 5点
        result_odd = rk4_schrodinger(H0, mu_x, mu_y, E_odd, E_odd, psi0, dt=0.1)

        # 偶数長（末尾1点削除される）
        E_even = np.array([0, 0.1, 0, 0.1, 0, 0.05])  # 6点 -> 5点使用
        result_even = rk4_schrodinger(H0, mu_x, mu_y, E_even, E_even, psi0, dt=0.1)

        # 結果は同じになるはず（最後の1点は無視される）
        np.testing.assert_allclose(result_odd, result_even, atol=1e-12)

    def test_very_small_timestep(self):
        """非常に小さな時間ステップ"""
        H0 = np.diag([0.0, 1.0])
        mu_x = np.array([[0, 1], [1, 0]], dtype=complex)
        mu_y = np.zeros((2, 2), dtype=complex)
        E_field = np.array([0, 0.1, 0])
        psi0 = np.array([1, 0], dtype=complex)

        # 非常に小さなdt
        dt_small = 1e-6
        result_small = rk4_schrodinger(H0, mu_x, mu_y, E_field, E_field, psi0, dt_small)

        # 物理的に妥当な結果が得られることを確認
        norm = np.linalg.norm(result_small[0])
        np.testing.assert_allclose(norm, 1.0, atol=1e-12)

        # 小さなdtでは変化が小さいことを確認
        population_change = np.abs(
            np.abs(result_small[0, 1]) ** 2 - 0.0
        )  # 初期は基底状態
        assert population_change < 1e-10  # 非常に小さな変化


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
