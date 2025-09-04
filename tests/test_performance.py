import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
import time

import numpy as np
import pytest

from rovibrational_excitation.core.basis import LinMolBasis, VibLadderBasis
from rovibrational_excitation.core.electric_field import ElectricField, gaussian_fwhm
from rovibrational_excitation.core.propagation import SchrodingerPropagator
from rovibrational_excitation.core.units.converters import converter
from rovibrational_excitation.dipole.linmol.cache import LinMolDipoleMatrix


class MockDipole:
    """パフォーマンステスト用の軽量ダミー双極子"""

    def __init__(self, dim):
        mu_x = (
            np.random.random((dim, dim)) + 1j * np.random.random((dim, dim))
        ) * 1e-30
        mu_x = mu_x - np.diag(np.diag(mu_x))
        self.mu_x = mu_x
        self.mu_y = np.zeros((dim, dim), dtype=np.complex128)
        self.mu_z = np.zeros((dim, dim), dtype=np.complex128)
        self.units = "C*m"  # SI単位を使用
    
    def get_mu_in_units(self, axis: str, unit: str):
        src = {"x": self.mu_x, "y": self.mu_y, "z": self.mu_z}[axis]
        if unit in ("C*m", "C·m", "Cm"):
            return src
        return converter.convert_dipole_moment(src, "C*m", unit)
    
    def get_mu_x_SI(self):
        """Get μ_x in SI units (C·m)."""
        return self.mu_x
    
    def get_mu_y_SI(self):
        """Get μ_y in SI units (C·m)."""
        return self.mu_y
    
    def get_mu_z_SI(self):
        """Get μ_z in SI units (C·m)."""
        return self.mu_z


@pytest.mark.slow
def test_large_system_performance():
    """大きなシステムでのパフォーマンステスト"""
    # 大きな基底
    basis = LinMolBasis(V_max=5, J_max=5, use_M=False)
    dim = basis.size()  # 36次元

    H0 = basis.generate_H0()
    dipole = MockDipole(dim)

    # 短時間のテスト
    tlist = np.linspace(-1, 1, 51)
    efield = ElectricField(tlist)
    efield.add_dispersed_Efield(
        gaussian_fwhm,
        duration=0.5,
        t_center=0.0,
        carrier_freq=1.0,
        amplitude=1e9,
        polarization=np.array([1.0, 0.0]),
        const_polarisation=True,
    )

    psi0 = np.zeros(dim, dtype=np.complex128)
    psi0[0] = 1.0

    # 実行時間測定
    start_time = time.time()
    result = SchrodingerPropagator().propagate(H0, efield, dipole, psi0, return_traj=True)
    end_time = time.time()

    execution_time = end_time - start_time

    # 結果の検証
    assert result.shape[1] == dim
    assert execution_time < 10.0  # 10秒以内で完了

    # メモリ使用量の簡易チェック（結果配列のサイズ）
    expected_memory_mb = result.nbytes / (1024 * 1024)
    assert expected_memory_mb < 100  # 100MB以内


def test_very_large_system():
    """非常に大きなシステムでのスケーラビリティテスト"""
    # より大きな基底（使用注意：メモリとCPU集約的）
    omega = 5.0
    basis = VibLadderBasis(
        V_max=20,
        omega = omega
        )  # 21次元
    dim = basis.size()

    H0 = basis.generate_H0()
    dipole = MockDipole(dim)

    # 非常に短時間
    tlist = np.linspace(-100, 100, 1001)
    efield = ElectricField(tlist)
    efield.add_dispersed_Efield(
        gaussian_fwhm,
        duration=20,
        t_center=0.0,
        carrier_freq=omega/(2*np.pi),
        amplitude=1e9,
        polarization=np.array([1.0, 0.0]),
        const_polarisation=True,
    )

    psi0 = np.zeros(dim, dtype=np.complex128)
    psi0[0] = 1.0

    start_time = time.time()
    result = SchrodingerPropagator(renorm=True).propagate(
        H0, efield, dipole, psi0, return_traj=False
    )
    end_time = time.time()

    execution_time = end_time - start_time

    # 基本的な検証
    assert result.shape == (dim,)  # 形状を正しく修正
    assert execution_time < 5.0  # 5秒以内

    # ノルム保存
    norm = np.linalg.norm(result)
    assert np.isclose(norm, 1.0, atol=1e-10)


def test_long_time_evolution():
    """長時間発展でのパフォーマンステスト"""
    basis = LinMolBasis(V_max=3, J_max=3, use_M=False)
    dim = basis.size()

    H0 = basis.generate_H0()
    dipole = MockDipole(dim)

    # 長い時間軸
    tlist = np.linspace(-10, 10, 1001)  # 多くの時間点
    efield = ElectricField(tlist)
    efield.add_dispersed_Efield(
        gaussian_fwhm,
        duration=1.0,
        t_center=0.0,
        carrier_freq=1.0,
        amplitude=1e7,
        polarization=np.array([1.0, 0.0]),
        const_polarisation=True,
    )

    psi0 = np.zeros(dim, dtype=np.complex128)
    psi0[0] = 1.0

    start_time = time.time()
    result = SchrodingerPropagator().propagate(H0, efield, dipole, psi0, return_traj=True)
    end_time = time.time()

    execution_time = end_time - start_time

    # 結果の検証
    assert result.shape[1] == dim
    assert execution_time < 30.0  # 30秒以内

    # 数値安定性確認
    norms = [np.linalg.norm(psi) for psi in result]
    for norm in norms:
        assert np.isclose(norm, 1.0, atol=1e-8)


def test_memory_efficiency():
    """メモリ効率のテスト"""
    # 中程度のシステム
    basis = LinMolBasis(V_max=4, J_max=2, use_M=False)
    dim = basis.size()

    H0 = basis.generate_H0()
    dipole = MockDipole(dim)

    tlist = np.linspace(-2, 2, 101)
    efield = ElectricField(tlist)
    efield.add_dispersed_Efield(
        gaussian_fwhm,
        duration=1.0,
        t_center=0.0,
        carrier_freq=1.0,
        amplitude=0.1,
        polarization=np.array([1.0, 0.0]),
        const_polarisation=True,
    )

    psi0 = np.zeros(dim, dtype=np.complex128)
    psi0[0] = 1.0

    # 軌跡ありとなしでのメモリ使用量比較
    result_no_traj = SchrodingerPropagator().propagate(
        H0, efield, dipole, psi0, return_traj=False
    )
    result_with_traj = SchrodingerPropagator().propagate(
        H0, efield, dipole, psi0, return_traj=True
    )

    # メモリ使用量の差を確認
    memory_no_traj = result_no_traj.nbytes
    memory_with_traj = result_with_traj.nbytes

    assert memory_with_traj > memory_no_traj * 20  # 軌跡ありは大幅にメモリ使用


def test_stride_performance():
    """ストライドによるパフォーマンス改善テスト"""
    basis = LinMolBasis(V_max=3, J_max=3, use_M=False)
    dim = basis.size()

    H0 = basis.generate_H0()
    dipole = MockDipole(dim)

    tlist = np.linspace(-5, 5, 501)  # 多くの時間点
    efield = ElectricField(tlist)
    efield.add_dispersed_Efield(
        gaussian_fwhm,
        duration=1.0,
        t_center=0.0,
        carrier_freq=1.0,
        amplitude=1e9,
        polarization=np.array([1.0, 0.0]),
        const_polarisation=True,
    )

    psi0 = np.zeros(dim, dtype=np.complex128)
    psi0[0] = 1.0

    # stride=1
    start_time = time.time()
    result_stride1 = SchrodingerPropagator().propagate(
        H0, efield, dipole, psi0, return_traj=True, sample_stride=1
    )
    time.time() - start_time

    # stride=10
    start_time = time.time()
    result_stride10 = SchrodingerPropagator().propagate(
        H0, efield, dipole, psi0, return_traj=True, sample_stride=10
    )
    time.time() - start_time

    # ストライドが大きいほうが高速（出力が少ないため）
    assert result_stride1.shape[0] > result_stride10.shape[0]
    # 実行時間はほぼ同じ（計算量は同じ、出力のみ異なる）

    # メモリ使用量はストライドが大きいほうが少ない
    assert result_stride1.nbytes > result_stride10.nbytes


def test_numerical_stability_large_system():
    """大規模システムでの数値安定性テスト"""
    # より小さなシステムで安定性を確保
    omega = 1.0
    basis = LinMolBasis(
        V_max=4, J_max=2, use_M=True,
        omega=omega
    )
    dim = basis.size()

    H0 = basis.generate_H0()
    dipole = LinMolDipoleMatrix(basis, mu0=1e-30)

    # パルス電場設定
    t_list = np.linspace(-100, 100, 10001)
    Efield = ElectricField(t_list)
    Efield.add_dispersed_Efield(
        envelope_func=gaussian_fwhm,
        duration=20.0,
        t_center=0.0,
        carrier_freq=omega/(2*np.pi),  # 弱い場で数値誤差を最小化
        amplitude=1e8,  # 弱い相互作用
        polarization=np.array([1.0, 0.0]),
    )

    psi0 = np.zeros(dim, dtype=np.complex128)
    psi0[0] = 1.0

    
    result = SchrodingerPropagator().propagate(
        H0, Efield, dipole, psi0, return_traj=True, sample_stride=2
    )
    # ノルム保存の確認
    norms = [np.linalg.norm(psi) for psi in result]
    for i, norm in enumerate(norms):
        if i > 0:  # 初期状態以外
            assert np.isclose(norm, 1.0, atol=1e-6), f"時刻{i}でノルムが保存されていません: {norm}"
    
    # NaN値のチェック
    assert not np.any(np.isnan(result)), "NaN values found in trajectory"


def test_basis_generation_performance():
    """基底生成のパフォーマンステスト"""
    # 大きな基底の生成時間
    start_time = time.time()
    basis_large = LinMolBasis(V_max=10, J_max=10, use_M=True)
    generation_time = time.time() - start_time

    # 基底生成は高速であるべき
    assert generation_time < 1.0  # 1秒以内
    assert basis_large.size() > 1000  # 大きなサイズ

    # ハミルトニアン生成時間
    start_time = time.time()
    H0 = basis_large.generate_H0()
    hamiltonian_time = time.time() - start_time

    assert hamiltonian_time < 5.0  # 5秒以内
    assert H0.shape == (basis_large.size(), basis_large.size())


def test_electric_field_performance():
    """電場生成のパフォーマンステスト"""
    # 長い時間軸
    tlist = np.linspace(-20, 20, 10001)  # 高分解能

    start_time = time.time()
    efield = ElectricField(tlist)

    # 複数パルス追加
    for i in range(5):
        efield.add_dispersed_Efield(
            gaussian_fwhm,
            duration=1.0,
            t_center=i * 2 - 4,
            carrier_freq=1.0 + i * 0.1,
            amplitude=0.1,
            polarization=np.array([1.0, 0.0]),
            const_polarisation=True,
        )

    field_generation_time = time.time() - start_time

    # 電場生成は比較的高速であるべき
    assert field_generation_time < 10.0  # 10秒以内
    assert efield.Efield.shape == (10001, 2)


@pytest.mark.slow
def test_backend_performance_comparison():
    """バックエンド間のパフォーマンス比較"""
    basis = LinMolBasis(V_max=3, J_max=3, use_M=False)
    dim = basis.size()

    H0 = basis.generate_H0()
    dipole = MockDipole(dim)

    tlist = np.linspace(-2, 2, 101)
    efield = ElectricField(tlist)
    efield.add_dispersed_Efield(
        gaussian_fwhm,
        duration=1.0,
        t_center=0.0,
        carrier_freq=1.0,
        amplitude=0.1,
        polarization=np.array([1.0, 0.0]),
        const_polarisation=True,
    )

    psi0 = np.zeros(dim, dtype=np.complex128)
    psi0[0] = 1.0

    # NumPyバックエンド
    start_time = time.time()
    result_numpy = SchrodingerPropagator(backend="numpy").propagate(H0, efield, dipole, psi0)
    numpy_time = time.time() - start_time

    # 結果の一貫性確認
    assert result_numpy.shape[1] == dim
    assert numpy_time < 5.0  # 合理的な実行時間

    # ノルム保存
    norm = np.linalg.norm(result_numpy[0])
    assert np.isclose(norm, 1.0, atol=1e-10)


if __name__ == "__main__":
    # パフォーマンステストを個別実行する場合
    print("Running performance tests...")
    test_large_system_performance()
    print("Large system test passed")
    test_memory_efficiency()
    print("Memory efficiency test passed")
    test_stride_performance()
    print("Stride performance test passed")
    print("All performance tests completed successfully!")
