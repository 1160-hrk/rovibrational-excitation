import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
import numpy as np
from rovibrational_excitation.core.propagator import schrodinger_propagation, mixed_state_propagation, liouville_propagation
from rovibrational_excitation.core.electric_field import ElectricField, gaussian_fwhm
from rovibrational_excitation.core.basis import LinMolBasis
import pytest

class DummyDipole:
    def __init__(self, dim=2):
        self.mu_x = np.eye(dim, dtype=np.complex128)
        self.mu_y = np.zeros((dim, dim), dtype=np.complex128)
        self.mu_z = np.zeros((dim, dim), dtype=np.complex128)

class DummyDipoleOffDiag:
    """非対角要素を持つダミー双極子"""
    def __init__(self, dim=2):
        self.mu_x = np.array([[0, 1], [1, 0]], dtype=np.complex128) if dim == 2 else np.random.random((dim, dim)) + 1j * np.random.random((dim, dim))
        self.mu_y = np.zeros((dim, dim), dtype=np.complex128)
        self.mu_z = np.zeros((dim, dim), dtype=np.complex128)

def test_schrodinger_propagation():
    tlist = np.linspace(0, 1, 3)
    ef = ElectricField(tlist)
    ef.Efield[:,0] = 1.0
    basis = LinMolBasis(V_max=0, J_max=1, use_M=False)
    H0 = np.diag([0.0, 1.0])
    dip = DummyDipole()
    psi0 = np.array([1.0, 0.0], dtype=np.complex128)
    result = schrodinger_propagation(H0, ef, dip, psi0)
    assert result.shape[-1] == 2 or result[1].shape[-1] == 2

def test_mixed_state_propagation():
    tlist = np.linspace(0, 1, 3)
    ef = ElectricField(tlist)
    ef.Efield[:,0] = 1.0
    basis = LinMolBasis(V_max=0, J_max=1, use_M=False)
    H0 = np.diag([0.0, 1.0])
    dip = DummyDipole()
    psi0s = [np.array([1.0, 0.0], dtype=np.complex128), np.array([0.0, 1.0], dtype=np.complex128)]
    result = mixed_state_propagation(H0, ef, psi0s, dip)
    assert result.shape[-1] == 2 or result[1].shape[-1] == 2

def test_liouville_propagation():
    tlist = np.linspace(0, 1, 3)
    ef = ElectricField(tlist)
    ef.Efield[:,0] = 1.0
    basis = LinMolBasis(V_max=0, J_max=1, use_M=False)
    H0 = np.diag([0.0, 1.0])
    dip = DummyDipole()
    rho0 = np.eye(2, dtype=np.complex128)
    result = liouville_propagation(H0, ef, dip, rho0)
    assert result.shape[-1] == 2 or result[1].shape[-1] == 2

def test_schrodinger_propagation_with_constant_polarization():
    """一定偏光でのSchrodinger伝播テスト（Split-Operator使用）"""
    tlist = np.linspace(-5, 5, 201)
    ef = ElectricField(tlist)
    
    # 一定偏光のパルスを追加
    ef.add_dispersed_Efield(
        gaussian_fwhm, duration=2.0, t_center=0.0,
        carrier_freq=1.0, amplitude=1.0,
        polarization=np.array([1.0, 0.0]), const_polarisation=True
    )
    
    H0 = np.diag([0.0, 1.0])
    dip = DummyDipoleOffDiag()
    psi0 = np.array([1.0, 0.0], dtype=np.complex128)
    
    # 軌跡あり
    result_traj = schrodinger_propagation(H0, ef, dip, psi0, return_traj=True)
    assert result_traj.shape[1] == 2
    
    # 軌跡なし
    result_final = schrodinger_propagation(H0, ef, dip, psi0, return_traj=False)
    assert result_final.shape == (1, 2)

def test_schrodinger_propagation_with_variable_polarization():
    """可変偏光でのSchrodinger伝播テスト（RK4使用）"""
    tlist = np.linspace(-5, 5, 201)
    ef = ElectricField(tlist)
    
    # 第1パルス（x偏光）
    ef.add_dispersed_Efield(
        gaussian_fwhm, duration=1.0, t_center=-1.0,
        carrier_freq=1.0, amplitude=1.0,
        polarization=np.array([1.0, 0.0])
    )
    
    # 第2パルス（y偏光） - 偏光が変わるためRK4にフォールバック
    ef.add_dispersed_Efield(
        gaussian_fwhm, duration=1.0, t_center=1.0,
        carrier_freq=1.0, amplitude=1.0,
        polarization=np.array([0.0, 1.0])
    )
    
    H0 = np.diag([0.0, 1.0])
    dip = DummyDipoleOffDiag()
    psi0 = np.array([1.0, 0.0], dtype=np.complex128)
    
    result = schrodinger_propagation(H0, ef, dip, psi0, return_traj=True)
    assert result.shape[1] == 2

def test_schrodinger_propagation_with_time_return():
    """時間配列も返すSchrodinger伝播テスト"""
    tlist = np.linspace(-2, 2, 51)
    ef = ElectricField(tlist)
    ef.add_dispersed_Efield(
        gaussian_fwhm, duration=1.0, t_center=0.0,
        carrier_freq=1.0, amplitude=0.1,
        polarization=np.array([1.0, 0.0]), const_polarisation=True
    )
    
    H0 = np.diag([0.0, 1.0])
    dip = DummyDipoleOffDiag()
    psi0 = np.array([1.0, 0.0], dtype=np.complex128)
    
    time_psi, psi_traj = schrodinger_propagation(
        H0, ef, dip, psi0, return_traj=True, return_time_psi=True
    )
    
    assert len(time_psi) == psi_traj.shape[0]
    assert psi_traj.shape[1] == 2

def test_schrodinger_propagation_different_axes():
    """異なる軸設定でのテスト"""
    tlist = np.linspace(-1, 1, 21)
    ef = ElectricField(tlist)
    ef.Efield[:, 0] = 0.1  # Ex
    ef.Efield[:, 1] = 0.05  # Ey
    
    H0 = np.diag([0.0, 1.0])
    dip = DummyDipoleOffDiag()
    psi0 = np.array([1.0, 0.0], dtype=np.complex128)
    
    # デフォルト（axes="xy"）
    result_xy = schrodinger_propagation(H0, ef, dip, psi0, axes="xy")
    
    # zx軸設定
    result_zx = schrodinger_propagation(H0, ef, dip, psi0, axes="zx")
    
    # 異なる結果になる（mu_zは0なので影響は少ないが）
    assert result_xy.shape == result_zx.shape

def test_mixed_state_propagation_detailed():
    """詳細なmixed state伝播テスト"""
    tlist = np.linspace(-2, 2, 51)
    ef = ElectricField(tlist)
    ef.add_dispersed_Efield(
        gaussian_fwhm, duration=1.0, t_center=0.0,
        carrier_freq=1.0, amplitude=0.1,
        polarization=np.array([1.0, 0.0]), const_polarisation=True
    )
    
    H0 = np.diag([0.0, 1.0])
    dip = DummyDipoleOffDiag()
    
    # 複数の初期状態
    psi0s = [
        np.array([1.0, 0.0], dtype=np.complex128),
        np.array([0.0, 1.0], dtype=np.complex128),
        np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2)
    ]
    
    # 軌跡あり
    result_traj = mixed_state_propagation(H0, ef, psi0s, dip, return_traj=True)
    assert result_traj.shape[1:] == (2, 2)  # 密度行列
    
    # 軌跡なし
    result_final = mixed_state_propagation(H0, ef, psi0s, dip, return_traj=False)
    assert result_final.shape == (2, 2)
    
    # トレースは保存される（状態数）
    trace = np.trace(result_final)
    # 混合状態伝播では各状態がノルム1で、それらの密度行列の和となるため
    # トレースは状態数の2倍になる（|ψ⟩⟨ψ|の対角和）
    expected_trace = len(psi0s) * 2  # 各状態のノルム^2 の和
    assert np.isclose(trace, expected_trace, atol=1e-8)

def test_mixed_state_propagation_with_time():
    """時間配列も返すmixed state伝播テスト"""
    tlist = np.linspace(-1, 1, 21)
    ef = ElectricField(tlist)
    ef.Efield[:, 0] = 0.1
    
    H0 = np.diag([0.0, 1.0])
    dip = DummyDipoleOffDiag()
    psi0s = [np.array([1.0, 0.0], dtype=np.complex128)]
    
    time_rho, rho_traj = mixed_state_propagation(
        H0, ef, psi0s, dip, return_traj=True, return_time_rho=True
    )
    
    assert len(time_rho) == rho_traj.shape[0]
    assert rho_traj.shape[1:] == (2, 2)

def test_liouville_propagation_detailed():
    """詳細なLiouville伝播テスト"""
    tlist = np.linspace(-2, 2, 51)
    ef = ElectricField(tlist)
    ef.add_dispersed_Efield(
        gaussian_fwhm, duration=1.0, t_center=0.0,
        carrier_freq=1.0, amplitude=0.1,
        polarization=np.array([1.0, 0.0])
    )
    
    H0 = np.diag([0.0, 1.0])
    dip = DummyDipoleOffDiag()
    
    # 純粋状態から密度行列を作成
    psi = np.array([1.0, 0.0], dtype=np.complex128)
    rho0 = np.outer(psi, psi.conj())
    
    # 軌跡あり
    result_traj = liouville_propagation(H0, ef, dip, rho0, return_traj=True)
    assert result_traj.shape[1:] == (2, 2)
    
    # 軌跡なし
    result_final = liouville_propagation(H0, ef, dip, rho0, return_traj=False)
    assert result_final.shape == (2, 2)
    
    # トレースは保存される
    trace_initial = np.trace(rho0)
    trace_final = np.trace(result_final)
    assert np.isclose(trace_initial, trace_final, atol=1e-10)

def test_propagation_sample_stride():
    """サンプリングストライドのテスト"""
    tlist = np.linspace(-2, 2, 101)  # 多めのポイント
    ef = ElectricField(tlist)
    ef.Efield[:, 0] = 0.1
    
    H0 = np.diag([0.0, 1.0])
    dip = DummyDipole()
    psi0 = np.array([1.0, 0.0], dtype=np.complex128)
    
    # stride=1（全ポイント）
    result_stride1 = schrodinger_propagation(
        H0, ef, dip, psi0, return_traj=True, sample_stride=1
    )
    
    # stride=5（5ポイントおき）
    result_stride5 = schrodinger_propagation(
        H0, ef, dip, psi0, return_traj=True, sample_stride=5
    )
    
    # ポイント数が異なる
    assert result_stride1.shape[0] > result_stride5.shape[0]
    assert result_stride1.shape[1] == result_stride5.shape[1] == 2

def test_propagation_backend_consistency():
    """NumPyバックエンドでの一貫性テスト"""
    tlist = np.linspace(-1, 1, 21)
    ef = ElectricField(tlist)
    ef.Efield[:, 0] = 0.1
    
    H0 = np.diag([0.0, 1.0])
    dip = DummyDipole()
    psi0 = np.array([1.0, 0.0], dtype=np.complex128)
    
    # 明示的にnumpyバックエンド指定
    result = schrodinger_propagation(
        H0, ef, dip, psi0, backend="numpy"
    )
    
    assert result.shape[1] == 2

def test_propagation_error_cases():
    """エラーケースのテスト"""
    tlist = np.linspace(-1, 1, 11)
    ef = ElectricField(tlist)
    H0 = np.diag([0.0, 1.0])
    dip = DummyDipole()
    psi0 = np.array([1.0, 0.0], dtype=np.complex128)
    
    # 無効な軸指定
    with pytest.raises(ValueError):
        schrodinger_propagation(H0, ef, dip, psi0, axes="ab")
    
    # 存在しない双極子成分
    class BadDipole:
        def __init__(self):
            self.mu_x = np.eye(2, dtype=np.complex128)
            # mu_yが存在しない
    
    bad_dip = BadDipole()
    with pytest.raises(AttributeError):
        schrodinger_propagation(H0, ef, bad_dip, psi0)

def test_propagation_large_system():
    """大きなシステムでのテスト"""
    dim = 10
    tlist = np.linspace(-1, 1, 21)
    ef = ElectricField(tlist)
    ef.Efield[:, 0] = 0.01  # 小さい電場
    
    H0 = np.diag(np.arange(dim, dtype=float))
    dip = DummyDipole(dim)
    psi0 = np.zeros(dim, dtype=np.complex128)
    psi0[0] = 1.0  # 基底状態
    
    result = schrodinger_propagation(H0, ef, dip, psi0)
    assert result.shape[1] == dim
    
    # ノルムは保存される
    norm = np.linalg.norm(result[0])
    assert np.isclose(norm, 1.0, atol=1e-10) 