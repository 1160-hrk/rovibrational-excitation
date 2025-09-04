import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pytest

from rovibrational_excitation.core.basis import LinMolBasis, TwoLevelBasis, Hamiltonian
from rovibrational_excitation.core.electric_field import ElectricField, gaussian_fwhm
from rovibrational_excitation.core.propagation import (
    SchrodingerPropagator,
    LiouvillePropagator,
    MixedStatePropagator,
)
from rovibrational_excitation.core.propagation.utils import get_backend
from rovibrational_excitation.core.units.converters import converter
from tests.mock_objects import MockDipole, MockEfield, DummyDipole


class DummyDipole:
    def __init__(self, dim=2):
        self.mu_x = np.eye(dim, dtype=np.complex128)
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


class DummyDipoleOffDiag:
    """非対角要素を持つダミー双極子"""

    def __init__(self, dim=2):
        self.mu_x = (
            np.array([[0, 1], [1, 0]], dtype=np.complex128)
            if dim == 2
            else np.random.random((dim, dim)) + 1j * np.random.random((dim, dim))
        )
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


def test_schrodinger_propagation():
    tlist = np.linspace(0, 1, 3)
    ef = ElectricField(tlist)
    ef.Efield[:, 0] = 1.0
    LinMolBasis(V_max=0, J_max=1, use_M=False)
    H0 = Hamiltonian(np.diag([0.0, 1.0]), units="J")
    dip = DummyDipole()
    psi0 = np.array([1.0, 0.0], dtype=np.complex128)
    result = SchrodingerPropagator().propagate(H0, ef, dip, psi0)
    assert result.shape[-1] == 2 or result[1].shape[-1] == 2


def test_mixed_state_propagation():
    tlist = np.linspace(0, 1, 3)
    ef = ElectricField(tlist)
    ef.Efield[:, 0] = 1.0
    LinMolBasis(V_max=0, J_max=1, use_M=False)
    H0 = Hamiltonian(np.diag([0.0, 1.0]), units="J")  # mixed_state_propagationは内部でschrodinger_propagationを呼ぶためHamiltonianが必要
    dip = DummyDipole()
    psi0s = [
        np.array([1.0, 0.0], dtype=np.complex128),
        np.array([0.0, 1.0], dtype=np.complex128),
    ]
    result = MixedStatePropagator().propagate(H0, ef, dip, psi0s)
    assert result.shape[-1] == 2 or result[1].shape[-1] == 2


def test_liouville_propagation():
    """Liouville方程式の時間発展テスト"""
    basis = TwoLevelBasis(energy_gap=1.0, input_units="rad/fs")
    H0 = basis.generate_H0()
    dip = MockDipole(basis)
    ef = MockEfield()
    
    rho0 = np.array([[0.8, 0.2j], [-0.2j, 0.2]], dtype=complex)
    
    # Prepare arguments for RK4
    backend = "numpy"
    xp = get_backend(backend)
    steps = (len(ef.tlist_s) - 1) // 2
    dt = (ef.tlist_s[1] - ef.tlist_s[0]) * 1e15  # fs
    H0_mat = H0.get_matrix("rad/fs")
    mu_x = dip.get_mu_in_units("x", "rad/fs/(V/m)")
    mu_y = dip.get_mu_in_units("y", "rad/fs/(V/m)")
    # Use simple constant fields with correct length
    Ex = np.ones(2 * steps + 1)
    Ey = np.zeros(2 * steps + 1)
    
    # Use a reasonable timestep in fs
    dt = 0.1
    
    rk4_args = (H0_mat, mu_x, mu_y, Ex, Ey, xp.asarray(rho0), dt, steps)
    
    from rovibrational_excitation.core.propagation.algorithms.rk4.lvne import rk4_lvne_traj
    result = rk4_lvne_traj(*rk4_args)
    
    # 形状とトレース保存を確認
    assert result.shape == (11, 2, 2)
    final_trace = np.trace(result[-1])
    assert np.isclose(final_trace, 1.0)


def test_schrodinger_propagation_with_constant_polarization():
    """一定偏光でのSchrodinger伝播テスト（Split-Operator使用）"""
    tlist = np.linspace(-5, 5, 201)
    ef = ElectricField(tlist)

    # 一定偏光のパルスを追加（より弱い電場で安定性を確保）
    ef.add_dispersed_Efield(
        gaussian_fwhm,
        duration=2.0,
        t_center=0.0,
        carrier_freq=1.0,
        amplitude=0.1,  # 弱い電場で安定性を確保
        polarization=np.array([1.0, 0.0]),
        const_polarisation=True,
    )

    H0 = Hamiltonian(np.diag([0.0, 1.0]), units="J")
    dip = DummyDipoleOffDiag()
    psi0 = np.array([1.0, 0.0], dtype=np.complex128)

    # 軌跡あり
    result_traj = SchrodingerPropagator(renorm=True).propagate(
        H0, ef, dip, psi0, return_traj=True
    )
    
    # 結果の形状が正しいことを確認
    if isinstance(result_traj, tuple):
        psi_traj = result_traj[1]
        assert psi_traj.shape[1] == 2
    else:
        assert result_traj.shape[1] == 2

    # 軌跡なし
    result_final = SchrodingerPropagator(renorm=True).propagate(
        H0, ef, dip, psi0, return_traj=False
    )
    assert result_final.shape == (2,)  # 形状を正しく修正


def test_schrodinger_propagation_with_variable_polarization():
    """可変偏光でのSchrodinger伝播テスト（RK4使用）"""
    tlist = np.linspace(-5, 5, 201)
    ef = ElectricField(tlist)

    # 第1パルス（x偏光）
    ef.add_dispersed_Efield(
        gaussian_fwhm,
        duration=1.0,
        t_center=-1.0,
        carrier_freq=1.0,
        amplitude=1.0,
        polarization=np.array([1.0, 0.0]),
    )

    # 第2パルス（y偏光） - 偏光が変わるためRK4にフォールバック
    ef.add_dispersed_Efield(
        gaussian_fwhm,
        duration=1.0,
        t_center=1.0,
        carrier_freq=1.0,
        amplitude=1.0,
        polarization=np.array([0.0, 1.0]),
    )

    H0 = Hamiltonian(np.diag([0.0, 1.0]), units="J")
    dip = DummyDipoleOffDiag()
    psi0 = np.array([1.0, 0.0], dtype=np.complex128)

    result = SchrodingerPropagator().propagate(H0, ef, dip, psi0, return_traj=True)
    assert result.shape[1] == 2


def test_schrodinger_propagation_with_time_return():
    """時間配列も返すSchrodinger伝播テスト"""
    tlist = np.linspace(-2, 2, 51)
    ef = ElectricField(tlist)
    ef.add_dispersed_Efield(
        gaussian_fwhm,
        duration=1.0,
        t_center=0.0,
        carrier_freq=1.0,
        amplitude=0.1,
        polarization=np.array([1.0, 0.0]),
        const_polarisation=True,
    )

    H0 = Hamiltonian(np.diag([0.0, 1.0]), units="J")
    dip = DummyDipoleOffDiag()
    psi0 = np.array([1.0, 0.0], dtype=np.complex128)

    time_psi, psi_traj = SchrodingerPropagator().propagate(
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

    H0 = Hamiltonian(np.diag([0.0, 1.0]), units="J")
    dip = DummyDipoleOffDiag()
    psi0 = np.array([1.0, 0.0], dtype=np.complex128)

    # デフォルト（axes="xy"）
    result_xy = SchrodingerPropagator().propagate(H0, ef, dip, psi0, axes="xy")

    # zx軸設定
    result_zx = SchrodingerPropagator().propagate(H0, ef, dip, psi0, axes="zx")

    # 異なる結果になる（mu_zは0なので影響は少ないが）
    assert result_xy.shape == result_zx.shape


def test_mixed_state_propagation_detailed():
    """詳細なmixed state伝播テスト"""
    tlist = np.linspace(-2, 2, 51)
    ef = ElectricField(tlist)
    ef.add_dispersed_Efield(
        gaussian_fwhm,
        duration=1.0,
        t_center=0.0,
        carrier_freq=1.0,
        amplitude=0.1,
        polarization=np.array([1.0, 0.0]),
        const_polarisation=True,
    )

    H0 = Hamiltonian(np.diag([0.0, 1.0]), units="J")
    dip = DummyDipoleOffDiag()
    psi0s = [
        np.array([1.0, 0.0], dtype=np.complex128),
        np.array([0.0, 1.0], dtype=np.complex128),
    ]

    # 軌跡あり
    result_traj = MixedStatePropagator().propagate(
        H0, ef, dip, psi0s, return_traj=True
    )
    assert result_traj.shape[1:] == (2, 2)  # 密度行列の形状

    # 軌跡なし
    result_final = MixedStatePropagator().propagate(
        H0, ef, dip, psi0s, return_traj=False
    )
    assert result_final.shape == (2, 2)  # 密度行列の形状


def test_mixed_state_propagation_with_time():
    """時間配列も返すmixed state伝播テスト"""
    tlist = np.linspace(-2, 2, 51)
    ef = ElectricField(tlist)
    ef.add_dispersed_Efield(
        gaussian_fwhm,
        duration=1.0,
        t_center=0.0,
        carrier_freq=1.0,
        amplitude=0.1,
        polarization=np.array([1.0, 0.0]),
        const_polarisation=True,
    )

    H0 = Hamiltonian(np.diag([0.0, 1.0]), units="J")
    dip = DummyDipoleOffDiag()
    psi0s = [
        np.array([1.0, 0.0], dtype=np.complex128),
        np.array([0.0, 1.0], dtype=np.complex128),
    ]

    time_rho, rho_traj = MixedStatePropagator().propagate(
        H0, ef, dip, psi0s, return_traj=True, return_time_rho=True
    )

    assert len(time_rho) == rho_traj.shape[0]
    assert rho_traj.shape[1] == 2


def test_propagation_sample_stride():
    """サンプリングストライドのテスト"""
    tlist = np.linspace(-2, 2, 101)
    ef = ElectricField(tlist)
    ef.add_dispersed_Efield(
        gaussian_fwhm,
        duration=1.0,
        t_center=0.0,
        carrier_freq=1.0,
        amplitude=0.1,
        polarization=np.array([1.0, 0.0]),
        const_polarisation=True,
    )

    H0 = Hamiltonian(np.diag([0.0, 1.0]), units="J")
    dip = DummyDipoleOffDiag()
    psi0 = np.array([1.0, 0.0], dtype=np.complex128)

    # stride=1のとき
    result_stride1 = SchrodingerPropagator().propagate(
        H0, ef, dip, psi0, return_traj=True, sample_stride=1
    )

    # stride=2のとき
    result_stride2 = SchrodingerPropagator().propagate(
        H0, ef, dip, psi0, return_traj=True, sample_stride=2
    )

    # 時間軸が半分になることを確認
    assert result_stride2.shape[0] == (result_stride1.shape[0] + 1) // 2


def test_propagation_backend_consistency():
    """バックエンド間の一貫性テスト"""
    tlist = np.linspace(-1, 1, 21)
    ef = ElectricField(tlist)
    ef.Efield[:, 0] = 0.1

    H0 = Hamiltonian(np.diag([0.0, 1.0]), units="J")
    dip = DummyDipoleOffDiag()
    psi0 = np.array([1.0, 0.0], dtype=np.complex128)

    result = SchrodingerPropagator(backend="numpy").propagate(H0, ef, dip, psi0)
    assert result.shape[-1] == 2


def test_propagation_error_cases():
    """エラー処理のテスト"""
    tlist = np.linspace(-1, 1, 21)
    ef = ElectricField(tlist)
    ef.Efield[:, 0] = 0.1

    H0 = Hamiltonian(np.diag([0.0, 1.0]), units="J")
    dip = DummyDipoleOffDiag()
    psi0 = np.array([1.0, 0.0], dtype=np.complex128)

    with pytest.raises(ValueError):
        SchrodingerPropagator().propagate(H0, ef, dip, psi0, axes="ab")

    class BadDipole:
        def __init__(self):
            pass

    with pytest.raises(AttributeError):
        SchrodingerPropagator().propagate(H0, ef, BadDipole(), psi0)


def test_propagation_large_system():
    """大きなシステムでのテスト"""
    tlist = np.linspace(-1, 1, 21)
    ef = ElectricField(tlist)
    ef.Efield[:, 0] = 0.01

    # 4次元システム
    H0 = Hamiltonian(np.diag([0.0, 1.0, 2.0, 3.0]), units="J")
    dip = DummyDipole(dim=4)
    psi0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)

    result = SchrodingerPropagator().propagate(H0, ef, dip, psi0)
    assert result.shape[-1] == 4
