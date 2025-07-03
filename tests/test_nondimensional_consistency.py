import numpy as np
import pytest

from rovibrational_excitation.core.basis import LinMolBasis
from rovibrational_excitation.core.electric_field import ElectricField, gaussian
from rovibrational_excitation.core.hamiltonian import generate_H0_LinMol
from rovibrational_excitation.core.propagator import schrodinger_propagation
from rovibrational_excitation.core.states import StateVector
from rovibrational_excitation.dipole.linmol.cache import LinMolDipoleMatrix


class TestNondimensionalConsistency:
    """無次元化の一致性をテストするクラス"""
    
    @classmethod
    def setup_class(cls):
        """テスト用の共通パラメータを設定"""
        cls.V_max, cls.J_max = 1, 1  # 最小サイズでテスト
        cls.omega01, cls.domega, cls.mu0_cm = 1.0, 0.01, 1e-30
        cls.axes = "xy"
        
        # 基底とハミルトニアン
        cls.basis = LinMolBasis(cls.V_max, cls.J_max)
        cls.H0 = generate_H0_LinMol(
            cls.basis,
            omega_rad_phz=cls.omega01,
            delta_omega_rad_phz=cls.domega,
            B_rad_phz=0.01,
        )
        
        # 双極子行列
        cls.dipole_matrix = LinMolDipoleMatrix(
            cls.basis,
            mu0=cls.mu0_cm,
            potential_type="harmonic",
            backend="numpy",
            dense=True,
        )
        
        # 初期状態
        state = StateVector(cls.basis)
        state.set_state((0, 0, 0), 1)
        cls.psi0 = state.data
        
    def create_test_field(self, duration=50, amplitude=1e9):
        """テスト用の電場を作成"""
        ti, tf = 0.0, 200  # 短時間
        dt4Efield = 0.02  # 粗いサンプリング
        time4Efield = np.arange(ti, tf + 2 * dt4Efield, dt4Efield)
        
        tc = (time4Efield[-1] + time4Efield[0]) / 2
        polarization = np.array([1, 0])
        
        Efield = ElectricField(tlist_fs=time4Efield)
        Efield.add_dispersed_Efield(
            envelope_func=gaussian,
            duration=duration,
            t_center=tc,
            carrier_freq=self.omega01 / (2 * np.pi),
            amplitude=amplitude,
            polarization=polarization,
            const_polarisation=True,
        )
        
        return Efield
    
    def test_final_state_consistency(self):
        """最終状態の一致性をテスト（軌道なし）"""
        Efield = self.create_test_field()
        
        # 次元ありでの計算
        psi_final_dimensional = schrodinger_propagation(
            H0=self.H0,
            Efield=Efield,
            dipole_matrix=self.dipole_matrix,
            psi0=self.psi0,
            axes=self.axes,
            return_traj=False,
            nondimensional=False,
        )
        
        # 無次元化での計算  
        psi_final_nondimensional = schrodinger_propagation(
            H0=self.H0,
            Efield=Efield,
            dipole_matrix=self.dipole_matrix,
            psi0=self.psi0,
            axes=self.axes,
            return_traj=False,
            nondimensional=True,
        )
        
        # 一致性の確認
        assert psi_final_dimensional.shape == psi_final_nondimensional.shape
        
        # 存在確率の比較（位相は無視）
        prob_dimensional = np.abs(psi_final_dimensional)**2
        prob_nondimensional = np.abs(psi_final_nondimensional)**2
        
        prob_diff = np.max(np.abs(prob_dimensional - prob_nondimensional))
        assert prob_diff < 1e-10, f"存在確率の差が大きすぎます: {prob_diff:.2e}"
        
        # 規格化の確認
        norm_dimensional = np.sum(prob_dimensional)
        norm_nondimensional = np.sum(prob_nondimensional)
        
        assert abs(norm_dimensional - 1.0) < 1e-12, f"次元あり系の規格化エラー: {norm_dimensional}"
        assert abs(norm_nondimensional - 1.0) < 1e-12, f"無次元化系の規格化エラー: {norm_nondimensional}"
    
    def test_trajectory_consistency(self):
        """時間発展軌道の一致性をテスト"""
        Efield = self.create_test_field()
        sample_stride = 5  # メモリ節約
        
        # 次元ありでの計算
        time_dimensional, psi_dimensional = schrodinger_propagation(
            H0=self.H0,
            Efield=Efield,
            dipole_matrix=self.dipole_matrix,
            psi0=self.psi0,
            axes=self.axes,
            return_traj=True,
            return_time_psi=True,
            sample_stride=sample_stride,
            nondimensional=False,
        )
        
        # 無次元化での計算
        time_nondimensional, psi_nondimensional = schrodinger_propagation(
            H0=self.H0,
            Efield=Efield,
            dipole_matrix=self.dipole_matrix,
            psi0=self.psi0,
            axes=self.axes,
            return_traj=True,
            return_time_psi=True,
            sample_stride=sample_stride,
            nondimensional=True,
        )
        
        # 形状の一致性
        assert time_dimensional.shape == time_nondimensional.shape
        assert psi_dimensional.shape == psi_nondimensional.shape
        
        # 時間配列の一致性
        time_diff = np.max(np.abs(time_dimensional - time_nondimensional))
        assert time_diff < 1e-12, f"時間配列の差が大きすぎます: {time_diff:.2e}"
        
        # 存在確率の一致性
        prob_dimensional = np.abs(psi_dimensional)**2
        prob_nondimensional = np.abs(psi_nondimensional)**2
        
        prob_diff = np.max(np.abs(prob_dimensional - prob_nondimensional))
        assert prob_diff < 1e-10, f"存在確率の差が大きすぎます: {prob_diff:.2e}"
        
        # 規格化の保存
        norm_dimensional = np.sum(prob_dimensional, axis=1)
        norm_nondimensional = np.sum(prob_nondimensional, axis=1)
        
        assert np.all(np.abs(norm_dimensional - 1.0) < 1e-10), "次元あり系の規格化が保存されていません"
        assert np.all(np.abs(norm_nondimensional - 1.0) < 1e-10), "無次元化系の規格化が保存されていません"
    
    def test_weak_field_consistency(self):
        """弱電場での一致性をテスト（摂動論が適用できる領域）"""
        # 弱い電場
        Efield = self.create_test_field(amplitude=1e7)
        
        # 次元ありでの計算
        psi_final_dimensional = schrodinger_propagation(
            H0=self.H0,
            Efield=Efield,
            dipole_matrix=self.dipole_matrix,
            psi0=self.psi0,
            axes=self.axes,
            return_traj=False,
            nondimensional=False,
        )
        
        # 無次元化での計算
        psi_final_nondimensional = schrodinger_propagation(
            H0=self.H0,
            Efield=Efield,
            dipole_matrix=self.dipole_matrix,
            psi0=self.psi0,
            axes=self.axes,
            return_traj=False,
            nondimensional=True,
        )
        
        # 弱電場では特に高精度な一致を期待
        prob_dimensional = np.abs(psi_final_dimensional)**2
        prob_nondimensional = np.abs(psi_final_nondimensional)**2
        
        prob_diff = np.max(np.abs(prob_dimensional - prob_nondimensional))
        assert prob_diff < 1e-12, f"弱電場での存在確率の差が大きすぎます: {prob_diff:.2e}" 