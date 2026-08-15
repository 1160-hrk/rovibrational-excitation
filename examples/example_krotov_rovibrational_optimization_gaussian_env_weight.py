#!/usr/bin/env python
"""
Krotov法による振動回転励起最適制御
================================

線形分子の振動回転準位間で100%ポピュレーション移行を実現する
電場波形をKrotov法で最適化。

実行方法:
    python examples/krotov_rovibrational_optimization.py

参考文献:
- D. M. Reich et al., J. Chem. Phys. 136, 104103 (2012)
- S. Machnes et al., Phys. Rev. Lett. 120, 053203 (2018)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Tuple, Optional

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from rovibrational_excitation.core.basis import LinMolBasis, StateVector
from rovibrational_excitation.fields import ElectricField, gaussian
from rovibrational_excitation.core.propagation import SchrodingerPropagator
from rovibrational_excitation.dipole.linmol import LinMolDipoleMatrix
from rovibrational_excitation.core.units.converters import converter
from rovibrational_excitation.core.units.constants import CONSTANTS
from rovibrational_excitation.core.propagation.utils import cm_to_rad_phz
from rovibrational_excitation.core.propagation.utils import J_to_rad_phz

# %% パラメータ設定
# システムパラメータ
V_MAX = 3  # 最大振動量子数
J_MAX = 3  # 最大回転量子数
USE_M = True  # 磁気量子数を使用

# 分子パラメータ
OMEGA_01 = 2349.1  # 振動周波数 [cm^-1]
DOMEGA = 25  # 非調和性補正 [cm^-1]
B_CONSTANT = 0.39  # 回転定数 [cm^-1]
ALPHA_CONSTANT = 0.0037  # 振動-回転相互作用定数 [cm^-1]
UNIT_FREQUENCY = "cm^-1"
MU0 = 1e-30  # 双極子行列要素の大きさを大きく [C·m]
UNIT_DIPOLE = "C*m"

# 初期状態と目標状態
INITIAL_STATE = (0, 0, 0)  # |v=0, J=0, M=0⟩
# TARGET_STATE = (1, 1, -1)   # |v=1, J=1, M=1⟩ - xy偏光で可能な遷移
# TARGET_STATE = (1, 1, -1)   # |v=1, J=1, M=1⟩ - xy偏光で可能な遷移
TARGET_STATE = (3, 3, 0)   # |v=3, J=1, M=0⟩ - zx偏光で可能な遷移

# 時間設定
TIME_TOTAL = 3000.0  # 最適化時間窓 [fs]
DT_EFIELD = 0.5     # 電場サンプリング間隔 [fs]
SAMPLE_STRIDE = 1   # サンプリングストライド（一致させる）

# 偏光軸マッピング設定（例: "xy", "zx", "yz", ...）
AXES = "zx"

# Krotov最適化パラメータ
MAX_ITER = 200      # 最大反復回数
CONVERGENCE_TOL = 1e-30  # 収束判定閾値をより厳しく
LAMBDA_A = 1e-19     # 電場制約パラメータ（適切な値に調整）
TARGET_FIDELITY = 1.0  # 目標フィデリティー

# 早期収束のガード
MIN_ITER_EARLYSTOP = 50           # これ未満では「小変化」で止めない
EARLYSTOP_MIN_FIDELITY = 0.90     # 小変化でもこのフィデリティ未満なら継続

# 電場制約
FIELD_MAX = 1e9     # 現実的な電場制限 [V/m] (1e8 ≈ 1.3e11 W/cm²)
ENFORCE_HARD_CLIP = False  # 単調性確保のためハードクリップは基本無効

# エンベロープ設定（FWHMに基づくガウシアン。sin^2窓で端点をゼロに）
ENVELOPE_DURATION = TIME_TOTAL*8  # [fs] ガウシアンFWHM。None/<=0 で無効
ENVELOPE_DURATION = None  # [fs] ガウシアンFWHM。None/<=0 で無効
USE_SIN2_ENVELOPE = True

# %% ユーティリティ関数
def calculate_fidelity(psi_final: np.ndarray, target_idx: int) -> float:
    """フィデリティー計算"""
    return abs(psi_final[target_idx])**2

def clip_field_amplitudes(Ex: np.ndarray, Ey: np.ndarray, max_field: float) -> tuple[np.ndarray, np.ndarray]:
    """
    時刻ごとの電場振幅制限（正しいクリップ）
    """
    Ex_clipped = np.clip(Ex, -max_field, max_field)
    Ey_clipped = np.clip(Ey, -max_field, max_field)
    return Ex_clipped, Ey_clipped

def shape_function(t: np.ndarray, T: float) -> np.ndarray:
    """
    Krotov形状関数: 境界でゼロになる窓関数
    S(t) = sin²(πt/T) for t ∈ [0, T]
    """
    return np.sin(np.pi * t / T) ** 2

def gaussian_envelope_fwhm(t: np.ndarray, t_center: float, fwhm: float) -> np.ndarray:
    """
    ガウシアン包絡（FWHM指定, 最大値1）
    g(t) = exp(-4 ln 2 * ((t - t0)/FWHM)^2)
    """
    if fwhm is None or fwhm <= 0:
        return np.ones_like(t)
    return np.exp(-4.0 * np.log(2.0) * ((t - t_center) / float(fwhm)) ** 2)

def windowed_gaussian_envelope(t: np.ndarray, T: float, t_center: float,
                               duration_fwhm: float, use_sin2: bool = True) -> np.ndarray:
    """
    ガウシアン×sin^2 窓（端点で厳密に0）。sin^2を無効化可能。
    """
    g = gaussian_envelope_fwhm(t, t_center, duration_fwhm)
    if use_sin2:
        return g * shape_function(t, T)
    return g

def apply_envelope_to_field(t: np.ndarray, field_data: np.ndarray, envelope: np.ndarray) -> np.ndarray:
    """電場配列 Nx2 にエンベロープを要素積で適用"""
    if envelope is None:
        return field_data
    return field_data * envelope[:, None]

def print_iteration_info(iteration: int, fidelity: float, field_norm: float):
    """反復情報の表示"""
    print(f"Iteration {iteration:3d}: Fidelity = {fidelity:.6f}, "
          f"Field norm = {field_norm:.3e}")

class KrotovOptimizer:
    """Krotov法による量子最適制御クラス"""
    
    def __init__(self, basis, hamiltonian, dipole_matrix, 
                 initial_idx: int, target_idx: int,
                 time_total: float, dt: float, sample_stride: int = 1,
                 axes: str = "xy"):
        """
        初期化
        
        Parameters
        ----------
        basis : LinMolBasis
            基底系
        hamiltonian : Hamiltonian
            ハミルトニアン
        dipole_matrix : LinMolDipoleMatrix
            双極子行列
        initial_idx : int
            初期状態のインデックス
        target_idx : int
            目標状態のインデックス
        time_total : float
            最適化時間窓 [fs]
        dt : float
            時間ステップ [fs]
        sample_stride : int
            サンプリングストライド
        """
        self.basis = basis
        self.hamiltonian = hamiltonian
        self.dipole_matrix = dipole_matrix
        self.initial_idx = initial_idx
        self.target_idx = target_idx
        self.dt = dt
        self.sample_stride = sample_stride
        self.axes = axes
        
        # RK4用の正しい時間軸設定
        # プロパゲーション軌跡長を基準に電場長を決定
        target_traj_steps = int(time_total / dt) + 1
        required_field_steps = 2 * (target_traj_steps - 1) + 1
        
        self.tlist = np.linspace(0, time_total, required_field_steps)
        self.n_steps = len(self.tlist)
        self.n_traj_steps = target_traj_steps
        
        # プロパゲータ初期化
        self.propagator = SchrodingerPropagator(
            backend="numpy",
            validate_units=True,
            renorm=True  # 長時間伝播でのノルム保存
        )
        
        # 初期状態・目標状態設定
        self.psi_initial = np.zeros(basis.size(), dtype=complex)
        self.psi_initial[initial_idx] = 1.0
        
        self.psi_target = np.zeros(basis.size(), dtype=complex)
        self.psi_target[target_idx] = 1.0
        
        print(f"初期化完了:")
        print(f"  基底次元: {basis.size()}")
        print(f"  電場時間ステップ数: {self.n_steps}")
        print(f"  予想プロパゲーション軌跡長: {self.n_traj_steps}")
        print(f"  サンプリングストライド: {sample_stride}")
        print(f"  初期状態: {basis.basis[initial_idx]} (index={initial_idx})")
        print(f"  目標状態: {basis.basis[target_idx]} (index={target_idx})")
        
        # RK4時間軸診断
        print(f"  RK4時間軸診断:")
        print(f"    電場時間窓: 0 - {time_total:.1f} fs")
        print(f"    電場配列長: {self.n_steps} (RK4必要長)")
        print(f"    軌跡配列長: {self.n_traj_steps} (予想)")
        print(f"    電場dt: {self.tlist[1] - self.tlist[0]:.3f} fs") 
        print(f"    プロパゲーションdt: {dt:.3f} fs")
        
        # 伝播と整合する双極子行列（μ' = μ / ħ_fs）を事前計算
        mu_x_si = self.dipole_matrix.get_mu_x_SI()
        mu_y_si = self.dipole_matrix.get_mu_y_SI()
        # z成分も準備（axesマッピングに対応）
        mu_z_si = self.dipole_matrix.get_mu_z_SI()
        if hasattr(mu_x_si, 'toarray'):
            mu_x_si = mu_x_si.toarray()
        if hasattr(mu_y_si, 'toarray'):
            mu_y_si = mu_y_si.toarray()
        if hasattr(mu_z_si, 'toarray'):
            mu_z_si = mu_z_si.toarray()
        self.mu_x_prime = cm_to_rad_phz(mu_x_si)
        self.mu_y_prime = cm_to_rad_phz(mu_y_si)
        self.mu_z_prime = cm_to_rad_phz(mu_z_si)
        # 軸→双極子のマップ（Ex↔axes[0], Ey↔axes[1]）
        self._mu_prime_map = {
            'x': self.mu_x_prime,
            'y': self.mu_y_prime,
            'z': self.mu_z_prime,
        }
    
    def forward_propagation(self, efield: ElectricField) -> Tuple[np.ndarray, np.ndarray]:
        """
        順方向時間発展
        
        Parameters
        ----------
        efield : ElectricField
            電場オブジェクト
            
        Returns
        -------
        tuple
            (time_array, psi_trajectory)
        """
        result = self.propagator.propagate(
            hamiltonian=self.hamiltonian,
            efield=efield,
            dipole_matrix=self.dipole_matrix,
            initial_state=self.psi_initial,
            axes=self.axes,
            return_traj=True,
            return_time_psi=True,
            sample_stride=self.sample_stride,
            algorithm="rk4",
            sparse=False
        )
        
        return result[0], result[1]  # time, psi_trajectory
    
    def backward_propagation(self, efield: ElectricField, psi_traj: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        逆方向時間発展（正しいKrotov随伴方程式）
        
        Parameters
        ----------
        efield : ElectricField
            電場オブジェクト
        psi_traj : np.ndarray
            順方向伝播の状態軌跡
            
        Returns
        -------
        tuple
            (time_array, chi_trajectory)
        """
        # 随伴境界条件: χ(T) = ⟨φ_tar|ψ(T)⟩ |φ_tar⟩ （f=|⟨φ|ψ(T)⟩|^2 の勾配で定義）
        psi_final = psi_traj[-1]
        overlap = np.vdot(self.psi_target, psi_final)  # ⟨φ_tar|ψ(T)⟩
        chi_initial = overlap * self.psi_target
        
        # 時間反転電場の作成（t を T→0 に減少させる：dt<0 を用いる）
        efield_backward = ElectricField(tlist=self.tlist[::-1])
        efield_data = efield.get_Efield()
        efield_backward.add_arbitrary_Efield(efield_data[::-1])  # 符号反転を削除
        
        # 逆方向伝播（dtを負にして実現）
        result = self.propagator.propagate(
            hamiltonian=self.hamiltonian,
            efield=efield_backward,
            dipole_matrix=self.dipole_matrix,
            initial_state=chi_initial,  # 正しい随伴初期条件
            axes=self.axes,
            return_traj=True,
            return_time_psi=True,
            sample_stride=self.sample_stride,
            algorithm="rk4",
            sparse=False
        )
        
        # 時間軸を元に戻す
        time_backward = -result[0][::-1]
        chi_backward = result[1][::-1]
        
        return time_backward, chi_backward
    
    def calculate_field_update(self, psi_traj: np.ndarray, 
                             chi_traj: np.ndarray, 
                             current_field: np.ndarray,
                             lambda_a: float) -> np.ndarray:
        """
        Krotov更新式（逐次更新に近い時間局所形）による電場更新
        
        Parameters
        ----------
        psi_traj : np.ndarray
            順方向軌跡
        chi_traj : np.ndarray
            逆方向軌跡
        current_field : np.ndarray
            現在の電場
        lambda_a : float
            制約パラメータ
        備考
        ----
        更新は学習率を使わず、S(t)/lambda_a * 2*Im⟨χ|μ|ψ⟩ を基本形とする。
            
        Returns
        -------
        np.ndarray
            更新後の電場
        """
        # 伝播と同一スケールの双極子（μ' = μ/ħ_fs）を使用
        # axesに基づき Ex↔axes[0], Ey↔axes[1] でマッピング
        ax0, ax1 = self.axes[0].lower(), self.axes[1].lower()
        mu_a = self._mu_prime_map[ax0]
        mu_b = self._mu_prime_map[ax1]
        
        # 逐次更新（時間順）: 各時刻の更新を即座にE(t)へ反映
        n_traj_steps = len(psi_traj)
        n_field_steps = self.n_steps
        updated_field = current_field.copy()
        for i in range(n_traj_steps):
            jf = i * 2
            if jf >= n_field_steps:
                break
            psi_i = psi_traj[i]
            chi_i = chi_traj[i]
            # Krotov勾配: 2 * Im[⟨χ|μ|ψ⟩]
            grad_x = 2.0 * float(np.imag(np.vdot(chi_i, (mu_a @ psi_i))))
            grad_y = 2.0 * float(np.imag(np.vdot(chi_i, (mu_b @ psi_i))))
            # 形状関数（境界0）
            S = float(np.sin(np.pi * self.tlist[jf] / self.tlist[-1]) ** 2)
            # 伝播は H = H0 + μ'·E（RK4実装と整合）→ ∂H/∂E = +μ' → ΔE = +(S/λ)·Im⟨χ|μ'|ψ⟩
            dEx = S * grad_x / lambda_a
            dEy = S * grad_y / lambda_a
            # 時刻jfとその次の半ステップにも同じ更新を適用
            updated_field[jf, 0] += dEx
            updated_field[jf, 1] += dEy
            if jf + 1 < n_field_steps:
                updated_field[jf + 1, 0] += dEx
                updated_field[jf + 1, 1] += dEy
        if ENFORCE_HARD_CLIP:
            updated_field[:, 0] = np.clip(updated_field[:, 0], -FIELD_MAX, FIELD_MAX)
            updated_field[:, 1] = np.clip(updated_field[:, 1], -FIELD_MAX, FIELD_MAX)
        return updated_field
    
    def optimize(self, lambda_a: float = LAMBDA_A, 
                max_iter: int = MAX_ITER,
                convergence_tol: float = CONVERGENCE_TOL,
                target_fidelity: float = TARGET_FIDELITY) -> Tuple[ElectricField, list, list]:
        """
        Krotov最適化実行
        
        Parameters
        ----------
        lambda_a : float
            制約パラメータ
        max_iter : int
            最大反復回数
        convergence_tol : float
            収束判定閾値
        target_fidelity : float
            目標フィデリティー
            
        Returns
        -------
        tuple
            (最適化電場, フィデリティー履歴, 電場ノルム履歴)
        """
        print(f"\nKrotov最適化開始:")
        print(f"  最大反復回数: {max_iter}")
        print(f"  収束閾値: {convergence_tol}")
        print(f"  目標フィデリティー: {target_fidelity}")
        print(f"  制約パラメータ: {lambda_a}")
        
        # 進捗用メトリクス
        best_fidelity = 0.0
        terminal_cost_history = []
        running_cost_history = []
        total_cost_history = []
        
        # 初期電場の設定（x/y両成分を持つ円偏光ベース）
        efield = ElectricField(tlist=self.tlist)
        tc = self.tlist[-1] / 2
        
        # 共鳴周波数の計算
        eigenvalues = self.hamiltonian.get_eigenvalues()
        carrier_freq = float(converter.convert_frequency(
            eigenvalues[self.target_idx] - eigenvalues[self.initial_idx],
            "rad/fs", UNIT_FREQUENCY
        ))
        carrier_freq = float(converter.convert_frequency(
            OMEGA_01,
            "cm^-1", UNIT_FREQUENCY
        ))
        duration = TIME_TOTAL/2
        # duration = 100
        initial_amplitude = 1e9  # 実用的電場スケールに調整 [V/m]
        # gdd = -5e4
        gdd = 0
        # X偏光成分（σ_x遷移用）
        efield.add_dispersed_Efield(
            envelope_func=gaussian,
            duration=duration,  # パルス幅を調整
            t_center=tc,
            carrier_freq=carrier_freq,
            carrier_freq_units=UNIT_FREQUENCY,
            amplitude=initial_amplitude,
            polarization=np.array([1, 0]),  # x方向偏光
            const_polarisation=False,
            gdd=gdd
        )
        
        # Y偏光成分（σ_y遷移用）
        efield.add_dispersed_Efield(
            envelope_func=gaussian,
            duration=duration,
            t_center=tc,
            carrier_freq=carrier_freq,
            carrier_freq_units=UNIT_FREQUENCY,
            # amplitude=initial_amplitude,  # 同じ振幅でスタート
            amplitude=0,  # 同じ振幅でスタート
            polarization=np.array([0, 1]),  # y方向偏光
            const_polarisation=False,
            gdd=gdd,
            phase_rad=-np.pi/2,
        )
        
        # 注意: carrier_phaseがサポートされていないため
        # 円偏光は最適化過程でx,y成分の位相関係が自動的に調整される
        
        # エンベロープ適用（ガウシアン×sin^2で端点ゼロ）
        if ENVELOPE_DURATION is not None and ENVELOPE_DURATION > 0:
            self.envelope_window = windowed_gaussian_envelope(
                self.tlist, self.tlist[-1], tc, ENVELOPE_DURATION, USE_SIN2_ENVELOPE
            )
            field_data_init = efield.get_Efield()
            field_data_init = apply_envelope_to_field(self.tlist, field_data_init, self.envelope_window)
            efield = ElectricField(tlist=self.tlist)
            efield.add_arbitrary_Efield(field_data_init)
        else:
            self.envelope_window = np.ones_like(self.tlist)

        # 最適化履歴
        fidelity_history = []
        field_norm_history = []
        
        print(f"  初期キャリア周波数: {carrier_freq:.3f} {UNIT_FREQUENCY}")
        print(f"  初期振幅: {initial_amplitude:.3e} V/m")
        
        start_time = time.time()
        
        # 形状関数とdt（走行コスト計算用）
        S_t = shape_function(self.tlist, self.tlist[-1])
        dt_field = float(self.tlist[1] - self.tlist[0])
        
        for iteration in range(max_iter):
            # Forward propagation
            time_forward, psi_traj = self.forward_propagation(efield)
            
            # Backward propagation
            time_backward, chi_traj = self.backward_propagation(efield, psi_traj)
            
            # フィデリティー計算
            final_psi = psi_traj[-1]
            fidelity = calculate_fidelity(final_psi, self.target_idx)
            
            # 電場ノルム計算
            field_data = efield.get_Efield()
            field_norm = float(np.linalg.norm(field_data))
            
            # コスト計算
            J_T = 1.0 - float(fidelity)
            E2 = field_data[:, 0]**2 + field_data[:, 1]**2
            J_a = 0.5 * float(LAMBDA_A) * float(np.sum(S_t * E2) * dt_field)
            J = J_T + J_a
            terminal_cost_history.append(J_T)
            running_cost_history.append(J_a)
            total_cost_history.append(J)

            # 履歴保存
            fidelity_history.append(fidelity)
            field_norm_history.append(field_norm)
            
            # 進捗表示（改良版）
            if iteration % 10 == 0 or iteration < 10 or fidelity > best_fidelity * 0.95:
                print(f"Iteration {iteration+1:3d}: Fidelity = {fidelity:.6f}, "
                      f"Field norm = {field_norm:.2e}, J_T = {J_T:.6e}, J_a = {J_a:.6e}, J = {J:.6e}")
            
            # 収束判定
            if fidelity >= target_fidelity:
                print(f"🎉 目標フィデリティー {target_fidelity} に到達しました！")
                break
            
            # 通常の収束判定（小変化 かつ 最低反復数以降 かつ 十分な到達度）
            if iteration + 1 >= MIN_ITER_EARLYSTOP:
                recent_changes = [abs(fidelity_history[i] - fidelity_history[i-1]) 
                                  for i in range(max(1, len(fidelity_history)-2), len(fidelity_history))]
                small_change = all(change < convergence_tol for change in recent_changes)
                if small_change and fidelity >= EARLYSTOP_MIN_FIDELITY:
                    print(f"フィデリティー変化が収束閾値 {convergence_tol} を下回り、"
                          f"Fidelity≥{EARLYSTOP_MIN_FIDELITY:.2f} のため終了します。")
                    break
            
            # 電場更新（逐次形）：最後の反復以外
            if iteration < max_iter - 1:
                updated_field = self.calculate_field_update(
                    psi_traj, chi_traj, field_data, lambda_a
                )
                # エンベロープを適用して端点ゼロを保証
                if self.envelope_window is not None:
                    updated_field = apply_envelope_to_field(self.tlist, updated_field, self.envelope_window)
                efield = ElectricField(tlist=self.tlist)
                efield.add_arbitrary_Efield(updated_field)
        
        elapsed_time = time.time() - start_time
        print(f"\n最適化完了 (所要時間: {elapsed_time:.2f}秒)")
        print(f"最終フィデリティー: {fidelity_history[-1]:.6f}")
        print(f"最大フィデリティー: {max(fidelity_history):.6f}")
        print(f"反復回数: {len(fidelity_history)}")
        # 追加統計
        if len(total_cost_history) > 0:
            print(f"最終コスト: J_T={terminal_cost_history[-1]:.6e}, J_a={running_cost_history[-1]:.6e}, J={total_cost_history[-1]:.6e}")
        
        # 履歴をインスタンスへ保存（可視化用）
        self.fidelity_history = fidelity_history
        self.field_norm_history = field_norm_history
        self.terminal_cost_history = terminal_cost_history
        self.running_cost_history = running_cost_history
        self.total_cost_history = total_cost_history
        
        # 最適化統計
        if len(fidelity_history) > 1:
            improvement = fidelity_history[-1] - fidelity_history[0]
            max_improvement = max(fidelity_history) - fidelity_history[0]
            print(f"フィデリティー改善: {improvement:.6f} (+{improvement/fidelity_history[0]*100:.1f}%)")
            print(f"最大フィデリティー改善: {max_improvement:.6f} (+{max_improvement/fidelity_history[0]*100:.1f}%)")
        
        return efield, fidelity_history, field_norm_history

# %% メイン実行部分
def main():
    """メイン実行関数"""
    print("=== Krotov法による振動回転励起最適制御 ===")
    
    # 基底・ハミルトニアン・双極子行列の生成
    print(f"基底設定: V_max={V_MAX}, J_max={J_MAX}, use_M={USE_M}")
    
    basis = LinMolBasis(
        V_max=V_MAX, J_max=J_MAX, use_M=USE_M,
        omega=OMEGA_01,
        delta_omega=DOMEGA,
        B=B_CONSTANT,
        alpha=ALPHA_CONSTANT,
        input_units=UNIT_FREQUENCY,
        output_units="rad/fs"
    )
    
    H0 = basis.generate_H0()
    
    dipole_matrix = LinMolDipoleMatrix(
        basis=basis,
        mu0=MU0,
        potential_type="harmonic",
        backend="numpy",
        dense=True,  # 最適化では密行列を使用
        units="C*m",
        units_input=UNIT_DIPOLE,
    )
    
    print(f"基底次元: {basis.size()}")
    
    # インデックス取得
    initial_idx = basis.get_index(INITIAL_STATE)
    target_idx = basis.get_index(TARGET_STATE)
    
    print(f"初期状態: {INITIAL_STATE} → インデックス {initial_idx}")
    print(f"目標状態: {TARGET_STATE} → インデックス {target_idx}")
    
    # エネルギー準位確認
    eigenvalues = H0.get_eigenvalues()
    initial_energy = eigenvalues[initial_idx]
    target_energy = eigenvalues[target_idx]
    energy_diff = target_energy - initial_energy
    
    print(f"初期状態エネルギー: {initial_energy:.6f} rad/fs")
    print(f"目標状態エネルギー: {target_energy:.6f} rad/fs")
    print(f"エネルギー差: {energy_diff:.6f} rad/fs")
    
    # 選択則チェック
    dv = TARGET_STATE[0] - INITIAL_STATE[0]
    dj = TARGET_STATE[1] - INITIAL_STATE[1] 
    dm = TARGET_STATE[2] - INITIAL_STATE[2]
    print(f"選択則チェック: Δv={dv}, ΔJ={dj}, ΔM={dm}")
    
    if 'z' in AXES:
        selection_rule_ok = (dv%2 == 0 and dj%2 == 0 or dv%2 == 1 and dj%2 == 1) and not (dv == 0 and dj == 0 and dm == 0)
    else:
        selection_rule_ok = (dv%2 == 0 and dj%2 == 0 and dm%2 == 0 or dv%2 == 1 and dj%2 == 1 and dm%2 == 1) and not (dv == 0 and dj == 0 and dm == 0)
    if selection_rule_ok:
        print("✓ 選択則を満たしています")
    else:
        print("⚠️ 選択則違反: 直接遷移が困難な可能性があります")
    
    # 双極子遷移要素の確認
    mu_x = dipole_matrix.get_mu_x_SI()
    mu_y = dipole_matrix.get_mu_y_SI()
    if hasattr(mu_x, 'toarray'):
        mu_x = mu_x.toarray()
    if hasattr(mu_y, 'toarray'):
        mu_y = mu_y.toarray()
    
    
    # Krotov最適化実行
    optimizer = KrotovOptimizer(
        basis=basis,
        hamiltonian=H0,
        dipole_matrix=dipole_matrix,
        initial_idx=initial_idx,
        target_idx=target_idx,
        time_total=TIME_TOTAL,
        dt=DT_EFIELD,
        sample_stride=SAMPLE_STRIDE,
        axes=AXES,
    )
    
    optimal_field, fidelity_hist, field_norm_hist = optimizer.optimize()
    
    # 結果の可視化
    plot_optimization_results(optimizer, optimal_field, fidelity_hist, field_norm_hist)

def plot_optimization_results(optimizer: KrotovOptimizer, 
                            optimal_field: ElectricField,
                            fidelity_hist: list, 
                            field_norm_hist: list):
    """最適化結果の可視化"""
    
    # 最適化電場での最終計算
    time_final, psi_final = optimizer.forward_propagation(optimal_field)
    prob_final = np.abs(psi_final)**2
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 最適化電場波形
    field_data = optimal_field.get_Efield()
    axes[0, 0].plot(optimizer.tlist, field_data[:, 0], 'r-', label='Ex(t)', linewidth=1.5)
    axes[0, 0].plot(optimizer.tlist, field_data[:, 1], 'b-', label='Ey(t)', linewidth=1.5)
    axes[0, 0].set_xlabel('Time [fs]')
    axes[0, 0].set_ylabel('Electric Field [V/m]')
    axes[0, 0].set_title('Optimized Electric Field')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. フィデリティー収束
    iterations = range(1, len(fidelity_hist) + 1)
    axes[0, 1].plot(iterations, fidelity_hist, 'go-', linewidth=2, markersize=4)
    axes[0, 1].axhline(y=TARGET_FIDELITY, color='r', linestyle='--', alpha=0.7, 
                       label=f'Target: {TARGET_FIDELITY}')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Fidelity')
    axes[0, 1].set_title('Convergence History')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1.05)
    
    # 3. 許容準位のポピュレーション時間発展（xy偏光: ΔV=±1, ΔJ=±1, ΔM=±1 → 多段階で V,J,M の偶奇が一致）
    def _is_allowed_state(v: int, J: int, M: int) -> bool:
        if 'z' in AXES:
            return (v % 2) == (J % 2)
        else:
            return (v % 2) == (J % 2) == (abs(M) % 2)

    # 初期・目標状態は強調表示
    highlight_states = [INITIAL_STATE, TARGET_STATE]
    for state in highlight_states:
        if state in optimizer.basis.index_map:
            idx = optimizer.basis.get_index(state)
            axes[1, 0].plot(time_final, prob_final[:, idx], linewidth=2.5,
                           label=f'|v={state[0]}, J={state[1]}, M={state[2]}⟩ (highlight)')

    # 許容準位をすべて描画（細線）
    for i, (v, J, M) in enumerate(optimizer.basis.basis):
        if _is_allowed_state(v, J, M):
            # 既に強調表示したものはスキップ
            if (v, J, M) in highlight_states:
                continue
            axes[1, 0].plot(time_final, prob_final[:, i], linewidth=1.0, alpha=0.9,
                           label=f'|v={v}, J={J}, M={M}⟩')

    # 許容準位のポピュレーション総和と全状態の総和を重ね描画
    allowed_indices = [idx for idx, (v, J, M) in enumerate(optimizer.basis.basis) if _is_allowed_state(v, J, M)]
    if len(allowed_indices) > 0:
        allowed_sum = np.sum(prob_final[:, allowed_indices], axis=1)
        axes[1, 0].plot(time_final, allowed_sum, 'k-', linewidth=2.5, label='Allowed sum')
    total_sum = np.sum(prob_final, axis=1)
    axes[1, 0].plot(time_final, total_sum, color='gray', linestyle='--', linewidth=2.0, label='Total sum')
    
    axes[1, 0].set_xlabel('Time [fs]')
    axes[1, 0].set_ylabel('Population')
    axes[1, 0].set_title('State Population Evolution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1.05)
    
    # 4. 電場ノルムとコスト履歴
    ax4 = axes[1, 1]
    ax4.plot(iterations, field_norm_hist, 'bo-', linewidth=2, markersize=4, label='||E||')
    # コストが利用可能なら併記
    if hasattr(optimizer, 'total_cost_history'):
        ax4.plot(iterations, optimizer.total_cost_history, 'k-', linewidth=2, label='J')
    if hasattr(optimizer, 'terminal_cost_history'):
        ax4.plot(iterations, optimizer.terminal_cost_history, 'g--', linewidth=1.5, label='J_T')
    if hasattr(optimizer, 'running_cost_history'):
        ax4.plot(iterations, optimizer.running_cost_history, 'r--', linewidth=1.5, label='J_a')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Metric')
    ax4.set_title('Field Norm and Cost Evolution')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    
    # 図の保存 [[memory:2714886]]
    figures_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"krotov_optimization_results_{timestamp}.png"
    filepath = os.path.join(figures_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"結果を保存しました: {filepath}")
    
    plt.show()

    # 追加可視化: 最適電場の強度スペクトルと位相スペクトル（Ex, Ey）を波数(cm^-1)で表示、0.1 cm^-1分解能へゼロパディング
    try:
        # 時間軸・電場
        t_fs = optimal_field.get_time_SI()
        dt_fs = float(t_fs[1] - t_fs[0])
        E_t = optimal_field.get_Efield()
        N = len(t_fs)

        # 目標分解能（0.1 cm^-1）→ cycles/fs (PHz) へ変換
        df_target_PHz = float(converter.convert_frequency(0.1, "cm^-1", "PHz"))
        # 必要FFT長
        Npad = int(np.ceil(1.0 / (dt_fs * df_target_PHz)))
        Npad = max(Npad, N)

        # ゼロパディングFFT（rfft）
        E_freq = np.fft.rfft(E_t, n=Npad, axis=0)
        freq_PHz = np.fft.rfftfreq(Npad, d=dt_fs)
        # 波数(cm^-1)へ変換
        freq_cm = np.asarray(converter.convert_frequency(freq_PHz, "PHz", "cm^-1"), dtype=float)

        # 時間中心での線形位相を除去（周波数はcycles/fsを使用）
        t_center = optimizer.tlist[-1] / 2.0
        E_freq_comp = E_freq * (np.exp(1j * 2 * np.pi * freq_PHz * t_center)).reshape((len(freq_PHz), 1))

        # 強度スペクトル
        intensity_x = np.abs(E_freq_comp[:, 0]) ** 2
        intensity_y = np.abs(E_freq_comp[:, 1]) ** 2
        intensity_sum = intensity_x + intensity_y

        # 中心周波数（位相補正用）は従来どおり強度ピークで取得
        peak_idx = int(np.argmax(intensity_sum))
        f0 = float(freq_cm[peak_idx])

        # 遷移周波数（V→V+1, ΔJ=±1）の最小・最大から表示範囲を決定
        try:
            eigenvalues = optimizer.hamiltonian.get_eigenvalues()  # rad/fs
            states = optimizer.basis.basis
            energy_by_vj: dict[tuple[int, int], float] = {}
            for idx, (v, J, M) in enumerate(states):
                key = (v, J)
                if key not in energy_by_vj or M == 0:
                    energy_by_vj[key] = float(eigenvalues[idx])
            trans_wn: list[float] = []
            for (v, J), E0 in energy_by_vj.items():
                v_up = v + 1
                for dJ in (+1, -1):
                    key = (v_up, J + dJ)
                    if key in energy_by_vj:
                        d_omega = energy_by_vj[key] - E0
                        wn = float(converter.convert_frequency(d_omega, "rad/fs", "cm^-1"))
                        if np.isfinite(wn) and wn > 0:
                            trans_wn.append(wn)
            if len(trans_wn) >= 1:
                wn_min = float(np.min(trans_wn))
                wn_max = float(np.max(trans_wn))
                center = 0.5 * (wn_min + wn_max)
                span = max(wn_max - wn_min, 1e-6)
                factor = 10
                half = 0.5 * span * factor
                fmin = max(center - half, float(freq_cm[0]))
                fmax = min(center + half, float(freq_cm[-1]))
            else:
                # フォールバック（OMEGA_01 を中心に±500 cm^-1）
                fmin = max(0.0, float(OMEGA_01) - 500.0)
                fmax = float(OMEGA_01) + 500.0
        except Exception:
            fmin = max(0.0, float(OMEGA_01) - 500.0)
            fmax = float(OMEGA_01) + 500.0

        # 位相（中心波数で傾きを除去）
        phase_x_raw = np.unwrap(np.angle(E_freq_comp[:, 0]))
        phase_y_raw = np.unwrap(np.angle(E_freq_comp[:, 1]))
        dphidk_x = np.gradient(phase_x_raw, freq_cm)
        dphidk_y = np.gradient(phase_y_raw, freq_cm)
        slope_x = float(dphidk_x[peak_idx])
        slope_y = float(dphidk_y[peak_idx])
        phase_x = phase_x_raw - (slope_x * (freq_cm - f0) + phase_x_raw[peak_idx])
        phase_y = phase_y_raw - (slope_y * (freq_cm - f0) + phase_y_raw[peak_idx])

        # スライス
        mask = (freq_cm >= fmin) & (freq_cm <= fmax)
        freq_p = freq_cm[mask]
        intensity_x_p = intensity_x[mask]
        intensity_y_p = intensity_y[mask]
        phase_x_p = phase_x[mask]
        phase_y_p = phase_y[mask]

        fig2, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Ex spectrum（波数）
        ax0.plot(freq_p, intensity_x_p, color='tab:blue', label='|Ex|²')
        ax0.set_ylabel('Intensity (a.u.)')
        ax0.grid(True, alpha=0.3)
        ax0_t = ax0.twinx()
        ax0_t.plot(freq_p, phase_x_p, color='tab:red', alpha=0.7, label='Phase Ex')
        ax0_t.set_ylabel('Phase (rad)')
        ax0.set_title('Optimized Field Spectrum (Ex)')
        ax0.set_xlim(fmin, fmax)
        lines0, labels0 = ax0.get_legend_handles_labels()
        lines0_t, labels0_t = ax0_t.get_legend_handles_labels()
        ax0.legend(lines0 + lines0_t, labels0 + labels0_t, loc='upper right')

        # Ey spectrum（波数）
        ax1.plot(freq_p, intensity_y_p, color='tab:green', label='|Ey|²')
        ax1.set_xlabel('Wavenumber (cm$^{-1}$)')
        ax1.set_ylabel('Intensity (a.u.)')
        ax1.grid(True, alpha=0.3)
        ax1_t = ax1.twinx()
        ax1_t.plot(freq_p, phase_y_p, color='tab:orange', alpha=0.7, label='Phase Ey')
        ax1_t.set_ylabel('Phase (rad)')
        ax1.set_title('Optimized Field Spectrum (Ey)')
        ax1.set_xlim(fmin, fmax)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines1_t, labels1_t = ax1_t.get_legend_handles_labels()
        ax1.legend(lines1 + lines1_t, labels1 + labels1_t, loc='upper right')

        # 固有ハミルトニアンと (V, J) の選択則から遷移周波数（波数）をオーバーレイ
        try:
            eigenvalues = optimizer.hamiltonian.get_eigenvalues()  # rad/fs
            states = optimizer.basis.basis
            # (v, J) → エネルギー（M=0を優先。なければ最初のもの）
            energy_by_vj: dict[tuple[int, int], float] = {}
            for idx, (v, J, M) in enumerate(states):
                key = (v, J)
                if key not in energy_by_vj or M == 0:
                    energy_by_vj[key] = float(eigenvalues[idx])

            # R/Pブランチ（V→V+1, ΔJ=±1）
            lines_vj: list[tuple[float, str]] = []
            for (v, J), E0 in energy_by_vj.items():
                v_up = v + 1
                # R: ΔJ=+1
                key_R = (v_up, J + 1)
                if key_R in energy_by_vj:
                    d_omega = energy_by_vj[key_R] - E0  # rad/fs
                    wn = float(converter.convert_frequency(d_omega, "rad/fs", "cm^-1"))
                    if np.isfinite(wn) and wn > 0:
                        label = rf"$R({J})_{{{v}}}$"
                        lines_vj.append((wn, label))
                # P: ΔJ=-1
                key_P = (v_up, J - 1)
                if key_P in energy_by_vj:
                    d_omega = energy_by_vj[key_P] - E0  # rad/fs
                    wn = float(converter.convert_frequency(d_omega, "rad/fs", "cm^-1"))
                    if np.isfinite(wn) and wn > 0:
                        label = rf"$P({J})_{{{v}}}$"
                        lines_vj.append((wn, label))

            # 描画（範囲内のみ）
            y0 = float(np.max(intensity_x_p)) if intensity_x_p.size else 1.0
            y1 = float(np.max(intensity_y_p)) if intensity_y_p.size else 1.0
            for wn, lbl in lines_vj:
                if fmin <= wn <= fmax:
                    ax0.axvline(wn, color='gray', alpha=0.35, linewidth=0.8)
                    ax0.text(wn, y0 * 0.9, lbl, rotation=90, va='bottom', ha='center', fontsize=8, color='gray')
                    ax1.axvline(wn, color='gray', alpha=0.35, linewidth=0.8)
                    ax1.text(wn, y1 * 0.9, lbl, rotation=90, va='bottom', ha='center', fontsize=8, color='gray')
        except Exception as e_lines:
            print(f"遷移線オーバーレイでエラー: {e_lines}")

        plt.tight_layout()
        # 図の保存 [[memory:2714886]]
        figures_dir = os.path.join(os.path.dirname(__file__), "figures")
        os.makedirs(figures_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename2 = f"krotov_optimization_spectrum_{timestamp}.png"
        filepath2 = os.path.join(figures_dir, filename2)
        plt.savefig(filepath2, dpi=300, bbox_inches='tight')
        print(f"スペクトル図を保存しました: {filepath2}")
        plt.show()
    except Exception as e:
        print(f"スペクトル可視化でエラー: {e}")
    
    # 最終結果サマリー
    print(f"\n=== 最適化結果サマリー ===")
    print(f"最終フィデリティー: {fidelity_hist[-1]:.6f}")
    print(f"目標達成: {'Yes' if fidelity_hist[-1] >= TARGET_FIDELITY else 'No'}")
    print(f"反復回数: {len(fidelity_hist)}")
    print(f"最終電場ノルム: {field_norm_hist[-1]:.3e}")
    
    # 最終状態分布
    final_probs = prob_final[-1, :]
    print(f"\n最終状態分布:")
    for i, (v, J, M) in enumerate(optimizer.basis.basis):
        if final_probs[i] > 1e-4:  # 閾値以上の確率のみ表示
            print(f"  |v={v}, J={J}, M={M}⟩: {final_probs[i]:.6f}")

if __name__ == "__main__":
    main()
