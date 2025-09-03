# %%
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
import time

import matplotlib.pyplot as plt
import numpy as np

from rovibrational_excitation.core.basis import TwoLevelBasis
from rovibrational_excitation.core.electric_field import ElectricField, gaussian
from rovibrational_excitation.core.propagator import schrodinger_propagation
from rovibrational_excitation.core.basis import StateVector
from rovibrational_excitation.dipole import TwoLevelDipoleMatrix

# %%
def evaluate_consistency(time_dimensional, time_nondimensional, psi_dimensional, psi_nondimensional):
    """一致性を評価する関数"""
    prob_diff = float('inf')  # デフォルト値
    shapes_match = False
    
    if time_dimensional.shape == time_nondimensional.shape:
        time_diff = np.max(np.abs(time_dimensional - time_nondimensional))
        print(f"時間配列の最大差: {time_diff:.2e}")
        
        if psi_dimensional.shape == psi_nondimensional.shape:
            shapes_match = True
            psi_diff = np.max(np.abs(psi_dimensional - psi_nondimensional))
            print(f"波動関数の最大差: {psi_diff:.2e}")
            
            # 存在確率の比較
            prob_dimensional = np.abs(psi_dimensional)**2
            prob_nondimensional = np.abs(psi_nondimensional)**2
            prob_diff = np.max(np.abs(prob_dimensional - prob_nondimensional))
            print(f"存在確率の最大差: {prob_diff:.2e}")
            
            # 規格化の確認
            norm_dimensional = np.sum(prob_dimensional, axis=1)
            norm_nondimensional = np.sum(prob_nondimensional, axis=1)
            print(f"次元あり規格化: min={np.min(norm_dimensional):.6f}, max={np.max(norm_dimensional):.6f}")
            print(f"無次元化規格化: min={np.min(norm_nondimensional):.6f}, max={np.max(norm_nondimensional):.6f}")
        else:
            print("警告: 波動関数の形状が異なります")
    else:
        print("警告: 時間配列の形状が異なります")
    
    return shapes_match, prob_diff

def print_conclusion(shapes_match, prob_diff):
    """結論を出力する関数"""
    if shapes_match:
        if prob_diff < 1e-10:
            print("✅ 無次元化の一致性確認: 良好（差異 < 1e-10）")
        elif prob_diff < 1e-8:
            print("⚠️  無次元化の一致性確認: 許容範囲（差異 < 1e-8）")
        else:
            print("❌ 無次元化の一致性確認: 問題あり（差異 > 1e-8）")
    else:
        print("❌ 無次元化の一致性確認: 形状不一致")

# %%
print("二準位系での無次元化の一致性確認を開始...")

# パラメータ設定
energy_gap_rad_pfs = 0.4  # rad/fs
mu0_cm = 1e-30  # C·m
axes = "xy"

# %%
# 基底とハミルトニアン
basis = TwoLevelBasis()
H0 = basis.generate_H0(energy_gap_rad_pfs=energy_gap_rad_pfs, return_energy_units=True)

print(f"基底サイズ: {basis.size()}")
print(f"H0対角成分: {np.diag(H0)}")

# %%
# 双極子行列
dipole_matrix = TwoLevelDipoleMatrix(basis, mu0=mu0_cm)

print(f"双極子行列最大値: mu_x={np.max(np.abs(dipole_matrix.mu_x)):.3e}, mu_y={np.max(np.abs(dipole_matrix.mu_y)):.3e}")

# %%
# 初期状態 |0⟩ (基底状態)
state = StateVector(basis)
state.set_state(0, 1)  # 二準位系では単一のインデックス
psi0 = state.data

print(f"初期状態: |0⟩")
print(f"初期状態規格化: {np.sum(np.abs(psi0)**2):.6f}")

# %%
# 電場設定
ti, tf = 0.0, 1000  # 比較的短時間で
dt4Efield = 0.01
time4Efield = np.arange(ti, tf + 2 * dt4Efield, dt4Efield)

duration = 100
tc = (time4Efield[-1] + time4Efield[0]) / 2
amplitude = 1e10  # 適度な振幅
polarization = np.array([1, 0])

Efield = ElectricField(tlist_fs=time4Efield)
Efield.add_dispersed_Efield(
    envelope_func=gaussian,
    duration=duration,
    t_center=tc,
    carrier_freq=energy_gap_rad_pfs / (2 * np.pi),  # 共鳴周波数
    amplitude=amplitude,
    polarization=polarization,
    const_polarisation=True,  # 一定偏光で split-operator も動作確認
)

print(f"電場時間範囲: {time4Efield[0]:.1f} - {time4Efield[-1]:.1f} fs")
print(f"電場最大値: {np.max(np.abs(Efield.get_Efield())):.3e} V/m")
print(f"共鳴周波数: {energy_gap_rad_pfs / (2 * np.pi):.6f} Hz")

# %%
sample_stride = 5  # メモリ節約

# 次元ありでの計算
print("次元ありでの計算中...")
start_time = time.time()
time_dimensional, psi_dimensional = schrodinger_propagation(
    H0=H0,
    Efield=Efield,
    dipole_matrix=dipole_matrix,  # type: ignore
    psi0=psi0,
    axes=axes,
    return_traj=True,
    return_time_psi=True,
    sample_stride=sample_stride,
    nondimensional=False,
)
time_dim = time.time() - start_time
print(f"次元あり計算完了 ({time_dim:.2f}秒)")

# %%
# 無次元化での計算
print("無次元化での計算中...")
start_time = time.time()
time_nondimensional, psi_nondimensional = schrodinger_propagation(
    H0=H0,
    Efield=Efield,
    dipole_matrix=dipole_matrix,  # type: ignore
    psi0=psi0,
    axes=axes,
    return_traj=True,
    return_time_psi=True,
    sample_stride=sample_stride,
    nondimensional=True,
)
time_nondim = time.time() - start_time
print(f"無次元化計算完了 ({time_nondim:.2f}秒)")

# %%
# 結果の比較
print("\n結果の比較:")
print(f"時間配列の形状: 次元あり {time_dimensional.shape}, 無次元化 {time_nondimensional.shape}")
print(f"波動関数の形状: 次元あり {psi_dimensional.shape}, 無次元化 {psi_nondimensional.shape}")

# 数値的な比較
shapes_match, prob_diff = evaluate_consistency(time_dimensional, time_nondimensional, 
                                             psi_dimensional, psi_nondimensional)

# %%
# プロット
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# 電場
Efield_data = Efield.get_Efield()
axes[0].plot(time4Efield, Efield_data[:, 0], 'k-', alpha=0.7, label='E_x')
axes[0].set_ylabel('Electric Field [V/m]')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 基底状態存在確率の比較
axes[1].plot(time_dimensional, np.abs(psi_dimensional[:, 0])**2, 
            'b-', label='次元あり |0⟩', linewidth=2)
axes[1].plot(time_nondimensional, np.abs(psi_nondimensional[:, 0])**2, 
            'r--', label='無次元化 |0⟩', linewidth=2, alpha=0.8)
axes[1].set_ylabel('Ground State |0⟩')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 励起状態存在確率の比較
axes[2].plot(time_dimensional, np.abs(psi_dimensional[:, 1])**2, 
            'b-', label='次元あり |1⟩', linewidth=2)
axes[2].plot(time_nondimensional, np.abs(psi_nondimensional[:, 1])**2, 
            'r--', label='無次元化 |1⟩', linewidth=2, alpha=0.8)

axes[2].set_xlabel('Time [fs]')
axes[2].set_ylabel('Excited State |1⟩')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('nondimensional_consistency_twolevel.png', dpi=300, bbox_inches='tight')
print(f"\nプロットを nondimensional_consistency_twolevel.png に保存しました")

# %%
# 追加の物理的確認
print("\n物理的確認:")
print("ラビ振動の確認:")

# ラビ振動の理論予測
def theoretical_rabi_frequency(E_field_amplitude, mu_transition, hbar=1.054571817e-34):
    """理論的なラビ周波数 [rad/s]"""
    return E_field_amplitude * mu_transition / hbar

E_max = np.max(np.abs(Efield_data))
mu_transition = dipole_matrix.mu0  # 遷移双極子モーメント
omega_rabi_theory = theoretical_rabi_frequency(E_max, mu_transition)

print(f"理論的ラビ周波数: {omega_rabi_theory:.3e} rad/s = {omega_rabi_theory * 1e15:.3f} rad/fs")

# 実際の振動を確認（基底状態から励起状態への遷移）
if shapes_match:
    excited_pop_dim = np.abs(psi_dimensional[:, 1])**2
    excited_pop_nondim = np.abs(psi_nondimensional[:, 1])**2
    
    max_excitation_dim = np.max(excited_pop_dim)
    max_excitation_nondim = np.max(excited_pop_nondim)
    
    print(f"最大励起確率 - 次元あり: {max_excitation_dim:.6f}")
    print(f"最大励起確率 - 無次元化: {max_excitation_nondim:.6f}")
    print(f"最大励起確率差: {abs(max_excitation_dim - max_excitation_nondim):.2e}")

# %%
# 結論
print_conclusion(shapes_match, prob_diff)
print("\n二準位系での無次元化一致性確認が完了しました。") 
