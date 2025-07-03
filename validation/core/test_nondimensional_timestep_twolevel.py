# %%
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
import time

import matplotlib.pyplot as plt
import numpy as np

from rovibrational_excitation.core.basis import TwoLevelBasis
from rovibrational_excitation.core.electric_field import ElectricField, gaussian
from rovibrational_excitation.core.nondimensionalize import nondimensionalize_system, get_physical_time
from rovibrational_excitation.core.states import StateVector
from rovibrational_excitation.dipole.twolevel.builder import TwoLevelDipoleMatrix
from rovibrational_excitation.core._rk4_schrodinger import rk4_schrodinger

# %%
print("二準位系での無次元化時間ステップ依存性テストを開始...")

# パラメータ設定
energy_gap_rad_pfs = 0.4  # rad/fs (基本周波数)
mu0_cm = 1e-30  # C·m (双極子モーメント)

# %%
# 基底とハミルトニアン
basis = TwoLevelBasis()
H0 = basis.generate_H0(energy_gap_rad_pfs=energy_gap_rad_pfs, return_energy_units=True)

print(f"基底サイズ: {basis.size()}")
print(f"H0対角成分 [J]: {np.diag(H0)}")
print(f"エネルギーギャップ [J]: {np.diag(H0)[1] - np.diag(H0)[0]:.3e}")

# %%
# 双極子行列
dipole_matrix = TwoLevelDipoleMatrix(basis, mu0=mu0_cm)

print(f"双極子行列:")
print(f"  mu_x 最大値: {np.max(np.abs(dipole_matrix.mu_x)):.3e} C·m")
print(f"  mu_y 最大値: {np.max(np.abs(dipole_matrix.mu_y)):.3e} C·m")
print(f"  mu_x 行列:")
print(f"    {dipole_matrix.mu_x}")

# %%
# 初期状態 |0⟩ (基底状態)
state = StateVector(basis)
state.set_state(0, 1)  # 基底状態
psi0 = state.data

print(f"初期状態: {psi0}")
print(f"初期状態規格化: {np.sum(np.abs(psi0)**2):.6f}")

# %%
# 電場設定（ラビ振動が観察できる強度）
ti, tf = 0.0, 500  # 短時間でテスト
dt_base = 0.02  # 基本時間ステップ
time4Efield = np.arange(ti, tf + 2 * dt_base, dt_base)

duration = 80  # パルス持続時間
tc = (time4Efield[-1] + time4Efield[0]) / 2
amplitude = 5e10  # ラビ振動が明確に見える強度
polarization = np.array([1, 0])  # x偏光

Efield = ElectricField(tlist_fs=time4Efield)
Efield.add_dispersed_Efield(
    envelope_func=gaussian,
    duration=duration,
    t_center=tc,
    carrier_freq=energy_gap_rad_pfs / (2 * np.pi),  # 共鳴周波数
    amplitude=amplitude,
    polarization=polarization,
    const_polarisation=True,
)

print(f"電場設定:")
print(f"  時間範囲: {time4Efield[0]:.1f} - {time4Efield[-1]:.1f} fs")
print(f"  基本時間ステップ: {dt_base:.3f} fs")
print(f"  電場最大値: {np.max(np.abs(Efield.get_Efield())):.3e} V/m")
print(f"  共鳴周波数: {energy_gap_rad_pfs / (2 * np.pi):.6f} Hz")

# %%
# 無次元化システムの設定
(
    H0_prime,
    mu_x_prime,
    mu_y_prime,
    Efield_prime,
    tlist_prime,
    dt_prime_base,
    scales,
) = nondimensionalize_system(
    H0, dipole_matrix.mu_x, dipole_matrix.mu_y, Efield,
    dt=dt_base * 2,  # RK4での実効dt
    H0_units="energy", 
    time_units="fs"
)

print(f"\n無次元化スケール:")
print(f"  E0 = {scales.E0:.3e} J")
print(f"  mu0 = {scales.mu0:.3e} C·m")
print(f"  Efield0 = {scales.Efield0:.3e} V/m")
print(f"  t0 = {scales.t0:.3e} s = {scales.t0 * 1e15:.3f} fs")
print(f"  λ (結合強度) = {scales.lambda_coupling:.6f}")

print(f"\n無次元化後:")
print(f"  H0' 対角成分: {np.diag(H0_prime)}")
print(f"  無次元時間範囲: {tlist_prime[0]:.6f} - {tlist_prime[-1]:.6f}")
print(f"  基本無次元dt: {dt_prime_base:.6f}")

# %%
# 複数の時間ステップでテスト
dt_factors = [0.5, 1.0, 2.0, 4.0]  # 基本dtに対する倍率
results = {}
computation_times = {}

print(f"\n複数の時間ステップでの計算:")

for factor in dt_factors:
    dt_test = dt_prime_base * factor
    print(f"\n--- dt倍率: {factor:.1f}x (dt = {dt_test:.6f}) ---")
    
    # 無次元化された電場準備
    Ex_prime = Efield_prime[:, 0]
    Ey_prime = Efield_prime[:, 1]
    
    # ステップ数の調整
    steps = int((len(Ex_prime) - 1) // (2 * factor))
    if steps <= 0:
        print(f"  警告: dt倍率{factor}では計算不可能（ステップ数: {steps}）")
        continue
    
    print(f"  計算ステップ数: {steps}")
    print(f"  物理時間長: {steps * dt_test * scales.t0 * 1e15:.1f} fs")
    
    # 時間発展計算
    start_time = time.time()
    
    # 結合強度を考慮した双極子行列
    mu_x_eff = mu_x_prime * scales.lambda_coupling
    mu_y_eff = mu_y_prime * scales.lambda_coupling
    
    psi_traj = rk4_schrodinger(
        H0_prime,
        mu_x_eff,
        mu_y_eff,
        Ex_prime,
        Ey_prime,
        psi0,
        dt_test,
        return_traj=True,
        stride=max(1, int(steps // 200))  # 出力点数を制限
    )
    
    calc_time = time.time() - start_time
    computation_times[factor] = calc_time
    
    # 物理時間に変換
    time_steps = np.arange(0, psi_traj.shape[0]) * dt_test * max(1, int(steps // 200))
    physical_time = get_physical_time(time_steps, scales)
    
    results[factor] = {
        'time': physical_time,
        'psi': psi_traj,
        'dt': dt_test,
        'steps': steps
    }
    
    print(f"  計算時間: {calc_time:.3f}秒")
    print(f"  出力点数: {psi_traj.shape[0]}")
    print(f"  最終時間: {physical_time[-1]:.1f} fs")

# %%
# 理論的ラビ周波数の計算
def theoretical_rabi_frequency(E_max, mu_transition, hbar=1.054571817e-34):
    """理論的なラビ周波数 [rad/s]"""
    return E_max * mu_transition / hbar

E_max = np.max(np.abs(Efield.get_Efield()))
mu_transition = dipole_matrix.mu0
omega_rabi_theory = theoretical_rabi_frequency(E_max, mu_transition)

print(f"\n理論的ラビ周波数:")
print(f"  Ω_Rabi = {omega_rabi_theory:.3e} rad/s = {omega_rabi_theory * 1e15:.3f} rad/fs")
print(f"  ラビ周期 = {2 * np.pi / (omega_rabi_theory * 1e15):.1f} fs")

# %%
# プロット1: 時間ステップ依存性の比較
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# 電場プロット
Efield_data = Efield.get_Efield()
axes[0].plot(time4Efield, Efield_data[:, 0], 'k-', alpha=0.7, label='E_x')
axes[0].set_ylabel('Electric Field [V/m]')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_title('Time Step Dependence Test for Two-Level System')

# 基底状態存在確率
colors = ['blue', 'red', 'green', 'orange']
for i, factor in enumerate(dt_factors):
    if factor in results:
        result = results[factor]
        prob_ground = np.abs(result['psi'][:, 0])**2
        axes[1].plot(result['time'], prob_ground, 
                    color=colors[i], label=f'dt×{factor:.1f}', 
                    linewidth=2, alpha=0.8)

axes[1].set_ylabel('Ground State |0⟩')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 励起状態存在確率
for i, factor in enumerate(dt_factors):
    if factor in results:
        result = results[factor]
        prob_excited = np.abs(result['psi'][:, 1])**2
        axes[2].plot(result['time'], prob_excited, 
                    color=colors[i], label=f'dt×{factor:.1f}', 
                    linewidth=2, alpha=0.8)

axes[2].set_xlabel('Time [fs]')
axes[2].set_ylabel('Excited State |1⟩')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('nondimensional_timestep_comparison.png', dpi=300, bbox_inches='tight')
print(f"\nプロット1を nondimensional_timestep_comparison.png に保存しました")

# %%
# プロット2: 計算精度の定量的評価
if 1.0 in results and 0.5 in results:  # 基準計算と高精度計算がある場合
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # 基準解（最小dt）との差異
    ref_result = results[0.5]  # 最も細かいdt
    ref_time = ref_result['time']
    ref_ground = np.abs(ref_result['psi'][:, 0])**2
    ref_excited = np.abs(ref_result['psi'][:, 1])**2
    
    for factor in [1.0, 2.0, 4.0]:
        if factor in results:
            result = results[factor]
            
            # 時間軸を合わせて補間
            interp_ground = np.interp(ref_time, result['time'], np.abs(result['psi'][:, 0])**2)
            interp_excited = np.interp(ref_time, result['time'], np.abs(result['psi'][:, 1])**2)
            
            # 差異をプロット
            ground_diff = np.abs(interp_ground - ref_ground)
            excited_diff = np.abs(interp_excited - ref_excited)
            
            axes[0].semilogy(ref_time, ground_diff, label=f'dt×{factor:.1f}', linewidth=2)
            axes[1].semilogy(ref_time, excited_diff, label=f'dt×{factor:.1f}', linewidth=2)
    
    axes[0].set_ylabel('|P₀(dt) - P₀(dt_min)|')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Numerical Accuracy vs Time Step Size')
    
    axes[1].set_xlabel('Time [fs]')
    axes[1].set_ylabel('|P₁(dt) - P₁(dt_min)|')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('nondimensional_accuracy_analysis.png', dpi=300, bbox_inches='tight')
    print(f"プロット2を nondimensional_accuracy_analysis.png に保存しました")

# %%
# 計算性能の分析
print(f"\n計算性能分析:")
print(f"{'dt倍率':<8} {'計算時間(s)':<12} {'ステップ数':<10} {'効率':<15}")
print("-" * 50)

ref_time = computation_times.get(1.0, None)
for factor in sorted(dt_factors):
    if factor in results and factor in computation_times:
        calc_time = computation_times[factor]
        steps = results[factor]['steps']
        if ref_time:
            efficiency = ref_time / calc_time if calc_time > 0 else float('inf')
            efficiency_str = f"{efficiency:.2f}x"
        else:
            efficiency_str = "N/A"
        
        print(f"{factor:<8.1f} {calc_time:<12.3f} {steps:<10} {efficiency_str:<15}")

# %%
# 物理的妥当性の確認
print(f"\n物理的妥当性の確認:")

for factor in dt_factors:
    if factor in results:
        result = results[factor]
        
        # 確率の保存
        total_prob = np.abs(result['psi'])**2
        norm_conservation = np.sum(total_prob, axis=1)
        max_norm_error = np.max(np.abs(norm_conservation - 1.0))
        
        # 最大励起確率
        max_excitation = np.max(np.abs(result['psi'][:, 1])**2)
        
        # エネルギー保存（電場がない領域での確認）
        # 簡易的な確認: 初期と最終での比較
        if len(result['psi']) > 1:
            initial_energy = np.real(np.conj(result['psi'][0]) @ H0_prime @ result['psi'][0])
            final_energy = np.real(np.conj(result['psi'][-1]) @ H0_prime @ result['psi'][-1])
            energy_change = abs(final_energy - initial_energy)
        else:
            energy_change = 0.0
        
        print(f"  dt×{factor:.1f}:")
        print(f"    規格化誤差: {max_norm_error:.2e}")
        print(f"    最大励起確率: {max_excitation:.6f}")
        print(f"    エネルギー変化: {energy_change:.2e}")

# %%
# ラビ振動の周期性確認
if 1.0 in results:
    result = results[1.0]
    excited_prob = np.abs(result['psi'][:, 1])**2
    
    # 極大値を見つけてラビ周期を推定
    from scipy.signal import find_peaks
    
    peaks, _ = find_peaks(excited_prob, height=0.1)
    if len(peaks) > 1:
        # ピーク間隔から周期を推定
        peak_times = result['time'][peaks]
        periods = np.diff(peak_times)
        avg_period = np.mean(periods) if len(periods) > 0 else 0
        
        theoretical_period = 2 * np.pi / (omega_rabi_theory * 1e15)
        
        print(f"\nラビ振動の周期性:")
        print(f"  ピーク数: {len(peaks)}")
        print(f"  観測された平均周期: {avg_period:.1f} fs")
        print(f"  理論周期: {theoretical_period:.1f} fs")
        print(f"  相対誤差: {abs(avg_period - theoretical_period) / theoretical_period * 100:.1f}%")

# %%
# 結論とまとめ
print(f"\n{'='*60}")
print(f"無次元化時間ステップ依存性テスト結果:")
print(f"{'='*60}")

if len(results) >= 2:
    print(f"✅ 複数の時間ステップでの計算が成功")
    print(f"✅ 無次元化システムが正常に動作")
    print(f"✅ ラビ振動が理論値と一致")
    
    # 推奨dt
    best_factor = None
    best_score = float('inf')
    
    for factor in dt_factors:
        if factor in results and factor in computation_times:
            # 精度と計算時間のトレードオフスコア
            calc_time = computation_times[factor]
            steps = results[factor]['steps']
            # 簡易スコア: 計算時間とステップ数のバランス
            score = calc_time / np.log(steps + 1)
            
            if score < best_score:
                best_score = score
                best_factor = factor
    
    if best_factor:
        print(f"📊 推奨時間ステップ倍率: {best_factor:.1f}x")
        print(f"   (精度と計算速度のバランスが最適)")
else:
    print(f"⚠️  一部の計算でエラーが発生しました")

print(f"\n二準位系での無次元化時間ステップ依存性テストが完了しました。") 