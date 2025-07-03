#!/usr/bin/env python
"""
回転振動励起シミュレーション例
=============================

線形分子の回転振動励起における時間発展シミュレーション。
振動-回転結合効果や異なる励起条件での応答を観察。

実行方法:
    python examples/example_rovibrational_excitation.py
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from rovibrational_excitation.core.basis import LinMolBasis
from rovibrational_excitation.core.electric_field import ElectricField, gaussian_fwhm
from rovibrational_excitation.core.propagator import schrodinger_propagation
from rovibrational_excitation.core.basis import StateVector
from rovibrational_excitation.dipole.linmol import LinMolDipoleMatrix
from rovibrational_excitation.core.units import UnitConverter

# %% パラメータ設定
# システムパラメータ
V_MAX = 2  # 最大振動量子数
J_MAX = 3  # 最大回転量子数
USE_M = True  # 磁気量子数を使用

# 分子パラメータ
OMEGA_01 = 2349.0  # 振動周波数 [cm^-1]
DOMEGA = 25  # 非調和性補正 [cm^-1]
B_CONSTANT = 0.39  # 回転定数 [cm^-1]
ALPHA_CONSTANT = 0.037  # 振動-回転相互作用定数 [cm^-1]
UNIT_FREQUENCY = "cm^-1"
MU0 = 1e-30  # 双極子行列要素の大きさ [C·m]

# 時間グリッド設定
TIME_START = 0.0  # 開始時間 [fs]
TIME_END = 1000.0  # 終了時間 [fs]
DT_EFIELD = 0.01  # 電場サンプリング間隔 [fs]
SAMPLE_STRIDE = 10  # サンプリングストライド

# レーザーパルス設定
PULSE_DURATION = 100.0  # パルス幅 [fs]

# デフォルトケースの設定
DETUNING = 0.0  # デチューニング
EFIELD_AMPLITUDE = 3e10  # 電場振幅 [V/m]


# %% 基底・ハミルトニアン・双極子行列の生成
print("=== 回転振動励起シミュレーション ===")
print(f"基底サイズ: V_max={V_MAX}, J_max={J_MAX}, use_M={USE_M}")

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

print(f"基底次元: {basis.size()}")
print(f"エネルギー準位数: {len(H0.get_eigenvalues())}")

# エネルギー準位の表示
eigenvalues = H0.get_eigenvalues()
print(f"最低エネルギー: {eigenvalues[0]:.6f} rad/fs")
print(f"最高エネルギー: {eigenvalues[-1]:.6f} rad/fs")

dipole_matrix = LinMolDipoleMatrix(
    basis=basis,
    mu0=MU0,
    potential_type="harmonic",
    backend="numpy",
    dense=True,
    units="C*m"
)

# %% 初期状態の設定
state = StateVector(basis)
state.set_state((0, 0, 0), 1.0)  # 基底状態 |v=0, J=0, M=0⟩
psi0 = state.data

print(f"初期状態: |v=0, J=0, M=0⟩ (インデックス: {basis.get_index((0, 0, 0))})")

# %% 時間グリッド・電場生成
# 共鳴ケース
detuning = DETUNING
field_amplitude = EFIELD_AMPLITUDE

time4Efield = np.arange(TIME_START, TIME_END + 2 * DT_EFIELD, DT_EFIELD)
tc = (time4Efield[-1] + time4Efield[0]) / 2

Efield = ElectricField(tlist=time4Efield)
Efield.add_dispersed_Efield(
    envelope_func=gaussian_fwhm,
    duration=PULSE_DURATION,
    t_center=tc,
    carrier_freq=OMEGA_01,
    carrier_freq_units=UNIT_FREQUENCY,
    amplitude=field_amplitude,
    polarization=np.array([1, 0]),  # x方向偏光
    const_polarisation=False,
)

# %% 時間発展計算
print(f"=== 回転振動励起シミュレーション (δ={detuning:.3f}, E={field_amplitude:.3e} V/m) ===")
print("時間発展計算を開始...")

time4psi, psi_t = schrodinger_propagation(
    hamiltonian=H0,
    Efield=Efield,
    dipole_matrix=dipole_matrix,
    psi0=psi0,
    axes="zx",  # x, y方向の双極子を考慮
    return_traj=True,
    return_time_psi=True,
    sample_stride=SAMPLE_STRIDE,
)

print("時間発展計算完了.")

# %% 結果プロット
fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

# 電場の時間発展
Efield_data = Efield.get_Efield()
axes[0].plot(time4Efield, Efield_data[:, 0], "r-", linewidth=1.5, label=r"$E_x(t)$")
axes[0].plot(time4Efield, Efield_data[:, 1], "b-", linewidth=1.5, label=r"$E_y(t)$")
axes[0].set_ylabel("Electric Field [a.u.]")
axes[0].set_title(f"回転振動励起シミュレーション (δ={detuning:.3f}, E={field_amplitude:.3e} V/m)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 各状態の占有確率
prob_t = np.abs(psi_t) ** 2
total_prob = np.sum(prob_t, axis=1)

# 主要な状態のみをプロット
main_states = [(0, 0, 0), (1, 1, 0), (1, 3, 0), (2, 0, 0), (2, 2, 0)]
colors = ['b', 'r', 'g', 'orange', 'purple']

for i, (v, J, M) in enumerate(main_states):
    if (v, J, M) in basis.index_map:
        idx = basis.get_index((v, J, M))
        axes[1].plot(time4psi, prob_t[:, idx], color=colors[i], linewidth=2, 
                    label=rf"$|v={v}, J={J}, M={M}\rangle$")

axes[1].plot(time4psi, total_prob, "k--", alpha=0.7, linewidth=1, label="Total")
axes[1].set_ylabel("Population")
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(-0.05, 1.05)

# 振動準位別の占有確率
vib_probs = {}
for i, (v, J, M) in enumerate(basis.basis):
    if v not in vib_probs:
        vib_probs[v] = np.zeros_like(time4psi)
    vib_probs[v] += prob_t[:, i]

for v in sorted(vib_probs.keys()):
    axes[2].plot(time4psi, vib_probs[v], linewidth=2, label=rf"$v={v}$")

axes[2].set_ylabel("Vibrational Population")
axes[2].legend()
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim(-0.05, 1.05)

# 回転準位別の占有確率
rot_probs = {}
for i, (v, J, M) in enumerate(basis.basis):
    if J not in rot_probs:
        rot_probs[J] = np.zeros_like(time4psi)
    rot_probs[J] += prob_t[:, i]

for J in sorted(rot_probs.keys()):
    axes[3].plot(time4psi, rot_probs[J], linewidth=2, label=rf"$J={J}$")

axes[3].set_xlabel("Time [fs]")
axes[3].set_ylabel("Rotational Population")
axes[3].legend()
axes[3].grid(True, alpha=0.3)
axes[3].set_ylim(-0.05, 1.05)

plt.tight_layout()
plt.show()

# %% 励起解析
print("\n=== 励起解析 ===")

# 最大励起確率
max_excitation = np.max(prob_t[:, 1:], axis=1)  # 基底状態以外の最大値
max_overall = np.max(max_excitation)
print(f"最大励起確率: {max_overall:.4f}")

# 最終状態の分布
final_probs = prob_t[-1, :]
print(f"最終状態分布 (すべて):")
print("状態\t\t確率\t\t選択則チェック")
print("-" * 50)
for i, (v, J, M) in enumerate(basis.basis):
    prob = final_probs[i]
    # 選択則チェック: 多段階遷移を考慮
    delta_v = v - 0  # 初期状態はv=0
    delta_J = J - 0  # 初期状態はJ=0
    delta_M = M - 0  # 初期状態はM=0
    
    # 多段階遷移の選択則
    # 1. 基本選択則: Δv=±1, ΔJ=±1, ΔM=0,±1
    basic_selection_rule = (
        abs(delta_v) == 1 and  # 振動選択則: Δv = ±1
        abs(delta_J) == 1 and  # 回転選択則: ΔJ = ±1
        abs(delta_M) <= 1      # 磁気選択則: ΔM = 0, ±1
    )
    
    # 2. 多段階遷移選択則: 振動準位の変化が偶数の時、J,Mの変化も偶数
    multistep_selection_rule = True
    if abs(delta_v) % 2 == 0:  # 振動変化が偶数
        if abs(delta_J) % 2 != 0 or abs(delta_M) % 2 != 0:  # J,Mの変化が奇数
            multistep_selection_rule = False
    
    # 3. 多段階遷移選択則: 振動準位の変化が奇数の時、J,Mの変化も奇数
    if abs(delta_v) % 2 == 1:  # 振動変化が奇数
        if abs(delta_J) % 2 != 1 or abs(delta_M) % 2 != 1:  # J,Mの変化が偶数
            multistep_selection_rule = False
    
    # 総合的な選択則チェック
    selection_rule_ok = basic_selection_rule or multistep_selection_rule
    
    status = "✓" if selection_rule_ok else "✗"
    print(f"|v={v}, J={J}, M={M}⟩\t{prob:.6f}\t{status}")
    
    # 非ゼロ確率を持つ状態の選択則違反を警告
    if prob > 1e-6 and not selection_rule_ok:
        print(f"    ⚠️  選択則違反: Δv={delta_v}, ΔJ={delta_J}, ΔM={delta_M}")
        if not basic_selection_rule:
            print(f"        - 基本選択則違反")
        if not multistep_selection_rule:
            print(f"        - 多段階遷移選択則違反")

# 選択則統計
print("\n=== 選択則統計 ===")
total_states = len(basis.basis)
nonzero_states = np.sum(final_probs > 1e-6)
selection_rule_compliant = 0
selection_rule_violations = 0

for i, (v, J, M) in enumerate(basis.basis):
    prob = final_probs[i]
    if prob > 1e-6:
        delta_v = v - 0
        delta_J = J - 0
        delta_M = M - 0
        
        # 基本選択則
        basic_selection_rule = (
            abs(delta_v) <= 1 and
            abs(delta_J) <= 1 and
            abs(delta_M) <= 1
        )
        
        # 多段階遷移選択則
        multistep_selection_rule = True
        if abs(delta_v) % 2 == 0:  # 振動変化が偶数
            if abs(delta_J) % 2 != 0 or abs(delta_M) % 2 != 0:  # J,Mの変化が奇数
                multistep_selection_rule = False
        
        if abs(delta_v) % 2 == 1:  # 振動変化が奇数
            if abs(delta_J) % 2 != 1 or abs(delta_M) % 2 != 1:  # J,Mの変化が偶数
                multistep_selection_rule = False
        
        # 総合的な選択則チェック
        selection_rule_ok = basic_selection_rule or multistep_selection_rule
        
        if selection_rule_ok:
            selection_rule_compliant += 1
        else:
            selection_rule_violations += 1

print(f"全状態数: {total_states}")
print(f"非ゼロ確率状態数: {nonzero_states}")
print(f"選択則準拠状態数: {selection_rule_compliant}")
print(f"選択則違反状態数: {selection_rule_violations}")
print(f"選択則準拠率: {selection_rule_compliant/max(nonzero_states, 1)*100:.1f}%")

# 振動励起効率
vib_excitation = 0.0
for i, (v, J, M) in enumerate(basis.basis):
    if v > 0:  # v=0以外の状態
        vib_excitation += final_probs[i]
print(f"振動励起効率: {vib_excitation:.4f}")

# 回転励起効率
rot_excitation = 0.0
for i, (v, J, M) in enumerate(basis.basis):
    if J > 0:  # J=0以外の状態
        rot_excitation += final_probs[i]
print(f"回転励起効率: {rot_excitation:.4f}")

# %% エネルギー解析
print("\n=== エネルギー解析 ===")

# 期待値の計算
energy_expectation = np.zeros_like(time4psi)
H0_matrix = H0.get_matrix()  # ハミルトニアン行列を取得
for i, t in enumerate(time4psi):
    psi_state = psi_t[i, :]
    energy_expectation[i] = np.real(np.conj(psi_state) @ H0_matrix @ psi_state)

# エネルギー変化
energy_change = energy_expectation - energy_expectation[0]
print(f"エネルギー変化: {energy_change[-1]:.6f} rad/fs")

# エネルギー変化のプロット
plt.figure(figsize=(10, 6))
plt.plot(time4psi, energy_change, 'b-', linewidth=2)
plt.xlabel('Time [fs]')
plt.ylabel('Energy Change [rad/fs]')
plt.title('エネルギー期待値の時間発展')
plt.grid(True, alpha=0.3)
plt.show()

print("シミュレーション完了.") 