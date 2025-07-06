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
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from rovibrational_excitation.core.basis import LinMolBasis
from rovibrational_excitation.core.electric_field import ElectricField, gaussian
from rovibrational_excitation.core.propagator import schrodinger_propagation
from rovibrational_excitation.core.basis import StateVector
from rovibrational_excitation.dipole.linmol import LinMolDipoleMatrix
from rovibrational_excitation.core.units.converters import converter
from rovibrational_excitation.core.units.constants import CONSTANTS

# %% パラメータ設定
# システムパラメータ
SPARSE = True
DENSE = False
# SPARSE = False
V_MAX = 10  # 最大振動量子数
J_MAX = 10  # 最大回転量子数
USE_M = True  # 磁気量子数を使用

# 分子パラメータ
OMEGA_01 = 2349.1  # 振動周波数 [cm^-1]
DOMEGA = 25  # 非調和性補正 [cm^-1]
B_CONSTANT = 0.39  # 回転定数 [cm^-1]
ALPHA_CONSTANT = 0.037  # 振動-回転相互作用定数 [cm^-1]
UNIT_FREQUENCY = "cm^-1"
MU0 = 1e-30  # 双極子行列要素の大きさ [C·m]
UNIT_DIPOLE = "C*m"

# レーザーパルス設定
PULSE_DURATION = 50.0  # パルス幅 [fs]
DETUNING = 0.0  # デチューニング
# EFIELD_AMPLITUDE = 7e9  # 電場振幅 [V/m]
cycle = 0.5 * np.sqrt(2)
EFIELD_AMPLITUDE = (
    cycle * 2 * np.pi * CONSTANTS.HBAR
    / (
        converter.convert_dipole_moment(MU0, UNIT_DIPOLE, "C*m")
        * converter.convert_time(PULSE_DURATION, "fs", "s")
        * np.sqrt(2*np.pi)  # gaussianの時間積分値
        )
    * np.sqrt(2/3)  # μx(0,0,1,1)*2
    )
print(f"EFIELD_AMPLITUDE: {EFIELD_AMPLITUDE:.3e} V/m")
POLARIZATION_1st = np.array([1, 0])  # x方向偏光
POLARIZATION_2nd = np.array([0, 1])  # y方向偏光
DELAY = -1/converter.convert_frequency(OMEGA_01, UNIT_FREQUENCY, "PHz")/4
AXES = "xy"  # x, y方向の双極子を考慮

# 時間グリッド設定
TIME_START = 0.0  # 開始時間 [fs]
TIME_END = PULSE_DURATION*10  # 終了時間 [fs]
DT_EFIELD = 0.1  # 電場サンプリング間隔 [fs]
SAMPLE_STRIDE = 1  # サンプリングストライド


# plot setting
def plot_states():
    states = []
    
    for V in range(V_MAX+1):
        for J in range(J_MAX+1):
            for M in range(-J, J+1):
                if AXES == "xy":
                    if (V + J) % 2 == 0 and (J + M)%2 == 0:
                        states.append((V, J, M))
                elif AXES == "zx":
                    if (V + J) % 2 == 0 and M == 0:
                        states.append((V, J, M))
                else:
                    raise ValueError(f"Invalid axes: {AXES}")
    return states


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
    dense=DENSE,
    units="C*m",
    units_input=UNIT_DIPOLE,
)

# %% 初期状態の設定
state = StateVector(basis)
state.set_state((0, 0, 0), 1.0)  # 基底状態 |v=0, J=0, M=0⟩
psi0 = state.data

print(f"初期状態: |v=0, J=0, M=0⟩ (インデックス: {basis.get_index((0, 0, 0))})")

# %% 時間グリッド・電場生成
# 共鳴ケース


time4Efield = np.arange(TIME_START, TIME_END + 2 * DT_EFIELD, DT_EFIELD)
tc = (time4Efield[-1] + time4Efield[0]) / 2

carrier_freq = converter.convert_frequency(
    H0.eigenvalues[basis.get_index((1, 1, 1))] - H0.eigenvalues[basis.get_index((0, 0, 0))],
    "rad/fs", UNIT_FREQUENCY
    )
Efield = ElectricField(tlist=time4Efield)
Efield.add_dispersed_Efield(
    envelope_func=gaussian,
    duration=PULSE_DURATION,
    t_center=tc,
    carrier_freq=carrier_freq,
    carrier_freq_units=UNIT_FREQUENCY,
    amplitude=EFIELD_AMPLITUDE,
    polarization=POLARIZATION_1st,  # x方向偏光
    const_polarisation=False,
)
Efield.add_dispersed_Efield(
    envelope_func=gaussian,
    duration=PULSE_DURATION,
    t_center=tc+DELAY,
    carrier_freq=carrier_freq,
    carrier_freq_units=UNIT_FREQUENCY,
    amplitude=EFIELD_AMPLITUDE,
    polarization=POLARIZATION_2nd,  # x方向偏光
    const_polarisation=False,
)

# %% 時間発展計算
print(f"=== 回転振動励起シミュレーション (δ={DETUNING:.3f}, E={EFIELD_AMPLITUDE:.3e} V/m) ===")
print("時間発展計算を開始...")
start = time.perf_counter()
time4psi, psi_t = schrodinger_propagation(
    hamiltonian=H0,
    Efield=Efield,
    dipole_matrix=dipole_matrix,
    psi0=psi0,
    axes=AXES,  # x, y方向の双極子を考慮
    return_traj=True,
    return_time_psi=True,
    sample_stride=SAMPLE_STRIDE,
    sparse=SPARSE,
)
end = time.perf_counter()
print(f"時間発展計算完了. 計算時間: {end - start:.3f} s")


# %% 結果プロット
fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

# 電場の時間発展
Efield_data = Efield.get_Efield()
axes[0].plot(time4Efield, Efield_data[:, 0], "r-", linewidth=1.5, label=r"$E_x(t)$")
axes[0].plot(time4Efield, Efield_data[:, 1], "b-", linewidth=1.5, label=r"$E_y(t)$")
axes[0].set_ylabel("Electric Field [a.u.]")
axes[0].set_title(f"Rovibrational Excitation, δ={DETUNING:.3f}, E={EFIELD_AMPLITUDE:.3e} V/m")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 各状態の占有確率
prob_t = np.abs(psi_t) ** 2
total_prob = np.sum(prob_t, axis=1)

# 主要な状態のみをプロット
main_states = plot_states()

for i, (v, J, M) in enumerate(main_states):
    if (v, J, M) in basis.index_map:
        idx = basis.get_index((v, J, M))
        axes[1].plot(time4psi, prob_t[:, idx], linewidth=2, 
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
if False:
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
        # Selection rule check: Consider multistep transition
        delta_v = v - 0  # Initial state is v=0
        delta_J = J - 0  # Initial state is J=0
        delta_M = M - 0  # Initial state is M=0
        
        # Multistep transition selection rule
        # 1. Basic selection rule: Δv=±1, ΔJ=±1, ΔM=0,±1
        basic_selection_rule = (
            abs(delta_v) == 1 and  # Vibrational selection rule: Δv = ±1
            abs(delta_J) == 1 and  # Rotational selection rule: ΔJ = ±1
            abs(delta_M) <= 1      # Magnetic selection rule: ΔM = 0, ±1
        )
        
        # 2. Multistep transition selection rule: When the vibrational level changes even, the change of J and M is also even
        multistep_selection_rule = True
        if abs(delta_v) % 2 == 0:  # When the vibrational level changes even
            if abs(delta_J) % 2 != 0 or abs(delta_M) % 2 != 0:  # When the change of J and M is odd
                multistep_selection_rule = False
        
        # 3. Multistep transition selection rule: When the vibrational level changes odd, the change of J and M is also odd
        if abs(delta_v) % 2 == 1:  # When the vibrational level changes odd
            if abs(delta_J) % 2 != 1 or abs(delta_M) % 2 != 1:  # When the change of J and M is even
                multistep_selection_rule = False
        
        # Overall selection rule check
        selection_rule_ok = basic_selection_rule or multistep_selection_rule
        
        status = "✓" if selection_rule_ok else "✗"
        print(f"|v={v}, J={J}, M={M}⟩\t{prob:.6f}\t{status}")
        
        # Warning for states with non-zero probability and selection rule violation
        if prob > 1e-6 and not selection_rule_ok:
            print(f"    ⚠️  Selection rule violation: Δv={delta_v}, ΔJ={delta_J}, ΔM={delta_M}")
            if not basic_selection_rule:
                print(f"        - Basic selection rule violation")
            if not multistep_selection_rule:
                print(f"        - Multistep transition selection rule violation")

    # Selection rule statistics
    print("\n=== Selection rule statistics ===")
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

    print(f"Total states: {total_states}")
    print(f"Non-zero probability states: {nonzero_states}")
    print(f"Selection rule compliant states: {selection_rule_compliant}")
    print(f"Selection rule violations: {selection_rule_violations}")
    print(f"Selection rule compliance: {selection_rule_compliant/max(nonzero_states, 1)*100:.1f}%")

    # Vibrational Excitation Efficiency
    vib_excitation = 0.0
    for i, (v, J, M) in enumerate(basis.basis):
        if v > 0:  # v=0以外の状態
            vib_excitation += final_probs[i]
    print(f"Vibrational Excitation Efficiency: {vib_excitation:.4f}")

    # Rotational Excitation Efficiency
    rot_excitation = 0.0
    for i, (v, J, M) in enumerate(basis.basis):
        if J > 0:  # J=0以外の状態
            rot_excitation += final_probs[i]
    print(f"Rotational Excitation Efficiency: {rot_excitation:.4f}")

# %% Energy Analysis
if False:
    print("\n=== Energy Analysis ===")

    # 期待値の計算
    energy_expectation = np.zeros_like(time4psi)
    H0_matrix = H0.get_matrix()  # ハミルトニアン行列を取得
    for i, t in enumerate(time4psi):
        psi_state = psi_t[i, :]
        energy_expectation[i] = np.real(np.conj(psi_state) @ H0_matrix @ psi_state)

    # Energy change
    energy_change = energy_expectation - energy_expectation[0]
    print(f"Energy change: {energy_change[-1]:.6f} rad/fs")

    # Energy change plot
    plt.figure(figsize=(10, 6))
    plt.plot(time4psi, energy_change, 'b-', linewidth=2)
    plt.xlabel('Time [fs]')
    plt.ylabel('Energy Change [rad/fs]')
    plt.title('Energy Expectation Time Evolution')
    plt.grid(True, alpha=0.3)
    plt.show()

    print("Simulation finished.") 