#!/usr/bin/env python
"""
振動励起シミュレーション例
======================

純粋な振動系（回転なし）の時間発展シミュレーション。
調和振動子とモース振動子の両方をテスト。

実行方法:
    python examples/example_vibrational_excitation.py
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from rovibrational_excitation.core.basis import VibLadderBasis
from rovibrational_excitation.core.electric_field import ElectricField, gaussian
from rovibrational_excitation.core.propagator import schrodinger_propagation
from rovibrational_excitation.core.basis import StateVector
from rovibrational_excitation.dipole.viblad import VibLadderDipoleMatrix
from rovibrational_excitation.core.units.converters import converter

# %% パラメータ設定
# システムパラメータ
SPARSE = True
SPARSE = False

V_MAX = 2  # 最大振動量子数
OMEGA_01 = 2349.1  # 振動周波数 [cm^-1]
DOMEGA = 25  # 非調和性補正 [cm^-1]
MU0 = 1e-30  # 双極子行列要素の大きさ [C·m]
UNIT_FREQUENCY = "cm^-1"
UNIT_DIPOLE = "C*m"

# レーザーパルス設定
PULSE_DURATION = 50.0  # パルス幅 [fs]
EFIELD_AMPLITUDE = 5e9  # 電場振幅 [V/m]
POLARIZATION = np.array([1, 0])  # x方向偏光
AXES = "zx"  # z, x方向の双極子を考慮

# 時間グリッド設定
TIME_START = 0.0  # 開始時間 [fs]
TIME_END = PULSE_DURATION * 10   # 終了時間 [fs]
DT_EFIELD = 0.1  # 電場サンプリング間隔 [fs]
SAMPLE_STRIDE = 5  # サンプリングストライド

# %% 基底・ハミルトニアン・双極子行列の生成
print("=== 振動励起シミュレーション ===")
print(f"基底サイズ: V_max={V_MAX}")

# 調和振動子の場合
basis_h = VibLadderBasis(
    V_max=V_MAX,
    omega=OMEGA_01,
    delta_omega=DOMEGA,
    input_units=UNIT_FREQUENCY,
    output_units="rad/fs"
)
H0_h = basis_h.generate_H0()

print(f"基底次元 (調和振動子): {basis_h.size()}")
print(f"エネルギー準位数: {len(H0_h.get_eigenvalues())}")

# モース振動子の場合
basis_m = VibLadderBasis(
    V_max=V_MAX,
    omega=OMEGA_01,
    delta_omega=DOMEGA,
    input_units=UNIT_FREQUENCY,
    output_units="rad/fs"
)
H0_m = basis_m.generate_H0()

print(f"基底次元 (モース振動子): {basis_m.size()}")
print(f"エネルギー準位数: {len(H0_m.get_eigenvalues())}")

# エネルギー準位の表示
eigenvalues_h = H0_h.get_eigenvalues()
eigenvalues_m = H0_m.get_eigenvalues()
print("\n=== エネルギー準位 ===")
print("調和振動子:")
for i, e in enumerate(eigenvalues_h):
    print(f"v={i}: {e:.6f} rad/fs")
print("\nモース振動子:")
for i, e in enumerate(eigenvalues_m):
    print(f"v={i}: {e:.6f} rad/fs")

# 双極子行列の生成
dipole_matrix_h = VibLadderDipoleMatrix(
    basis=basis_h,
    mu0=MU0,
    potential_type="harmonic",
    units="C*m",
    units_input=UNIT_DIPOLE,
)

dipole_matrix_m = VibLadderDipoleMatrix(
    basis=basis_h,
    mu0=MU0,
    potential_type="morse",
    units="C*m",
    units_input=UNIT_DIPOLE,
)

# %% 初期状態の設定
# 調和振動子
state_h = StateVector(basis_h)
state_h.set_state((0,), 1.0)  # 基底状態 |v=0⟩
psi0_h = state_h.data

# モース振動子
state_m = StateVector(basis_m)
state_m.set_state((0,), 1.0)  # 基底状態 |v=0⟩
psi0_m = state_m.data

print(f"\n初期状態: |v=0⟩")

# %% 時間グリッド・電場生成
time4Efield = np.arange(TIME_START, TIME_END + 2 * DT_EFIELD, DT_EFIELD)
tc = (time4Efield[-1] + time4Efield[0]) / 2

# 共鳴周波数の計算
carrier_freq = float(converter.convert_frequency(OMEGA_01, UNIT_FREQUENCY, "rad/fs"))  # float型に明示的に変換

# 電場の生成
Efield = ElectricField(tlist=time4Efield)
Efield.add_dispersed_Efield(
    envelope_func=gaussian,
    duration=PULSE_DURATION,
    t_center=tc,
    carrier_freq=carrier_freq,
    carrier_freq_units='rad/fs',
    amplitude=EFIELD_AMPLITUDE,
    polarization=POLARIZATION,
    const_polarisation=False,
)

# %% 時間発展計算
print(f"=== 振動励起シミュレーション (E={EFIELD_AMPLITUDE:.3e} V/m) ===")

# 調和振動子
print("\n調和振動子の時間発展計算を開始...")
start = time.perf_counter()
time4psi_h, psi_t_h = schrodinger_propagation(
    hamiltonian=H0_h,
    Efield=Efield,
    dipole_matrix=dipole_matrix_h,
    psi0=psi0_h,
    axes=AXES,
    return_traj=True,
    return_time_psi=True,
    sample_stride=SAMPLE_STRIDE,
    sparse=SPARSE,
)
end = time.perf_counter()
print(f"時間発展計算完了. 計算時間: {end - start:.3f} s")

# モース振動子
print("\nモース振動子の時間発展計算を開始...")
start = time.perf_counter()
time4psi_m, psi_t_m = schrodinger_propagation(
    hamiltonian=H0_m,
    Efield=Efield,
    dipole_matrix=dipole_matrix_m,
    psi0=psi0_m,
    axes=AXES,
    return_traj=True,
    return_time_psi=True,
    sample_stride=SAMPLE_STRIDE,
    sparse=SPARSE,
)
end = time.perf_counter()
print(f"時間発展計算完了. 計算時間: {end - start:.3f} s")

# %% 結果プロット
# メインプロット（調和振動子とモース振動子の比較）
fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

# 電場の時間発展
Efield_data = Efield.get_Efield()
axes[0].plot(time4Efield, Efield_data[:, 0], "r-", linewidth=1.5, label=r"$E_x(t)$")
axes[0].plot(time4Efield, Efield_data[:, 1], "b-", linewidth=1.5, label=r"$E_y(t)$")
axes[0].set_ylabel("Electric Field [V/m]")
axes[0].set_title(f"Vibrational Excitation, E={EFIELD_AMPLITUDE:.3e} V/m")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 調和振動子の占有確率
prob_t_h = np.abs(psi_t_h) ** 2
total_prob_h = np.sum(prob_t_h, axis=1)

for v in range(V_MAX + 1):
    axes[1].plot(time4psi_h, prob_t_h[:, v], linewidth=2, label=f"|v={v}⟩")

axes[1].plot(time4psi_h, total_prob_h, "k--", alpha=0.7, linewidth=1, label="Total")
axes[1].set_ylabel("Population (Harmonic)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(-0.05, 1.05)

# モース振動子の占有確率
prob_t_m = np.abs(psi_t_m) ** 2
total_prob_m = np.sum(prob_t_m, axis=1)

for v in range(V_MAX + 1):
    axes[2].plot(time4psi_m, prob_t_m[:, v], linewidth=2, label=f"|v={v}⟩")

axes[2].plot(time4psi_m, total_prob_m, "k--", alpha=0.7, linewidth=1, label="Total")
axes[2].set_xlabel("Time [fs]")
axes[2].set_ylabel("Population (Morse)")
axes[2].legend()
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim(-0.05, 1.05)

plt.tight_layout()
plt.show()

# %% 励起解析
if False:
    print("\n=== 励起解析 ===")
    
    # 最終状態の分布（調和振動子）
    final_probs_h = prob_t_h[-1, :]
    print("\n調和振動子の最終状態分布:")
    print("状態\t\t確率")
    print("-" * 30)
    for v in range(V_MAX + 1):
        prob = final_probs_h[v]
        print(f"|v={v}⟩\t\t{prob:.6f}")

    # 最終状態の分布（モース振動子）
    final_probs_m = prob_t_m[-1, :]
    print("\nモース振動子の最終状態分布:")
    print("状態\t\t確率")
    print("-" * 30)
    for v in range(V_MAX + 1):
        prob = final_probs_m[v]
        print(f"|v={v}⟩\t\t{prob:.6f}")

    # 励起効率の比較
    print("\n=== 励起効率の比較 ===")
    excited_prob_h = 1.0 - final_probs_h[0]
    excited_prob_m = 1.0 - final_probs_m[0]
    print(f"調和振動子の励起効率: {excited_prob_h:.4f}")
    print(f"モース振動子の励起効率: {excited_prob_m:.4f}")

# %% エネルギー解析
if False:
    print("\n=== エネルギー解析 ===")

    # 期待値の計算（調和振動子）
    energy_expectation_h = np.zeros_like(time4psi_h)
    H0_matrix_h = H0_h.get_matrix()
    for i, t in enumerate(time4psi_h):
        psi_state = psi_t_h[i, :]
        energy_expectation_h[i] = np.real(np.conj(psi_state) @ H0_matrix_h @ psi_state)

    # 期待値の計算（モース振動子）
    energy_expectation_m = np.zeros_like(time4psi_m)
    H0_matrix_m = H0_m.get_matrix()
    for i, t in enumerate(time4psi_m):
        psi_state = psi_t_m[i, :]
        energy_expectation_m[i] = np.real(np.conj(psi_state) @ H0_matrix_m @ psi_state)

    # エネルギー変化
    energy_change_h = energy_expectation_h - energy_expectation_h[0]
    energy_change_m = energy_expectation_m - energy_expectation_m[0]
    print(f"エネルギー変化 (調和振動子): {energy_change_h[-1]:.6f} rad/fs")
    print(f"エネルギー変化 (モース振動子): {energy_change_m[-1]:.6f} rad/fs")

    # エネルギー変化プロット
    plt.figure(figsize=(10, 6))
    plt.plot(time4psi_h, energy_change_h, 'b-', linewidth=2, label='Harmonic')
    plt.plot(time4psi_m, energy_change_m, 'r-', linewidth=2, label='Morse')
    plt.xlabel('Time [fs]')
    plt.ylabel('Energy Change [rad/fs]')
    plt.title('Energy Expectation Time Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

print("\nシミュレーション完了")
