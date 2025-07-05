#!/usr/bin/env python
"""
二準位系の励起シミュレーション例
===============================

量子光学の基本的な二準位系における時間発展シミュレーション。
ラビ振動や異なる駆動周波数での応答を観察。

実行方法:
    python examples/example_twolevel_excitation.py
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from rovibrational_excitation.core.basis import TwoLevelBasis
from rovibrational_excitation.core.electric_field import ElectricField, gaussian_fwhm
from rovibrational_excitation.core.propagator import schrodinger_propagation
from rovibrational_excitation.core.basis import StateVector
from rovibrational_excitation.dipole.twolevel import TwoLevelDipoleMatrix
from rovibrational_excitation.core.units.converters import converter

# %% パラメータ設定
# システムパラメータ
ENERGY_GAP = 1  # |1⟩ - |0⟩ のエネルギー差 [rad/fs]
UNIT_ENERGY_GAP = "rad/fs"
MU0 = 1.0  # 双極子行列要素の大きさ [D]
UNIT_MU0 = "D"

# 時間グリッド設定
TIME_START = 0.0  # 開始時間 [fs]
TIME_END = 400.0  # 終了時間 [fs]
DT_EFIELD = 0.2  # 電場サンプリング間隔 [fs]
UNIT_TIME = "fs"
SAMPLE_STRIDE = 2  # サンプリングストライド

# レーザーパルス設定
PULSE_DURATION = 40.0  # パルス幅 [fs]
EFIELD_AMPLITUDE = 1e10  # デフォルトの電場振幅 [V/m]
UNIT_EFIELD_AMPLITUDE = "V/m"


os.makedirs("examples/results", exist_ok=True)

# %% 基底・ハミルトニアン・双極子行列の生成
basis = TwoLevelBasis(energy_gap=ENERGY_GAP, input_units=UNIT_ENERGY_GAP)
H0 = basis.generate_H0()
print(f"Energy levels: E0={H0.get_eigenvalues()[0]:.3f}, E1={H0.get_eigenvalues()[1]:.3f}")
dipole_matrix = TwoLevelDipoleMatrix(basis, mu0=MU0, units_input=UNIT_MU0)

# %% 初期状態の設定
state = StateVector(basis)
state.set_state(0, 1.0)
psi0 = state.data

# %% 時間グリッド・電場生成
# 共鳴ケース

time4Efield = np.arange(TIME_START, TIME_END + 2 * DT_EFIELD, DT_EFIELD)
tc = (time4Efield[-1] + time4Efield[0]) / 2

transition_freq = converter.convert_energy(ENERGY_GAP, UNIT_ENERGY_GAP, "PHz")
carrier_freq = transition_freq

Efield = ElectricField(
    tlist=time4Efield,
    field_units=UNIT_EFIELD_AMPLITUDE,
    time_units=UNIT_TIME,
)
Efield.add_dispersed_Efield(
    envelope_func=gaussian_fwhm,
    duration=PULSE_DURATION,
    t_center=tc,
    carrier_freq=carrier_freq,
    amplitude=EFIELD_AMPLITUDE,
    polarization=np.array([1, 0]),
    const_polarisation=True,
)

# %% 時間発展計算
print(f"=== Two-Level System Simulation (E={EFIELD_AMPLITUDE:.3e} V/m) ===")
print("Starting time evolution calculation...")
time4psi, psi_t = schrodinger_propagation(
    hamiltonian=H0,
    Efield=Efield,
    dipole_matrix=dipole_matrix,
    psi0=psi0,
    axes="xy",
    return_traj=True,
    return_time_psi=True,
    sample_stride=SAMPLE_STRIDE,
)
print("Time evolution calculation completed.")

# %% 結果プロット
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
Efield_data = Efield.get_Efield()
axes[0].plot(time4Efield, Efield_data[:, 0], "r-", linewidth=1.5, label=r"$E_x(t)$")
axes[0].set_ylabel("Electric Field [a.u.]")
axes[0].set_title(f"Two-Level System Excitation (E={EFIELD_AMPLITUDE:.3e} V/m)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

prob_0 = np.abs(psi_t[:, 0]) ** 2
prob_1 = np.abs(psi_t[:, 1]) ** 2
axes[1].plot(time4psi, prob_0, "b-", linewidth=2, label=r"|0⟩ (ground)")
axes[1].plot(time4psi, prob_1, "r-", linewidth=2, label=r"|1⟩ (excited)")
axes[1].plot(time4psi, prob_0 + prob_1, "k--", alpha=0.7, linewidth=1, label="Total")
axes[1].set_ylabel("Population")
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(-0.05, 1.05)

inversion = prob_1 - prob_0
axes[2].plot(time4psi, inversion, "g-", linewidth=2, label=r"⟨σ_z⟩ = P₁ - P₀")
axes[2].axhline(y=0, color="k", linestyle="--", alpha=0.5)
axes[2].set_xlabel("Time [fs]")
axes[2].set_ylabel("Population Inversion")
axes[2].legend()
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim(-1.05, 1.05)

plt.tight_layout()
plt.show()

# %% ラビ振動の解析
mu0_SI = MU0 * 3.336e-30  # Debye to C·m
hbar = 1.055e-34  # J·s
rabi_freq = mu0_SI * EFIELD_AMPLITUDE / hbar  # rad/s
prob_1 = np.abs(psi_t[:, 1]) ** 2
max_excitation = np.max(prob_1)
final_excitation = prob_1[-1]
print(f"Maximum excitation probability: {max_excitation:.4f}")
print(f"Final excitation probability: {final_excitation:.4f}")
