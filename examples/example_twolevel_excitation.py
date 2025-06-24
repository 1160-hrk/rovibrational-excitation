#!/usr/bin/env python
"""
二準位系の励起シミュレーション例
===============================

量子光学の基本的な二準位系における時間発展シミュレーション。
ラビ振動や異なる駆動周波数での応答を観察。

実行方法:
    python examples/example_twolevel_excitation.py
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpy as np
import matplotlib.pyplot as plt
from rovibrational_excitation.core.basis import TwoLevelBasis
from rovibrational_excitation.core.states import StateVector
from rovibrational_excitation.core.electric_field import ElectricField, gaussian
from rovibrational_excitation.core.propagator import schrodinger_propagation
from rovibrational_excitation.dipole.twolevel import TwoLevelDipoleMatrix

# ======================================================================
# パラメータ設定 (Parameters Configuration)
# ======================================================================

# システムパラメータ (System Parameters)
ENERGY_GAP = 2.0        # |1⟩ - |0⟩ のエネルギー差 [a.u.]
MU0 = 1.0              # 双極子行列要素の大きさ

# 時間グリッド設定 (Time Grid Settings)
TIME_START = 0.0       # 開始時間 [fs]
TIME_END = 400.0       # 終了時間 [fs]
DT_EFIELD = 0.01        # 電場サンプリング間隔 [fs]
SAMPLE_STRIDE = 2      # サンプリングストライド

# レーザーパルス設定 (Laser Pulse Settings)
PULSE_DURATION = 40.0  # パルス幅 [fs]

# デフォルトケースの設定 (Default Case Settings)
DEFAULT_DETUNING = 0.0    # デフォルトのデチューニング
DEFAULT_RABI_FREQ = 1e-20   # デフォルトのラビ周波数
DETUNED_CASE_DETUNING = 0.2  # デチューニングのあるケース

# スイープ設定 (Sweep Settings)
RABI_FREQ_MIN = 1e-21      # ラビ周波数スイープの最小値
RABI_FREQ_MAX = 1e-20       # ラビ周波数スイープの最大値
RABI_FREQ_POINTS = 10     # ラビ周波数スイープの点数

DETUNING_MIN = -0.5       # デチューニングスイープの最小値
DETUNING_MAX = 0.5        # デチューニングスイープの最大値
DETUNING_POINTS = 15      # デチューニングスイープの点数

# タイルプロット設定 (Tile Plot Settings)
TILE_PLOT_ROWS = 3        # タイルプロットの行数
TILE_PLOT_COLS = 4        # タイルプロットの列数

# ======================================================================


def run_twolevel_simulation(detuning=DEFAULT_DETUNING, rabi_freq=DEFAULT_RABI_FREQ, pulse_duration=PULSE_DURATION):
    """
    二準位系励起シミュレーションを実行
    
    Parameters
    ----------
    detuning : float
        デチューニング（レーザー周波数 - 遷移周波数）
    rabi_freq : float
        ラビ周波数（電場強度に比例）
    pulse_duration : float
        パルス幅 [fs]
    """
    print(f"=== Two-Level System Simulation (δ={detuning:.3f}, Ω_R={rabi_freq:.3f}) ===")
    
    # 基底とハミルトニアンの生成
    basis = TwoLevelBasis()
    H0 = basis.generate_H0(energy_gap=ENERGY_GAP)
    print(f"Energy levels: E0={np.diag(H0)[0]:.3f}, E1={np.diag(H0)[1]:.3f}")
    
    # 双極子行列の生成
    dipole_matrix = TwoLevelDipoleMatrix(basis, mu0=MU0)
    
    # 初期状態: 基底状態 |0⟩
    state = StateVector(basis)
    state.set_state(0, 1.0)  # 二準位系では整数で指定可能
    psi0 = state.data
    
    # 時間グリッドの設定
    time4Efield = np.arange(TIME_START, TIME_END + 2*DT_EFIELD, DT_EFIELD)
    
    # レーザーパルスの設定
    tc = (time4Efield[-1] + time4Efield[0]) / 2
    transition_freq = ENERGY_GAP  # 遷移周波数
    laser_freq = transition_freq + detuning  # レーザー周波数
    carrier_freq = laser_freq / (2*np.pi)
    
    # ラビ周波数から電場振幅を計算: Ω_R = μ·E/ℏ
    amplitude = rabi_freq / MU0  # 簡略化
    polarization = np.array([1, 0])  # 2要素ベクトル：Ex方向（σ_x結合）
    
    Efield = ElectricField(tlist=time4Efield)
    Efield.add_dispersed_Efield(
        envelope_func=gaussian,
        duration=pulse_duration,
        t_center=tc,
        carrier_freq=carrier_freq,
        amplitude=amplitude,
        polarization=polarization,
        const_polarisation=False,
    )
    
    # 時間発展計算
    print("Starting time evolution calculation...")
    time4psi, psi_t = schrodinger_propagation(
        H0=H0,
        Efield=Efield,
        dipole_matrix=dipole_matrix,
        psi0=psi0,
        axes="xy",  # Exとμ_x、Eyとμ_yをカップリング
        return_traj=True,
        return_time_psi=True,
        sample_stride=SAMPLE_STRIDE
    )
    print("Time evolution calculation completed.")
    
    return time4Efield, Efield, time4psi, psi_t, basis, detuning, rabi_freq


def plot_twolevel_results(time4Efield, Efield, time4psi, psi_t, basis, detuning, rabi_freq):
    """二準位系の結果をプロット"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # 電場の時間発展
    Efield_data = Efield.get_Efield()
    axes[0].plot(time4Efield, Efield_data[:, 0], 'r-', linewidth=1.5, label=r"$E_x(t)$")
    axes[0].set_ylabel("Electric Field [a.u.]")
    axes[0].set_title(f"Two-Level System Excitation (δ={detuning:.3f}, Ω_R={rabi_freq:.3f})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 各状態の占有確率
    prob_0 = np.abs(psi_t[:, 0])**2  # |0⟩ state
    prob_1 = np.abs(psi_t[:, 1])**2  # |1⟩ state
    
    axes[1].plot(time4psi, prob_0, 'b-', linewidth=2, label=r"|0⟩ (ground)")
    axes[1].plot(time4psi, prob_1, 'r-', linewidth=2, label=r"|1⟩ (excited)")
    axes[1].plot(time4psi, prob_0 + prob_1, 'k--', alpha=0.7, linewidth=1, label="Total")
    
    axes[1].set_ylabel("Population")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-0.05, 1.05)
    
    # ブロッホベクトルのz成分（反転）
    inversion = prob_1 - prob_0  # ⟨σ_z⟩ = P_1 - P_0
    axes[2].plot(time4psi, inversion, 'g-', linewidth=2, label=r"⟨σ_z⟩ = P₁ - P₀")
    axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[2].set_xlabel("Time [fs]")
    axes[2].set_ylabel("Population Inversion")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(-1.05, 1.05)
    
    plt.tight_layout()
    
    # ファイル保存
    filename = f"examples/results/twolevel_detuning_{detuning:.3f}_rabi_{rabi_freq:.3f}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {filename}")
    
    return fig


def analyze_rabi_oscillation(time4psi, psi_t, detuning, rabi_freq):
    """ラビ振動の解析"""
    print(f"\n=== Rabi Oscillation Analysis (δ={detuning:.3f}, Ω_R={rabi_freq:.3f}) ===")
    
    prob_1 = np.abs(psi_t[:, 1])**2
    max_excitation = np.max(prob_1)
    final_excitation = prob_1[-1]
    
    print(f"Maximum excitation probability: {max_excitation:.4f}")
    print(f"Final excitation probability: {final_excitation:.4f}")
    
    # 振動周期の推定（共鳴の場合）
    if abs(detuning) < 0.01:  # ほぼ共鳴
        effective_rabi = rabi_freq
        expected_period = 2*np.pi / effective_rabi
        print(f"Expected Rabi period: {expected_period:.2f} fs")
    else:
        effective_rabi = np.sqrt(rabi_freq**2 + detuning**2)
        expected_period = 2*np.pi / effective_rabi
        print(f"Effective Rabi frequency: {effective_rabi:.4f}")
        print(f"Expected oscillation period: {expected_period:.2f} fs")
    
    return max_excitation, final_excitation


def rabi_frequency_sweep():
    """ラビ周波数スイープによる励起効率の調査"""
    print("\n=== Rabi Frequency Sweep ===")
    
    rabi_frequencies = np.linspace(RABI_FREQ_MIN, RABI_FREQ_MAX, RABI_FREQ_POINTS)
    max_excitations = []
    final_excitations = []
    time_data = []
    psi_data = []
    
    for rabi_freq in rabi_frequencies:
        print(f"Ω_R = {rabi_freq:.3f}")
        _, _, time4psi, psi_t, _, _, _ = run_twolevel_simulation(
            detuning=0.0, 
            rabi_freq=rabi_freq, 
            pulse_duration=PULSE_DURATION
        )
        
        prob_1 = np.abs(psi_t[:, 1])**2
        max_excitations.append(np.max(prob_1))
        final_excitations.append(prob_1[-1])
        time_data.append(time4psi)
        psi_data.append(psi_t)
    
    # プロット
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(rabi_frequencies, max_excitations, 'bo-', linewidth=2, 
            markersize=6, label='Maximum Excitation')
    ax.plot(rabi_frequencies, final_excitations, 'rs-', linewidth=2, 
            markersize=6, label='Final Excitation')
    
    ax.set_xlabel('Rabi Frequency Ω_R')
    ax.set_ylabel('Excitation Probability')
    ax.set_title('Two-Level System: Rabi Frequency vs Excitation Efficiency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("examples/results/twolevel_rabi_sweep.png", dpi=300, bbox_inches='tight')
    print("Rabi frequency sweep results saved to examples/results/twolevel_rabi_sweep.png")
    
    return rabi_frequencies, max_excitations, final_excitations, time_data, psi_data


def detuning_sweep():
    """デチューニングスイープによる共鳴特性の調査"""
    print("\n=== Detuning Sweep ===")
    
    detunings = np.linspace(DETUNING_MIN, DETUNING_MAX, DETUNING_POINTS)
    max_excitations = []
    final_excitations = []
    time_data = []
    psi_data = []
    
    for detuning in detunings:
        print(f"δ = {detuning:.3f}")
        _, _, time4psi, psi_t, _, _, _ = run_twolevel_simulation(
            detuning=detuning, 
            rabi_freq=DEFAULT_RABI_FREQ, 
            pulse_duration=PULSE_DURATION
        )
        
        prob_1 = np.abs(psi_t[:, 1])**2
        max_excitations.append(np.max(prob_1))
        final_excitations.append(prob_1[-1])
        time_data.append(time4psi)
        psi_data.append(psi_t)
    
    # プロット
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(detunings, max_excitations, 'bo-', linewidth=2, 
            markersize=6, label='Maximum Excitation')
    ax.plot(detunings, final_excitations, 'rs-', linewidth=2, 
            markersize=6, label='Final Excitation')
    
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5, label='Resonance')
    ax.set_xlabel('Detuning δ')
    ax.set_ylabel('Excitation Probability')
    ax.set_title('Two-Level System: Detuning vs Excitation Efficiency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("examples/results/twolevel_detuning_sweep.png", dpi=300, bbox_inches='tight')
    print("Detuning sweep results saved to examples/results/twolevel_detuning_sweep.png")
    
    return detunings, max_excitations, final_excitations, time_data, psi_data


def plot_rabi_sweep_tile(rabi_freqs, time_data, psi_data):
    """ラビ周波数スイープの時間発展をタイルプロット"""
    print("\n=== Creating Rabi Frequency Sweep Tile Plot ===")
    
    n_cases = len(rabi_freqs)
    n_cols = min(TILE_PLOT_COLS, n_cases)
    n_rows = (n_cases + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows), 
                           sharex=True, sharey=True)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, (rabi_freq, time, psi) in enumerate(zip(rabi_freqs, time_data, psi_data)):
        row = i // n_cols
        col = i % n_cols
        
        prob_0 = np.abs(psi[:, 0])**2  # |0⟩ state
        prob_1 = np.abs(psi[:, 1])**2  # |1⟩ state
        
        axes[row, col].plot(time, prob_0, 'b-', linewidth=2, label='|0⟩')
        axes[row, col].plot(time, prob_1, 'r-', linewidth=2, label='|1⟩')
        axes[row, col].set_title(f'Ω_R = {rabi_freq:.3f}')
        axes[row, col].grid(True, alpha=0.3)
        axes[row, col].set_ylim(0, 1)
        
        if row == n_rows - 1:
            axes[row, col].set_xlabel('Time [fs]')
        if col == 0:
            axes[row, col].set_ylabel('Population')
        if i == 0:
            axes[row, col].legend()
    
    # 未使用のサブプロットを非表示
    for i in range(n_cases, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    fig.suptitle('Two-Level System: Time Evolution for Different Rabi Frequencies', fontsize=16)
    plt.tight_layout()
    plt.savefig("examples/results/twolevel_rabi_sweep_tile.png", dpi=300, bbox_inches='tight')
    print("Rabi frequency sweep tile plot saved to examples/results/twolevel_rabi_sweep_tile.png")
    
    return fig


def plot_detuning_sweep_tile(detunings, time_data, psi_data):
    """デチューニングスイープの時間発展をタイルプロット"""
    print("\n=== Creating Detuning Sweep Tile Plot ===")
    
    n_cases = len(detunings)
    n_cols = min(TILE_PLOT_COLS, n_cases)
    n_rows = (n_cases + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows), 
                           sharex=True, sharey=True)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, (detuning, time, psi) in enumerate(zip(detunings, time_data, psi_data)):
        row = i // n_cols
        col = i % n_cols
        
        prob_0 = np.abs(psi[:, 0])**2  # |0⟩ state
        prob_1 = np.abs(psi[:, 1])**2  # |1⟩ state
        
        axes[row, col].plot(time, prob_0, 'b-', linewidth=2, label='|0⟩')
        axes[row, col].plot(time, prob_1, 'r-', linewidth=2, label='|1⟩')
        axes[row, col].set_title(f'δ = {detuning:.2f}')
        axes[row, col].grid(True, alpha=0.3)
        axes[row, col].set_ylim(0, 1)
        
        if row == n_rows - 1:
            axes[row, col].set_xlabel('Time [fs]')
        if col == 0:
            axes[row, col].set_ylabel('Population')
        if i == 0:
            axes[row, col].legend()
    
    # 未使用のサブプロットを非表示
    for i in range(n_cases, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    fig.suptitle('Two-Level System: Time Evolution for Different Detunings', fontsize=16)
    plt.tight_layout()
    plt.savefig("examples/results/twolevel_detuning_sweep_tile.png", dpi=300, bbox_inches='tight')
    print("Detuning sweep tile plot saved to examples/results/twolevel_detuning_sweep_tile.png")
    
    return fig


def main():
    """メイン実行関数"""
    print("Two-Level System Excitation Simulation")
    print("=" * 50)
    
    # results/ディレクトリを作成
    os.makedirs("examples/results", exist_ok=True)
    
    # 基本的なラビ振動（共鳴）
    time4Efield, Efield, time4psi, psi_t, basis, detuning, rabi_freq = run_twolevel_simulation(
        detuning=DEFAULT_DETUNING, rabi_freq=DEFAULT_RABI_FREQ, pulse_duration=PULSE_DURATION
    )
    fig1 = plot_twolevel_results(time4Efield, Efield, time4psi, psi_t, basis, detuning, rabi_freq)
    analyze_rabi_oscillation(time4psi, psi_t, detuning, rabi_freq)
    
    # デチューニングのあるケース
    time4Efield2, Efield2, time4psi2, psi_t2, basis2, detuning2, rabi_freq2 = run_twolevel_simulation(
        detuning=DETUNED_CASE_DETUNING, rabi_freq=DEFAULT_RABI_FREQ, pulse_duration=PULSE_DURATION
    )
    fig2 = plot_twolevel_results(time4Efield2, Efield2, time4psi2, psi_t2, basis2, detuning2, rabi_freq2)
    analyze_rabi_oscillation(time4psi2, psi_t2, detuning2, rabi_freq2)
    
    # パラメータスイープ
    rabi_freqs, max_exc_rabi, final_exc_rabi, time_data_rabi, psi_data_rabi = rabi_frequency_sweep()
    detunings, max_exc_det, final_exc_det, time_data_det, psi_data_det = detuning_sweep()
    
    # タイルプロット作成
    fig_rabi_tile = plot_rabi_sweep_tile(rabi_freqs, time_data_rabi, psi_data_rabi)
    fig_det_tile = plot_detuning_sweep_tile(detunings, time_data_det, psi_data_det)
    
    plt.show()
    
    print("\nAll calculations completed!")
    print("Results:")
    print("- Confirmed resonant Rabi oscillation and detuning effects")
    print("- Investigated effects of Rabi frequency and detuning on excitation efficiency")
    print("- Created tile plots showing time evolution for all sweep conditions")
    print("- Use example_twolevel_2d_map.py for 2D amplitude-duration mapping")


if __name__ == "__main__":
    main() 