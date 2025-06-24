#!/usr/bin/env python
"""
二準位系の二次元パラメータマッピング
===================================

電場振幅とパルス時間幅の組み合わせで最終ポピュレーションの
二次元マップを生成・可視化。

実行方法:
    python examples/example_twolevel_2d_map.py
"""
import sys
import os
from typing import Dict, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpy as np
import matplotlib.pyplot as plt
from rovibrational_excitation.core.basis import TwoLevelBasis
from rovibrational_excitation.core.states import StateVector
from rovibrational_excitation.core.electric_field import ElectricField, gaussian
from rovibrational_excitation.core.propagator import schrodinger_propagation
from rovibrational_excitation.dipole.twolevel import TwoLevelDipoleMatrix


@dataclass
class SimulationConfig:
    """シミュレーション設定を管理するクラス"""
    # システムパラメータ
    energy_gap: float = 2.0
    mu0: float = 1.0e-30
    planck_constant: float = 6.62607015e-34 * 1e15  # [J・fs]
    
    # 時間グリッド設定
    time_start: float = 0.0
    time_end: float = 10000.0
    dt_efield: float = 0.2
    sample_stride: int = 50
    
    # デフォルトケース設定
    default_detuning: float = 0.0
    
    # 2次元スイープ設定
    duration_min: float = 1000.0
    duration_max: float = 3000.0
    duration_points: int = 200
    amplitude_min: float = 0
    amplitude_max: float = 5e11
    amplitude_points: int = 200
    
    @property
    def dirac_constant(self) -> float:
        """ディラック定数 [J・fs]"""
        return self.planck_constant / (2 * np.pi)


@dataclass
class PlotConfig:
    """プロット設定を管理するクラス"""
    figsize: Tuple[int, int] = (10, 8)
    dpi: int = 300
    condition_line_colors: Tuple[str, ...] = ('red', 'orange', 'yellow', 'cyan', 'magenta')
    condition_line_width: int = 2
    condition_line_style: str = '--'
    contour_levels: int = 21
    results_dir: str = "examples/results"


# グローバル設定インスタンス
config = SimulationConfig()
plot_config = PlotConfig()

# 実行時刻（ファイル名用）
execution_time = ""


def calculate_condition_lines(durations: np.ndarray, n_max: int = 4) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    MU0*amplitude * duration * sqrt(2*pi)/DIRAC_CONSTANT = (2n+1)*pi
    の条件線を計算
    
    Parameters:
    -----------
    durations : np.ndarray
        パルス時間幅の配列
    n_max : int
        計算する最大のn値
        
    Returns:
    --------
    condition_lines : Dict[int, Tuple[np.ndarray, np.ndarray]]
        各nに対する (durations, amplitudes) のペア
    """
    condition_lines = {}
    
    for n in range(n_max + 1):
        # amplitude = (2n+1)*pi * DIRAC_CONSTANT / (MU0 * duration * sqrt(2*pi))
        amplitudes = (2*n + 1) * np.pi * config.dirac_constant / (config.mu0 * durations * np.sqrt(2*np.pi))
        condition_lines[n] = (durations, amplitudes)
    
    return condition_lines


def setup_plot_axes(ax: plt.Axes, title: str) -> None:
    """プロット軸の共通設定を行う"""
    ax.set_xlabel('Pulse Duration [fs]')
    ax.set_ylabel('Electric Field Amplitude [a.u.]')
    ax.set_title(title)
    ax.set_ylim(config.amplitude_min, config.amplitude_max)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))


def add_condition_lines(ax: plt.Axes, durations: np.ndarray) -> None:
    """条件線をプロットに追加"""
    condition_lines = calculate_condition_lines(durations)
    
    for n, (dur, amp) in condition_lines.items():
        if n < len(plot_config.condition_line_colors):
            ax.plot(dur, amp, 
                   color=plot_config.condition_line_colors[n], 
                   linewidth=plot_config.condition_line_width,
                   linestyle=plot_config.condition_line_style,
                   label=f'n={n}: (2n+1)π condition')
    
    ax.legend(loc='upper right')


def save_plot(filename: str) -> None:
    """プロットを保存（実行日時をファイル名に含む）"""
    os.makedirs(plot_config.results_dir, exist_ok=True)
    
    # ファイル名と拡張子を分離
    name, ext = os.path.splitext(filename)
    
    # 実行日時を含むファイル名を作成
    timestamped_filename = f"{name}_{execution_time}{ext}"
    filepath = os.path.join(plot_config.results_dir, timestamped_filename)
    
    plt.savefig(filepath, dpi=plot_config.dpi, bbox_inches='tight')
    print(f"Plot saved to {filepath}")


def run_twolevel_simulation(detuning: Optional[float] = None, amplitude: float = 1e-20, pulse_duration: float = 40.0) -> Tuple[np.ndarray, Any, np.ndarray, np.ndarray]:
    """二準位系励起シミュレーションを実行"""
    if detuning is None:
        detuning = config.default_detuning
    
    # 基底とハミルトニアンの生成
    basis = TwoLevelBasis()
    H0 = basis.generate_H0(energy_gap=config.energy_gap)
    
    # 双極子行列の生成
    dipole_matrix = TwoLevelDipoleMatrix(basis, mu0=config.mu0)
    
    # 初期状態: 基底状態 |0⟩
    state = StateVector(basis)
    state.set_state(0, 1.0)
    psi0 = state.data
    
    # 時間グリッドの設定
    time4Efield = np.arange(config.time_start, config.time_end + 2*config.dt_efield, config.dt_efield)
    
    # レーザーパルスの設定
    tc = (time4Efield[-1] + time4Efield[0]) / 2
    transition_freq = config.energy_gap
    laser_freq = transition_freq + detuning
    carrier_freq = laser_freq / (2*np.pi)
    
    polarization = np.array([1, 0])
    
    Efield = ElectricField(tlist=time4Efield)
    Efield.add_dispersed_Efield(
        envelope_func=gaussian,
        duration=pulse_duration,
        t_center=tc,
        carrier_freq=carrier_freq,
        amplitude=amplitude,
        polarization=polarization,
        const_polarisation=True,
    )
    
    # 時間発展計算
    time4psi, psi_t = schrodinger_propagation(
        H0=H0,
        Efield=Efield,
        dipole_matrix=dipole_matrix,
        psi0=psi0,
        axes="xy",
        return_traj=True,
        return_time_psi=True,
        sample_stride=config.sample_stride
    )
    
    return time4Efield, Efield, time4psi, psi_t


def amplitude_duration_2d_sweep() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """電場振幅とパルス時間幅の2次元スイープ"""
    print("=== 2D Sweep: Electric Field Amplitude vs Pulse Duration ===")
    
    amplitudes = np.linspace(config.amplitude_min, config.amplitude_max, config.amplitude_points)
    durations = np.linspace(config.duration_min, config.duration_max, config.duration_points)
    
    # 結果を格納する2次元配列
    final_populations = np.zeros((len(durations), len(amplitudes)))
    
    total_cases = len(amplitudes) * len(durations)
    case_count = 0
    
    for i, duration in enumerate(durations):
        for j, amplitude in enumerate(amplitudes):
            case_count += 1
            if case_count % 500 == 0 or case_count <= 10:  # 進捗表示を調整
                print(f"Case {case_count}/{total_cases}: Amplitude={amplitude:.2e}, Duration={duration:.1f}fs")
            
            try:
                _, _, time4psi, psi_t = run_twolevel_simulation(
                    detuning=config.default_detuning,
                    amplitude=amplitude,
                    pulse_duration=duration
                )
                
                prob_1 = np.abs(psi_t[:, 1])**2  # 励起状態の確率
                final_populations[i, j] = prob_1[-1]  # 最終確率
                
            except Exception as e:
                print(f"Error in case {case_count}: {e}")
                final_populations[i, j] = 0.0
    
    return amplitudes, durations, final_populations


def plot_imshow_map(amplitudes: np.ndarray, durations: np.ndarray, final_populations: np.ndarray) -> plt.Figure:
    """imshowを使った2次元マップのプロット"""
    print("\n=== Creating imshow 2D Map ===")
    
    fig, ax = plt.subplots(figsize=plot_config.figsize)
    
    # imshowでマップを作成（x, y軸を反転）
    extent = [durations[0], durations[-1], amplitudes[0], amplitudes[-1]]
    im = ax.imshow(final_populations.T, aspect='auto', origin='lower', extent=extent, 
                   cmap='viridis', vmin=0, vmax=1, interpolation='bilinear')
    
    setup_plot_axes(ax, 'Final Population in Excited State |1⟩ (imshow)')
    add_condition_lines(ax, durations)
    
    # カラーバー
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Final Population')
    
    plt.tight_layout()
    save_plot("twolevel_2d_imshow.png")
    
    return fig


def plot_pcolormesh_map(amplitudes: np.ndarray, durations: np.ndarray, final_populations: np.ndarray) -> plt.Figure:
    """pcolormeshを使った2次元マップのプロット"""
    print("\n=== Creating pcolormesh 2D Map ===")
    
    fig, ax = plt.subplots(figsize=plot_config.figsize)
    
    # メッシュグリッド作成（x, y軸を反転）
    D, A = np.meshgrid(durations, amplitudes)
    
    # pcolormeshでマップを作成
    pcm = ax.pcolormesh(D, A, final_populations.T, cmap='plasma', vmin=0, vmax=1, shading='auto')
    
    setup_plot_axes(ax, 'Final Population in Excited State |1⟩ (pcolormesh)')
    add_condition_lines(ax, durations)
    
    # カラーバー
    cbar = plt.colorbar(pcm, ax=ax)
    cbar.set_label('Final Population')
    
    plt.tight_layout()
    save_plot("twolevel_2d_pcolormesh.png")
    
    return fig


def plot_contour_map(amplitudes: np.ndarray, durations: np.ndarray, final_populations: np.ndarray) -> plt.Figure:
    """contourを使った等高線プロット"""
    print("\n=== Creating Contour Map ===")
    
    fig, ax = plt.subplots(figsize=plot_config.figsize)
    
    # メッシュグリッド作成（x, y軸を反転）
    D, A = np.meshgrid(durations, amplitudes)
    
    # 等高線プロット
    levels = np.linspace(0, 1, plot_config.contour_levels)
    cs = ax.contour(D, A, final_populations.T, levels=levels, colors='black', linewidths=0.5)
    cs_filled = ax.contourf(D, A, final_populations.T, levels=levels, cmap='viridis', alpha=0.8)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.2f')
    
    setup_plot_axes(ax, 'Final Population in Excited State |1⟩ (Contour)')
    add_condition_lines(ax, durations)
    
    # カラーバー
    cbar = plt.colorbar(cs_filled, ax=ax)
    cbar.set_label('Final Population')
    
    plt.tight_layout()
    save_plot("twolevel_2d_contour.png")
    
    return fig


def main() -> None:
    """メイン実行関数"""
    global execution_time
    
    # 実行開始時刻を記録
    execution_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("Two-Level System 2D Parameter Mapping")
    print("=" * 50)
    print(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration:")
    print(f"  Duration range: {config.duration_min} - {config.duration_max} fs ({config.duration_points} points)")
    print(f"  Amplitude range: {config.amplitude_min:.2e} - {config.amplitude_max:.2e} a.u. ({config.amplitude_points} points)")
    print(f"  Total calculations: {config.duration_points * config.amplitude_points}")
    print("=" * 50)
    
    # 2次元スイープ実行
    amplitudes, durations, final_pops = amplitude_duration_2d_sweep()
    
    # 各種プロット作成
    fig_imshow = plot_imshow_map(amplitudes, durations, final_pops)
    fig_pcolormesh = plot_pcolormesh_map(amplitudes, durations, final_pops)
    fig_contour = plot_contour_map(amplitudes, durations, final_pops)
    
    plt.show()
    
    print("\nAll 2D mapping completed!")
    print("Generated plots:")
    print(f"- twolevel_2d_imshow_{execution_time}.png")
    print(f"- twolevel_2d_pcolormesh_{execution_time}.png") 
    print(f"- twolevel_2d_contour_{execution_time}.png")
    print("- All plots include (2n+1)π condition lines")
    print("- Condition lines show: MU0*amplitude * duration * sqrt(2*pi)/DIRAC_CONSTANT = (2n+1)*pi")


if __name__ == "__main__":
    main() 