#!/usr/bin/env python
"""
二準位系の二次元パラメータマッピング
===================================

電場振幅とパルス時間幅の組み合わせで最終ポピュレーションの
二次元マップを生成・可視化。

実行方法:
    python examples/example_twolevel_2d_map.py
"""

import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import matplotlib.pyplot as plt
import numpy as np

from rovibrational_excitation.core.basis import TwoLevelBasis
from rovibrational_excitation.core.electric_field import ElectricField, gaussian
from rovibrational_excitation.core.propagator import schrodinger_propagation
from rovibrational_excitation.core.states import StateVector
from rovibrational_excitation.dipole.twolevel import TwoLevelDipoleMatrix

# ======================================================================
# 主要設定 (Main Configuration) - 編集はここで行う
# ======================================================================

# プロット軸ラベル設定
XLABEL = "Pulse Duration [fs]"
YLABEL = "Electric Field Amplitude [a.u.]"

# 条件線設定
CONDITION_LINE_N_MAX = 10  # 条件線の最大n値 (n=0 から n_max まで描画)
CONDITION_LINE_COLORS = (
    "tab:red",
    "tab:orange",
    "tab:green",
    "tab:blue",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
) * 4  # 40色まで対応（10色を4回繰り返し）

# 2次元スイープ範囲設定
DURATION_MIN = 500.0  # パルス時間幅の最小値 [fs]
DURATION_MAX = 1000.0  # パルス時間幅の最大値 [fs]
DURATION_POINTS = 100  # パルス時間幅の点数

AMPLITUDE_MIN = 0  # 電場振幅の最小値 [a.u.]
AMPLITUDE_MAX = 5e12  # 電場振幅の最大値 [a.u.]
AMPLITUDE_POINTS = 100  # 電場振幅の点数

# 並列計算設定
MAX_WORKERS = mp.cpu_count()  # 使用するCPUコア数（自動検出）
CHUNK_SIZE = 1000  # 一度に処理するタスク数

# ======================================================================


@dataclass(slots=True)
class SimulationConfig:
    """シミュレーション設定を管理するクラス"""

    # システムパラメータ
    energy_gap: float = 0.5
    mu0: float = 1.0e-30
    planck_constant: float = 6.62607015e-34 * 1e15  # [J・fs]

    # 時間グリッド設定
    time_start: float = 0.0
    time_end: float = DURATION_MAX * 5  # 4: E=1/e^2
    dt_efield: float = 0.01
    sample_stride: int = 100

    # デフォルトケース設定
    default_detuning: float = 0.0

    # 2次元スイープ設定（上部の定数を参照）
    duration_min: float = DURATION_MIN
    duration_max: float = DURATION_MAX
    duration_points: int = DURATION_POINTS
    amplitude_min: float = AMPLITUDE_MIN
    amplitude_max: float = AMPLITUDE_MAX
    amplitude_points: int = AMPLITUDE_POINTS

    @property
    def dirac_constant(self) -> float:
        """ディラック定数 [J・fs]"""
        return self.planck_constant / (2 * np.pi)


@dataclass(slots=True)
class PlotConfig:
    """プロット設定を管理するクラス"""

    figsize: tuple[int, int] = (10, 8)
    dpi: int = 300
    condition_line_colors: tuple[str, ...] = CONDITION_LINE_COLORS
    condition_line_width: int = 2
    condition_line_style: str = "--"
    contour_levels: int = 100
    results_dir: str = "examples/results"


# グローバル設定インスタンス
config = SimulationConfig()
plot_config = PlotConfig()

# 実行時刻（ファイル名用）
execution_time = ""


def calculate_condition_lines(
    durations: np.ndarray, n_max: int = CONDITION_LINE_N_MAX
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
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
        amplitudes = (
            (2 * n + 1)
            * np.pi
            * config.dirac_constant
            / (config.mu0 * durations * np.sqrt(2 * np.pi))
        )
        condition_lines[n] = (durations, amplitudes)

    return condition_lines


def setup_plot_axes(ax: plt.Axes, title: str) -> None:
    """プロット軸の共通設定を行う"""
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    ax.set_title(title)
    ax.set_ylim(config.amplitude_min, config.amplitude_max)
    ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))


def add_condition_lines(ax: plt.Axes, durations: np.ndarray) -> None:
    """条件線をプロットに追加"""
    condition_lines = calculate_condition_lines(durations)

    for n, (dur, amp) in condition_lines.items():
        if n < len(plot_config.condition_line_colors):
            if n == 0:
                label = f"pulse area = {2 * n + 1}π"
            else:
                label = f"                  = {2 * n + 1}π"

            ax.plot(
                dur,
                amp,
                color=plot_config.condition_line_colors[n],
                linewidth=plot_config.condition_line_width,
                linestyle=plot_config.condition_line_style,
                label=label,
            )

    ax.legend(loc="upper right")


def save_plot(filename: str) -> None:
    """プロットを保存（実行日時をファイル名に含む）"""
    os.makedirs(plot_config.results_dir, exist_ok=True)

    # ファイル名と拡張子を分離
    name, ext = os.path.splitext(filename)

    # 実行日時を含むファイル名を作成
    timestamped_filename = f"{name}_{execution_time}{ext}"
    filepath = os.path.join(plot_config.results_dir, timestamped_filename)

    plt.savefig(filepath, dpi=plot_config.dpi, bbox_inches="tight")
    print(f"Plot saved to {filepath}")


def run_twolevel_simulation(
    detuning: float | None = None,
    amplitude: float = 1e-20,
    pulse_duration: float = 40.0,
) -> tuple[np.ndarray, Any, np.ndarray, np.ndarray]:
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
    time4Efield = np.arange(
        config.time_start, config.time_end + 2 * config.dt_efield, config.dt_efield
    )

    # レーザーパルスの設定
    tc = (time4Efield[-1] + time4Efield[0]) / 2
    transition_freq = config.energy_gap
    laser_freq = transition_freq + detuning
    carrier_freq = laser_freq / (2 * np.pi)

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
        sample_stride=config.sample_stride,
    )

    return time4Efield, Efield, time4psi, psi_t


def compute_single_case(
    params: tuple[float, float, float],
) -> tuple[float, float, float]:
    """単一の計算ケースを実行（並列化用）"""
    amplitude, duration, detuning = params

    try:
        _, _, time4psi, psi_t = run_twolevel_simulation(
            detuning=detuning, amplitude=amplitude, pulse_duration=duration
        )

        prob_1 = np.abs(psi_t[:, 1]) ** 2  # 励起状態の確率
        final_population = prob_1[-1]  # 最終確率

        return amplitude, duration, final_population

    except Exception as e:
        print(f"Error in case (A={amplitude:.2e}, D={duration:.1f}): {e}")
        return amplitude, duration, 0.0


def amplitude_duration_2d_sweep() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """電場振幅とパルス時間幅の2次元スイープ（並列化版）"""
    print("=== 2D Sweep: Electric Field Amplitude vs Pulse Duration (Parallel) ===")

    amplitudes = np.linspace(
        config.amplitude_min, config.amplitude_max, config.amplitude_points
    )
    durations = np.linspace(
        config.duration_min, config.duration_max, config.duration_points
    )

    # 結果を格納する2次元配列
    final_populations = np.zeros((len(durations), len(amplitudes)))

    # 計算パラメータリストを準備
    params_list = []
    for i, duration in enumerate(durations):
        for j, amplitude in enumerate(amplitudes):
            params_list.append((amplitude, duration, config.default_detuning))

    total_cases = len(params_list)
    print(f"Total cases: {total_cases}")
    print(f"Using {MAX_WORKERS} CPU cores")
    print(f"Chunk size: {CHUNK_SIZE}")
    print("Starting parallel computation...")

    # 並列計算実行
    completed_cases = 0
    results = {}

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # タスクを送信
        future_to_params = {
            executor.submit(compute_single_case, params): params
            for params in params_list
        }

        # 結果を収集
        for future in as_completed(future_to_params):
            amplitude, duration, final_pop = future.result()
            results[(amplitude, duration)] = final_pop

            completed_cases += 1
            if completed_cases % (CHUNK_SIZE * 10) == 0 or completed_cases <= 10:
                print(
                    f"Completed: {completed_cases}/{total_cases} ({completed_cases / total_cases * 100:.1f}%)"
                )

    # 結果を2次元配列に格納
    print("Organizing results...")
    for i, duration in enumerate(durations):
        for j, amplitude in enumerate(amplitudes):
            final_populations[i, j] = results.get((amplitude, duration), 0.0)

    print("Parallel computation completed!")
    return amplitudes, durations, final_populations


def plot_imshow_map(
    amplitudes: np.ndarray, durations: np.ndarray, final_populations: np.ndarray
) -> plt.Figure:
    """imshowを使った2次元マップのプロット"""
    print("\n=== Creating imshow 2D Map ===")

    fig, ax = plt.subplots(figsize=plot_config.figsize)

    # imshowでマップを作成（x, y軸を反転）
    extent = [durations[0], durations[-1], amplitudes[0], amplitudes[-1]]
    im = ax.imshow(
        final_populations.T,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=0,
        vmax=1,
        interpolation="bilinear",
    )

    setup_plot_axes(ax, "Final Population in Excited State |1⟩ (imshow)")
    add_condition_lines(ax, durations)

    # カラーバー
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Final Population")

    plt.tight_layout()
    save_plot("twolevel_2d_imshow.png")

    return fig


def plot_pcolormesh_map(
    amplitudes: np.ndarray, durations: np.ndarray, final_populations: np.ndarray
) -> plt.Figure:
    """pcolormeshを使った2次元マップのプロット"""
    print("\n=== Creating pcolormesh 2D Map ===")

    fig, ax = plt.subplots(figsize=plot_config.figsize)

    # メッシュグリッド作成（x, y軸を反転）
    D, A = np.meshgrid(durations, amplitudes)

    # pcolormeshでマップを作成
    pcm = ax.pcolormesh(
        D, A, final_populations.T, cmap="plasma", vmin=0, vmax=1, shading="auto"
    )

    setup_plot_axes(ax, "Final Population in Excited State |1⟩ (pcolormesh)")
    add_condition_lines(ax, durations)

    # カラーバー
    cbar = plt.colorbar(pcm, ax=ax)
    cbar.set_label("Final Population")

    plt.tight_layout()
    save_plot("twolevel_2d_pcolormesh.png")

    return fig


def plot_contour_map(
    amplitudes: np.ndarray, durations: np.ndarray, final_populations: np.ndarray
) -> plt.Figure:
    """contourを使った等高線プロット"""
    print("\n=== Creating Contour Map ===")

    fig, ax = plt.subplots(figsize=plot_config.figsize)

    # メッシュグリッド作成（x, y軸を反転）
    D, A = np.meshgrid(durations, amplitudes)

    # 等高線プロット（塗りつぶしのみ）
    levels = np.linspace(0, 1, plot_config.contour_levels)
    cs_filled = ax.contourf(D, A, final_populations.T, levels=levels, cmap="viridis")

    setup_plot_axes(ax, "Final Population in Excited State |1⟩ (Contour)")
    add_condition_lines(ax, durations)

    # カラーバー
    cbar = plt.colorbar(cs_filled, ax=ax)
    cbar.set_label("Final Population")

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
    print("Configuration:")
    print(
        f"  Duration range: {DURATION_MIN} - {DURATION_MAX} fs ({DURATION_POINTS} points)"
    )
    print(
        f"  Amplitude range: {AMPLITUDE_MIN:.2e} - {AMPLITUDE_MAX:.2e} a.u. ({AMPLITUDE_POINTS} points)"
    )
    print(f"  Total calculations: {DURATION_POINTS * AMPLITUDE_POINTS}")
    print(f"  Condition lines: n=0 to n={CONDITION_LINE_N_MAX}")
    print(f"  Parallel processing: {MAX_WORKERS} CPU cores, chunk size: {CHUNK_SIZE}")
    print("=" * 50)

    # 2次元スイープ実行
    amplitudes, durations, final_pops = amplitude_duration_2d_sweep()

    # 各種プロット作成
    # plot_imshow_map(amplitudes, durations, final_pops)
    plot_pcolormesh_map(amplitudes, durations, final_pops)
    plot_contour_map(amplitudes, durations, final_pops)

    plt.show()

    print("\nAll 2D mapping completed!")
    print("Generated plots:")
    print(f"- twolevel_2d_imshow_{execution_time}.png")
    print(f"- twolevel_2d_pcolormesh_{execution_time}.png")
    print(f"- twolevel_2d_contour_{execution_time}.png")
    print("- All plots include (2n+1)π condition lines")
    print(
        "- Condition lines show: MU0*amplitude * duration * sqrt(2*pi)/DIRAC_CONSTANT = (2n+1)*pi"
    )


if __name__ == "__main__":
    main()
