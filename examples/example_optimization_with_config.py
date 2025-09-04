#!/usr/bin/env python
"""
最適化設定ファイルを使用した振動回転励起最適制御
================================================

optimize_runner.py と YAML 設定ファイルを使用して、
振動回転励起の最適化を実行する例。

実行方法:
    python examples/example_optimization_with_config.py

設定ファイル:
    configs/config_v3j2m_g3.75e19.yaml

この例では:
- YAML設定ファイルから最適化パラメータを読み込み
- 複数の最適化アルゴリズムを試行
- 結果の可視化と保存
"""

import os
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from rovibrational_excitation.simulation.optimize_runner import run_from_config
from rovibrational_excitation.core.units.converters import converter
from utils.fft_utils import spectrogram_fast

"""メイン実行関数"""
print("=== 最適化設定ファイルを使用した振動回転励起最適制御 ===")

# CONFIG_NAME = "config_v3j2m_g3.75e19.yaml"
# CONFIG_NAME = "config_temp_local_linmol.yaml"
# CONFIG_NAME = "config_temp_local_viblad.yaml"
# CONFIG_NAME = "config_temp_krotov_viblad_v3.yaml"
# CONFIG_NAME = "config_temp_krotov_viblad_v4.yaml"
# CONFIG_NAME = "config_temp_krotov_viblad_v5.yaml"
# CONFIG_NAME = "config_temp_krotov_viblad_v10.yaml"
CONFIG_NAME = "config_temp_krotov_linmol2.yaml"

config_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "configs", CONFIG_NAME
))
if not os.path.exists(config_path):
    print(f"❌ 設定ファイルが見つかりません: {config_path}")


print(f"📁 設定ファイル: {config_path}")
    
# 最適化実行
res = run_from_config(
    config=config_path,
    out_dir=None,  # デフォルトの results/ ディレクトリを使用
    do_plot=True
)
cfg = res["cfg"]
basis = res["basis"]
H0 = res["H0"]
dipole = res["dipole"]
result = res["result"]
figures_dir = res["out_dir"]
filename_prefix = res["cfg"]["algorithm"]["selected"]

# %%
# 共通パラメータ初期化（v3j3 相当）
os.makedirs(figures_dir, exist_ok=True)
omega_center_cm = cfg.get("system", {}).get("params", {}).get("omega_cm", 2349.1)
fmin = float(max(0.0, omega_center_cm - 500.0))
fmax = float(omega_center_cm + 500.0)
Npad = 0
sample_stride = int(cfg.get("time", {}).get("sample_stride", 1))

# result 取り出し
efield = result.get("efield")
time_full = result.get("time")
psi_traj = result.get("psi_traj")
tlist = result.get("tlist")
if tlist is None or (hasattr(tlist, "size") and getattr(tlist, "size", 0) == 0):
    tlist = time_full
field_data = result.get("field_data")
target_idx = result.get("target_idx")

# %% Figure: Designed Electric Field (Ex, Ey)
try:
    t = tlist
    plt.figure(figsize=(12, 4))
    plt.plot(t, field_data[:, 0], 'r-', label='Ex(t)')
    plt.plot(t, field_data[:, 1], 'b-', label='Ey(t)')
    plt.xlabel('Time [fs]')
    plt.ylabel('Electric Field [V/m]')
    plt.title('Designed Electric Field (Local Optimization-like)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filepath = os.path.join(figures_dir, f"{filename_prefix}_field_{time.strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"結果を保存しました: {filepath}")
    plt.show()
except Exception as e_plot_field:
    print(f"Electric Field プロットでエラー: {e_plot_field}")

# %% Figure: Target Population vs Time
try:
    prob = np.abs(psi_traj) ** 2
    t = tlist
    if target_idx is not None and target_idx >= 0:
        plt.figure(figsize=(12, 4))
        plt.plot(t[::2*sample_stride], prob[:, int(target_idx)], 'g-')
        plt.xlabel('Time [fs]')
        plt.ylabel('Population of target')
        plt.title('Target Population vs Time')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)
        plt.tight_layout()
        filepath = os.path.join(figures_dir, f"{filename_prefix}_fidelity_{time.strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"結果を保存しました: {filepath}")
        plt.show()
except Exception as e_plot_fid:
    print(f"Fidelity プロットでエラー: {e_plot_fid}")

# %% Figure: State Population Evolution
try:
    t = tlist
    prob = np.abs(psi_traj) ** 2
    plt.figure(figsize=(12, 6))
    # ハイライト（初期・目標）
    def _norm_state(state: tuple[int, ...]) -> tuple[int, ...]:
        if basis.__class__.__name__ == "VibLadderBasis":
            return (int(state[0]),)
        if basis.__class__.__name__ == "LinMolBasis":
            if getattr(basis, "use_M", False):
                if len(state) == 2:
                    return (int(state[0]), int(state[1]), 0)
                return tuple(int(x) for x in state)
            if len(state) == 3:
                return (int(state[0]), int(state[1]))
            return tuple(int(x) for x in state)
        if basis.__class__.__name__ == "TwoLevelBasis":
            return (int(state[0]),)
    try:
        init_state = _norm_state(tuple(cfg["states"]["initial"]))
        targ_state = _norm_state(tuple(cfg["states"]["target"]))
    except Exception:
        init_state = None
        targ_state = None
    highlight_states = [s for s in (init_state, targ_state) if s is not None]
    highlight_set = {tuple(s) for s in highlight_states}
    for state in highlight_states:
        if state in getattr(basis, "index_map", {}):
            idx = basis.get_index(state)
            if len(state) == 1:
                label = f'|v={state[0]}⟩ (highlight)'
            elif len(state) == 2:
                label = f'|v={state[0]}, J={state[1]}⟩ (highlight)'
            elif len(state) == 3:
                label = f'|v={state[0]}, J={state[1]}, M={state[2]}⟩ (highlight)'
            else:
                label = f'|{state}⟩ (highlight)'
            plt.plot(t[::2*sample_stride], prob[:, idx], linewidth=2.5, label=label)
    for i, st in enumerate(basis.basis):
        st_t = tuple(int(x) for x in st)
        if st_t in highlight_set:
            continue
        if len(st_t) == 1:
            v = st_t[0]
            label = f'|v={v}⟩'
        elif len(st_t) == 2:
            v, J = st_t
            label = f'|v={v}, J={J}⟩'
        elif len(st_t) == 3:
            v, J, M = st_t
            label = f'|v={v}, J={J}, M={M}⟩'
        else:
            label = f'|{st_t}⟩'
        plt.plot(t[::2*sample_stride], prob[:, i], linewidth=1.0, alpha=0.9, label=label)
    plt.xlabel('Time [fs]')
    plt.ylabel('Population')
    plt.title('State Population Evolution')
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    filepath = os.path.join(figures_dir, f"{filename_prefix}_populations_{time.strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"結果を保存しました: {filepath}")
    plt.show()
except Exception as e_plot_pop:
    print(f"Population プロットでエラー: {e_plot_pop}")


# %% Figure: Spectrum (Ex/Ey intensity and phase) with transition lines
Npad = 0
freq_lim = [1500, 2600]
try:
    t_fs = efield.get_time_SI()
    dt_fs = float(t_fs[1] - t_fs[0])
    E_t = efield.get_Efield()
    N = len(t_fs)
    df_target_PHz = float(converter.convert_frequency(0.1, "cm^-1", "PHz"))
    Npad = int(np.ceil(1.0 / (dt_fs * df_target_PHz)))
    Npad = max(Npad, N)
    E_freq = np.fft.rfft(E_t, n=Npad, axis=0)
    freq_PHz = np.fft.rfftfreq(Npad, d=dt_fs)
    freq_cm = np.asarray(converter.convert_frequency(freq_PHz, "PHz", "cm^-1"), dtype=float)
    t_center = tlist[-1] / 2.0
    E_freq_comp = E_freq * (np.exp(1j * 2 * np.pi * freq_PHz * t_center)).reshape((len(freq_PHz), 1))
    intensity_x = np.abs(E_freq_comp[:, 0]) ** 2
    intensity_y = np.abs(E_freq_comp[:, 1]) ** 2
    intensity_sum = intensity_x + intensity_y
    peak_idx = int(np.argmax(intensity_x))
    f0 = float(freq_cm[peak_idx])
    I0 = float(intensity_sum[peak_idx])
    I_th = I0 / np.exp(4.0)
    def _interp_cross(fa, Ia, fb, Ib, Ith):
        if Ib == Ia:
            return fb
        return float(fa + (Ith - Ia) * (fb - fa) / (Ib - Ia))
    f_left = float(freq_cm[0])
    found_left = False
    for i in range(peak_idx - 1, 0, -1):
        if intensity_sum[i] >= I_th and intensity_sum[i - 1] < I_th:
            f_left = _interp_cross(freq_cm[i], intensity_sum[i], freq_cm[i - 1], intensity_sum[i - 1], I_th)
            found_left = True
            break
    if not found_left:
        f_left = float(freq_cm[0])
    f_right = float(freq_cm[-1])
    found_right = False
    for i in range(peak_idx + 1, len(freq_cm)):
        if intensity_sum[i - 1] >= I_th and intensity_sum[i] < I_th:
            f_right = _interp_cross(freq_cm[i - 1], intensity_sum[i - 1], freq_cm[i], intensity_sum[i], I_th)
            found_right = True
            break
    if not found_right:
        f_right = float(freq_cm[-1])
    fmin = float(max(0.0, omega_center_cm - 500.0))
    fmax = float(omega_center_cm + 500.0)
    try:
        eigenvalues = H0.get_eigenvalues()  # rad/fs
        states = basis.basis
        energy_by_vj_map: dict[tuple[int, int], float] = {}
        for idx, st in enumerate(states):
            v = int(st[0]); J = int(st[1])
            M = int(st[2]) if len(st) == 3 else 0
            key = (v, J)
            if key not in energy_by_vj_map or M == 0:
                energy_by_vj_map[key] = float(eigenvalues[idx])
        trans_wn: list[float] = []
        for (v, J), E0 in energy_by_vj_map.items():
            v_up = v + 1
            for dJ in (+1, -1):
                key = (v_up, J + dJ)
                if key in energy_by_vj_map:
                    d_omega = energy_by_vj_map[key] - E0
                    wn = float(converter.convert_frequency(d_omega, "rad/fs", "cm^-1"))
                    if np.isfinite(wn) and wn > 0:
                        trans_wn.append(wn)
        if len(trans_wn) >= 1:
            wn_min = float(np.min(trans_wn))
            wn_max = float(np.max(trans_wn))
            center = 0.5 * (wn_min + wn_max)
            span = max(wn_max - wn_min, 1e-6)
            factor = 2
            half = 0.5 * span * factor
            fmin = max(center - half, float(freq_cm[0]))
            fmax = min(center + half, float(freq_cm[-1]))
    except Exception:
        pass
    if freq_lim is not None:
        fmin = freq_lim[0]
        fmax = freq_lim[1]
    phase_x_raw = -np.unwrap(np.angle(E_freq_comp[:, 0]))
    phase_y_raw = -np.unwrap(np.angle(E_freq_comp[:, 1]))
    dphidk_x = np.gradient(phase_x_raw, freq_cm)
    dphidk_y = np.gradient(phase_y_raw, freq_cm)
    slope_x = float(dphidk_x[peak_idx])
    slope_y = float(dphidk_y[peak_idx])
    phase_x = phase_x_raw - (slope_x * (freq_cm - f0) + phase_x_raw[peak_idx])
    phase_y = phase_y_raw - (slope_y * (freq_cm - f0) + phase_y_raw[peak_idx])
    mask = (freq_cm >= fmin) & (freq_cm <= fmax)
    freq_p = freq_cm[mask]
    intensity_x_p = intensity_x[mask]
    intensity_y_p = intensity_y[mask]
    phase_x_p = phase_x[mask]
    phase_y_p = phase_y[mask]
    fig2, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax0.plot(freq_p, intensity_x_p, color='tab:blue', label='|Ex|²')
    ax0.set_ylabel('Intensity (a.u.)')
    ax0_t = ax0.twinx()
    ax0_t.plot(freq_p, phase_x_p, color='tab:red', alpha=0.7, label='Phase Ex')
    ax0_t.set_ylabel('Phase (rad)')
    ax0.set_title('Designed Field Spectrum (Ex)')
    # ax0.set_xlim(fmin, fmax)
    lines0, labels0 = ax0.get_legend_handles_labels()
    lines0_t, labels0_t = ax0_t.get_legend_handles_labels()
    ax0.legend(lines0 + lines0_t, labels0 + labels0_t, loc='upper right')
    ax1.plot(freq_p, intensity_y_p, color='tab:green', label='|Ey|²')
    ax1.set_xlabel('Wavenumber (cm$^{-1}$)')
    ax1.set_ylabel('Intensity (a.u.)')
    ax1_t = ax1.twinx()
    ax1_t.plot(freq_p, phase_y_p, color='tab:orange', alpha=0.7, label='Phase Ey')
    ax1_t.set_ylabel('Phase (rad)')
    ax1.set_title('Designed Field Spectrum (Ey)')
    # ax1.set_xlim(fmin, fmax)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines1_t, labels1_t = ax1_t.get_legend_handles_labels()
    ax1.legend(lines1 + lines1_t, labels1 + labels1_t, loc='upper right')
    try:
        eigenvalues = H0.get_eigenvalues()
        states = basis.basis
        lines: list[tuple[float, str]] = []
        if basis.__class__.__name__ == "VibLadderBasis":
            energy_by_v_lines: dict[int, float] = {}
            for idx, st in enumerate(states):
                key = int(st[0])
                if key not in energy_by_v_lines:
                    energy_by_v_lines[key] = float(eigenvalues[idx])
            for v, E0 in energy_by_v_lines.items():
                if v == len(basis.V_array) - 1:
                    continue
                v_up = v + 1
                d_omega = energy_by_v_lines[v_up] - E0
                wn = float(converter.convert_frequency(d_omega, "rad/fs", "cm^-1"))
                if np.isfinite(wn) and wn > 0:
                    lines.append((wn, f"{v}→{v + 1}"))
        elif basis.__class__.__name__ == "LinMolBasis":
            energy_by_vj_lines: dict[tuple[int, int], float] = {}
            for idx, st in enumerate(states):
                v = int(st[0]); J = int(st[1])
                M = int(st[2]) if len(st) == 3 else 0
                key = (v, J)
                if key not in energy_by_vj_lines or M == 0:
                    energy_by_vj_lines[key] = float(eigenvalues[idx])
            for (v, J), E0 in energy_by_vj_lines.items():
                v_up = v + 1
                key_R = (v_up, J + 1)
                if key_R in energy_by_vj_lines:
                    d_omega = energy_by_vj_lines[key_R] - E0
                    wn = float(converter.convert_frequency(d_omega, "rad/fs", "cm^-1"))
                    if np.isfinite(wn) and wn > 0:
                        lines.append((wn, rf"$R({J})_{{{v}}}$"))
                key_P = (v_up, J - 1)
                if key_P in energy_by_vj_lines:
                    d_omega = energy_by_vj_lines[key_P] - E0
                    wn = float(converter.convert_frequency(d_omega, "rad/fs", "cm^-1"))
                    if np.isfinite(wn) and wn > 0:
                        lines.append((wn, rf"$P({J})_{{{v}}}$"))
        y0 = float(np.max(intensity_x_p)) if intensity_x_p.size else 1.0
        y1 = float(np.max(intensity_y_p)) if intensity_y_p.size else 1.0
        for wn, lbl in lines:
            if fmin <= wn <= fmax:
                ax0.axvline(wn, color='gray', alpha=0.35, linewidth=0.8)
                ax0.text(wn, y0, lbl, rotation=-90, va='top', ha='center', fontsize=8, color='gray')
                ax1.axvline(wn, color='gray', alpha=0.35, linewidth=0.8)
                ax1.text(wn, y1, lbl, rotation=-90, va='top', ha='center', fontsize=8, color='gray')
    except Exception as e_lines:
        print(f"遷移線オーバーレイでエラー: {e_lines}")
    plt.tight_layout()
    filepath2 = os.path.join(figures_dir, f"{filename_prefix}_spectrum_{time.strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(filepath2, dpi=300, bbox_inches='tight')
    print(f"スペクトル図を保存しました: {filepath2}")
    plt.show()
except Exception as e_spec:
    print(f"スペクトル可視化でエラー: {e_spec}")
    if 'fmin' not in locals():
        fmin = float(max(0.0, omega_center_cm - 500.0))
    if 'fmax' not in locals():
        fmax = float(omega_center_cm + 500.0)
    if 'Npad' not in locals():
        Npad = 0

# %% Figure: Spectrogram (Ex) with peak ridge and transition hlines
try:
    t_fs = efield.get_time_SI()
    Ex = efield.get_Efield()[:, 0]
    if 'fmin' not in locals() or 'fmax' not in locals():
        center = float(omega_center_cm)
        span = 500.0
        fmin = center - span
        fmax = center + span
    T_index = len(t_fs) // 20
    res = spectrogram_fast(t_fs, Ex, T=T_index, unit_T='index', window_type='hamming', step=max(1, T_index // 8), N_pad=Npad)
    if len(res) == 4:
        x_spec, freq_1fs, spec, _max_idx = res
    else:
        x_spec, freq_1fs, spec = res
    freq_cm_full = np.asarray(converter.convert_frequency(freq_1fs, "PHz", "cm^-1"), dtype=float)
    mask_rng = (freq_cm_full >= fmin) & (freq_cm_full <= fmax)
    freq_cm_plot = freq_cm_full[mask_rng]
    spec_plot = spec[mask_rng, :]
    X, Y = np.meshgrid(x_spec, freq_cm_plot)
    fig3, ax3 = plt.subplots(1, 1, figsize=(12, 6))
    cf = ax3.pcolormesh(X, Y, spec_plot, shading='auto', cmap='viridis')
    ax3.set_xlabel('Time [fs]')
    ax3.set_ylabel('Wavenumber (cm$^{-1}$)')
    ax3.set_title('Spectrogram (Ex)')
    ax3.set_ylim(fmin, fmax)
    fig3.colorbar(cf, ax=ax3, label='|FFT|')
    try:
        if spec_plot.ndim == 2 and spec_plot.size > 0:
            peak_indices = np.argmax(spec_plot, axis=0)
            ridge_cm = freq_cm_plot[peak_indices]
            ax3.plot(x_spec, ridge_cm, color='red', linewidth=1.2, label='Peak frequency', zorder=4)
    except Exception as e_ridge:
        print(f"ピークリッジ描画でエラー: {e_ridge}")
    try:
        eigenvalues = H0.get_eigenvalues()  # rad/fs
        states = basis.basis
        lines_wn_spec: list[float] = []
        if basis.__class__.__name__ == "VibLadderBasis":
            energy_by_v_lines_spec: dict[int, float] = {}
            for idx, st in enumerate(states):
                key = int(st[0])
                if key not in energy_by_v_lines_spec:
                    energy_by_v_lines_spec[key] = float(eigenvalues[idx])
            for v, E0 in energy_by_v_lines_spec.items():
                if v == len(basis.V_array) - 1:
                    continue
                v_up = v + 1
                d_omega = energy_by_v_lines_spec[v_up] - E0
                wn = float(converter.convert_frequency(d_omega, "rad/fs", "cm^-1"))
                if np.isfinite(wn) and wn > 0:
                    lines_wn_spec.append(wn)
        elif basis.__class__.__name__ == "LinMolBasis":
            energy_by_vj_lines_spec: dict[tuple[int, int], float] = {}
            for idx, st in enumerate(states):
                v = int(st[0]); J = int(st[1])
                M = int(st[2]) if len(st) == 3 else 0
                key = (v, J)
                if key not in energy_by_vj_lines_spec or M == 0:
                    energy_by_vj_lines_spec[key] = float(eigenvalues[idx])
            for (v, J), E0 in energy_by_vj_lines_spec.items():
                v_up = v + 1
                for dJ in (+1, -1):
                    key = (v_up, J + dJ)
                    if key in energy_by_vj_lines_spec:
                        d_omega = energy_by_vj_lines_spec[key] - E0
                        wn = float(converter.convert_frequency(d_omega, "rad/fs", "cm^-1"))
                        if np.isfinite(wn) and wn > 0:
                            lines_wn_spec.append(wn)
        for wn in lines_wn_spec:
            if fmin <= wn <= fmax:
                ax3.hlines(wn, x_spec[0], x_spec[-1], colors='white', linestyles='-', linewidth=0.6, alpha=0.7, zorder=3)
        handles, labels = ax3.get_legend_handles_labels()
        if handles:
            ax3.legend(loc='upper right')
    except Exception as e_hlines:
        print(f"スペクトログラムの遷移線でエラー: {e_hlines}")
    plt.tight_layout()
    filepath3 = os.path.join(figures_dir, f"{filename_prefix}_spectrogram_{time.strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(filepath3, dpi=300, bbox_inches='tight')
    print(f"スペクトログラム図を保存しました: {filepath3}")
    plt.show()
except Exception as e_spect:
    print(f"スペクトログラム可視化でエラー: {e_spect}")





