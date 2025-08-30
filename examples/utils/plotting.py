from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from rovibrational_excitation.core.units.converters import converter
from .fft_utils import spectrogram_fast


def plot_all(
    *,
    basis,
    optimizer_like,
    efield,
    psi_traj: np.ndarray,
    field_data: np.ndarray,
    sample_stride: int = 1,
    omega_center_cm: Optional[float] = None,
    figures_dir: str | None = None,
    do_spectrum: bool = True,
    do_spectrogram: bool = True,
    filename_prefix: str = "local_optimization",
):
    t = optimizer_like.tlist if hasattr(optimizer_like, "tlist") else efield.get_time_SI()
    try:
        print(f"[debug.plot] t_len={len(t)}, psi_traj_shape={getattr(psi_traj, 'shape', None)}, field_data_shape={getattr(field_data, 'shape', None)}")
    except Exception:
        pass
    prob = np.abs(psi_traj) ** 2

    # 時系列の整合（local_optimization_rovibrational_2.py に準拠）
    # RK4の都合で状態サンプルは tlist の 2*sample_stride ごと
    stride = max(1, int(sample_stride))
    t_prob = t[::2 * stride]
    if prob.shape[0] != len(t_prob):
        # 長さが一致しない場合は短い方に合わせる（安全側）
        L = min(prob.shape[0], len(t_prob))
        t_prob = t_prob[:L]
        prob = prob[:L, :]

    # 1) Field and populations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].plot(t, field_data[:, 0], 'r-', label='Ex(t)')
    axes[0, 0].plot(t, field_data[:, 1], 'b-', label='Ey(t)')
    axes[0, 0].set_xlabel('Time [fs]')
    axes[0, 0].set_ylabel('Electric Field [V/m]')
    axes[0, 0].set_title('Designed Electric Field')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Target population (if optimizer carries target_idx)
    if hasattr(optimizer_like, "target_idx") and optimizer_like.target_idx is not None and optimizer_like.target_idx >= 0:
        idx_tar = int(optimizer_like.target_idx)
        idx_tar = idx_tar if 0 <= idx_tar < prob.shape[1] else None
    else:
        idx_tar = None
    if idx_tar is not None:
        axes[0, 1].plot(t_prob, prob[:, idx_tar], 'g-')
        axes[0, 1].set_xlabel('Time [fs]')
        axes[0, 1].set_ylabel('Population of target')
        axes[0, 1].set_title('Target Population vs Time')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 1.05)
    else:
        axes[0, 1].axis('off')

    # Populations overview
    for i, st in enumerate(getattr(basis, "basis", [])):
        axes[1, 0].plot(t_prob, prob[:, i], linewidth=1.0, alpha=0.9)
    axes[1, 0].set_xlabel('Time [fs]')
    axes[1, 0].set_ylabel('Population')
    axes[1, 0].set_title('State Population Evolution')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1.05)

    field_norm = np.linalg.norm(field_data)
    axes[1, 1].plot([0, len(t)-1], [field_norm, field_norm], 'k-')
    axes[1, 1].set_xlabel('Index')
    axes[1, 1].set_ylabel('||E||')
    axes[1, 1].set_title('Field Norm (overall)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if figures_dir:
        os.makedirs(figures_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(figures_dir, f"{filename_prefix}_results_{timestamp}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"結果を保存しました: {filepath}")
    plt.show()

    # 2) Spectrum around omega_center_cm
    if bool(do_spectrum):
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

            t_center = t[-1] / 2.0
            E_freq_comp = E_freq * (np.exp(1j * 2 * np.pi * freq_PHz * t_center)).reshape((len(freq_PHz), 1))

            intensity_x = np.abs(E_freq_comp[:, 0]) ** 2
            intensity_y = np.abs(E_freq_comp[:, 1]) ** 2

            if omega_center_cm is None:
                idx_max = int(np.argmax(intensity_x + intensity_y))
                center = float(freq_cm[idx_max])
            else:
                center = float(omega_center_cm)

            span = 500.0
            fmin = max(center - span, float(freq_cm[0]))
            fmax = min(center + span, float(freq_cm[-1]))

            mask = (freq_cm >= fmin) & (freq_cm <= fmax)
            freq_p = freq_cm[mask]

            fig2, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            ax0.plot(freq_p, intensity_x[mask], color='tab:blue', label='|Ex|²')
            ax0.set_ylabel('Intensity (a.u.)')
            ax0.set_title('Designed Field Spectrum (Ex)')
            ax0.set_xlim(fmin, fmax)
            ax0.legend(loc='upper right')

            ax1.plot(freq_p, intensity_y[mask], color='tab:green', label='|Ey|²')
            ax1.set_xlabel('Wavenumber (cm$^{-1}$)')
            ax1.set_ylabel('Intensity (a.u.)')
            ax1.set_title('Designed Field Spectrum (Ey)')
            ax1.set_xlim(fmin, fmax)
            ax1.legend(loc='upper right')

            plt.tight_layout()
            if figures_dir:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filepath = os.path.join(figures_dir, f"{filename_prefix}_spectrum_{timestamp}.png")
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"スペクトル図を保存しました: {filepath}")
            plt.show()
        except Exception as e:
            print(f"スペクトル可視化でエラー: {e}")

    # 3) Spectrogram (Ex)
    if bool(do_spectrogram):
        try:
            t_fs = efield.get_time_SI()
            Ex = efield.get_Efield()[:, 0]
            center = float(omega_center_cm) if omega_center_cm is not None else 0.0
            span = 500.0 if omega_center_cm is not None else 1000.0
            fmin = max(center - span, 0.0)
            fmax = center + span

            T_index = max(1, len(t_fs) // 20)
            res = spectrogram_fast(t_fs, Ex, T=T_index, unit_T='index', window_type='hamming', step=max(1, T_index // 8), N_pad=None)
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
            plt.tight_layout()
            if figures_dir:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filepath = os.path.join(figures_dir, f"{filename_prefix}_spectrogram_{timestamp}.png")
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"スペクトログラム図を保存しました: {filepath}")
            plt.show()
        except Exception as e:
            print(f"スペクトログラム可視化でエラー: {e}")


