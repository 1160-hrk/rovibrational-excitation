from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

from rovibrational_excitation.core.units.converters import converter
from utils.fft_utils import spectrogram_fast  # examples util – assumed present in project


def plot_all(
    *,
    basis,
    optimizer_like: Any,
    efield,
    psi_traj: np.ndarray,
    field_data: np.ndarray,
    sample_stride: int,
    omega_center_cm: float | None,
    figures_dir: str,
    filename_prefix: str = "opt",
    do_spectrum: bool = True,
    do_spectrogram: bool = True,
):
    os.makedirs(figures_dir, exist_ok=True)
    t = getattr(optimizer_like, "tlist", None)
    if t is None:
        t = efield.get_time_SI()

    # 1) Field
    plt.figure(figsize=(12, 4))
    plt.plot(t, field_data[:, 0], 'r-', label='Ex(t)')
    plt.plot(t, field_data[:, 1], 'b-', label='Ey(t)')
    plt.xlabel('Time [fs]')
    plt.ylabel('Electric Field [V/m]')
    plt.title('Designed Electric Field')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    fp = os.path.join(figures_dir, f"{filename_prefix}_field_{time.strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(fp, dpi=300, bbox_inches='tight'); plt.show()

    # 2) Target population
    prob = np.abs(psi_traj) ** 2
    idx_tar = getattr(optimizer_like, "target_idx", -1)
    if idx_tar is not None and int(idx_tar) >= 0:
        plt.figure(figsize=(12, 4))
        plt.plot(t[::2*max(1, sample_stride)], prob[:, int(idx_tar)], 'g-')
        plt.xlabel('Time [fs]'); plt.ylabel('Population of target')
        plt.title('Target Population vs Time')
        plt.grid(True, alpha=0.3); plt.ylim(0, 1.05); plt.tight_layout()
        fp = os.path.join(figures_dir, f"{filename_prefix}_fidelity_{time.strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(fp, dpi=300, bbox_inches='tight'); plt.show()

    # 3) Spectrum (optional)
    if do_spectrum:
        try:
            t_fs = efield.get_time_SI(); dt_fs = float(t_fs[1] - t_fs[0])
            E_t = efield.get_Efield(); N = len(t_fs)
            df_target_PHz = float(converter.convert_frequency(0.1, "cm^-1", "PHz"))
            Npad = max(int(np.ceil(1.0 / (dt_fs * df_target_PHz))), N)
            E_freq = np.fft.rfft(E_t, n=Npad, axis=0)
            freq_PHz = np.fft.rfftfreq(Npad, d=dt_fs)
            freq_cm = np.asarray(converter.convert_frequency(freq_PHz, "PHz", "cm^-1"), dtype=float)
            t_center = t[-1] / 2.0
            E_freq_comp = E_freq * (np.exp(1j * 2 * np.pi * freq_PHz * t_center)).reshape((len(freq_PHz), 1))
            intensity_x = np.abs(E_freq_comp[:, 0]) ** 2
            intensity_y = np.abs(E_freq_comp[:, 1]) ** 2
            if omega_center_cm is None:
                omega_center_cm = float(freq_cm[np.argmax(intensity_x + intensity_y)])
            span = 500.0
            fmin = float(max(0.0, omega_center_cm - span)); fmax = float(omega_center_cm + span)
            mask = (freq_cm >= fmin) & (freq_cm <= fmax)
            freq_p = freq_cm[mask]
            int_x_p = intensity_x[mask]; int_y_p = intensity_y[mask]
            fig2, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            ax0.plot(freq_p, int_x_p, color='tab:blue', label='|Ex|²')
            ax0.set_ylabel('Intensity (a.u.)'); ax0.set_title('Field Spectrum (Ex)')
            ax0.set_xlim(fmin, fmax)
            ax1.plot(freq_p, int_y_p, color='tab:green', label='|Ey|²')
            ax1.set_xlabel('Wavenumber (cm$^{-1}$)'); ax1.set_ylabel('Intensity (a.u.)')
            ax1.set_title('Field Spectrum (Ey)'); ax1.set_xlim(fmin, fmax)
            plt.tight_layout()
            fp = os.path.join(figures_dir, f"{filename_prefix}_spectrum_{time.strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(fp, dpi=300, bbox_inches='tight'); plt.show()
        except Exception as e:
            print(f"Spectrum plotting failed: {e}")

    # 4) Spectrogram (optional)
    if do_spectrogram:
        try:
            t_fs = efield.get_time_SI(); Ex = efield.get_Efield()[:, 0]
            center = float(omega_center_cm) if omega_center_cm is not None else 2349.1
            span = 500.0; fmin = center - span; fmax = center + span
            T_index = len(t_fs) // 20
            res = spectrogram_fast(t_fs, Ex, T=T_index, unit_T='index', window_type='hamming', step=max(1, T_index // 8), N_pad=0)
            if len(res) == 4:
                x_spec, freq_1fs, spec, _ = res
            else:
                x_spec, freq_1fs, spec = res
            freq_cm_full = np.asarray(converter.convert_frequency(freq_1fs, "PHz", "cm^-1"), dtype=float)
            mask_rng = (freq_cm_full >= fmin) & (freq_cm_full <= fmax)
            freq_cm_plot = freq_cm_full[mask_rng]
            spec_plot = spec[mask_rng, :]
            X, Y = np.meshgrid(x_spec, freq_cm_plot)
            fig3, ax3 = plt.subplots(1, 1, figsize=(12, 6))
            cf = ax3.pcolormesh(X, Y, spec_plot, shading='auto', cmap='viridis')
            ax3.set_xlabel('Time [fs]'); ax3.set_ylabel('Wavenumber (cm$^{-1}$)')
            ax3.set_title('Spectrogram (Ex)'); ax3.set_ylim(fmin, fmax)
            fig3.colorbar(cf, ax=ax3, label='|FFT|'); plt.tight_layout()
            fp = os.path.join(figures_dir, f"{filename_prefix}_spectrogram_{time.strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(fp, dpi=300, bbox_inches='tight'); plt.show()
        except Exception as e:
            print(f"Spectrogram plotting failed: {e}")


