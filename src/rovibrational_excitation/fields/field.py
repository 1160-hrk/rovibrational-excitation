"""
電場コアクラス
==============

電場波形を表現するメインクラス。
単位変換機能を持ち、時間を fs、電場を V/m で内部保持する。
無次元化機能は dynamics.scaling.converter に統一。
"""

import inspect
from typing import TYPE_CHECKING, Union

import numpy as np
from numpy import pi
from scipy.fft import irfft, rfft, rfftfreq

if TYPE_CHECKING:
    from ..core.time import TimeGrid

from rovibrational_excitation.core.units.converters import converter

from .modulation import (
    _remove_linear_phase,
    _select_window,
    apply_dispersion,
    apply_sinusoidal_mod,
    get_mod_spectrum_from_bin_setting,
)


class ElectricField:
    """
    電場波形を表現するクラス（偏光、包絡線、GDD/TOD付き）

    時間を fs、電場を V/m の固定単位系で内部保持し、単位変換機能を提供。
    無次元化機能は dynamics.scaling.converter に委譲。
    """

    def __init__(self, tlist: np.ndarray, *, time_units: str):
        """
        Parameters
        ----------
        tlist : np.ndarray
            時間軸（指定単位）
        time_units : str
            時間の単位 ("fs", "ps", "ns", "s")

        Notes
        -----
        The initially zero field and every stored field sample use canonical
        V/m. Field units belong to field addition, setting, or output methods;
        there is no field-valued constructor argument to label.
        """
        # 時間配列を内部単位（fs）に変換
        self.tlist = self._convert_time_to_fs(np.asarray(tlist), time_units)
        if self.tlist.ndim != 1 or self.tlist.size < 2:
            raise ValueError(
                "tlist must be a one-dimensional array with at least 2 points"
            )
        if not np.all(np.isfinite(self.tlist)):
            raise ValueError("tlist must contain only finite values")
        intervals = np.diff(self.tlist)
        if np.any(intervals <= 0):
            raise ValueError("tlist must be strictly increasing")
        if not np.allclose(intervals, intervals[0], rtol=1e-10, atol=1e-12):
            raise ValueError("tlist must be uniformly spaced")
        self.dt = self.tlist[1] - self.tlist[0]  # fs単位
        self.dt_state = self.dt * 2
        self.steps_state = len(self.tlist) // 2

        # 電場配列を初期化（内部はV/m単位で保持）
        self.Efield = np.zeros((len(self.tlist), 2))  # V/m単位
        self.add_history = []
        self._constant_pol: Union[np.ndarray, None, bool] = None
        self._scalar_field: np.ndarray | None = None

    @classmethod
    def from_time_grid(
        cls,
        time_grid: "TimeGrid",
    ) -> "ElectricField":
        """Construct a canonical V/m field from the canonical fs grid."""
        return cls(
            time_grid.field_times_fs,
            time_units="fs",
        )

    # ------------------------------------------------------------------
    # Unit conversion helpers
    # ------------------------------------------------------------------
    def _convert_time_to_fs(
        self, time_array: np.ndarray, from_units: str
    ) -> np.ndarray:
        """Convert a time array to femtoseconds via UnitConverter."""
        return np.asarray(converter.convert_time(time_array, from_units, "fs"))

    def _convert_time_from_fs(
        self, time_array_fs: np.ndarray, to_units: str
    ) -> np.ndarray:
        """Convert a time array from femtoseconds to the requested units."""
        return np.asarray(converter.convert_time(time_array_fs, "fs", to_units))

    def _convert_field_to_SI(
        self, field_array: np.ndarray, from_units: str
    ) -> np.ndarray:
        """Convert electric-field array to SI (V/m) units via UnitConverter."""
        return np.asarray(
            converter.convert_electric_field(field_array, from_units, "V/m")
        )

    def _convert_field_from_SI(
        self, field_array_SI: np.ndarray, to_units: str
    ) -> np.ndarray:
        """Convert electric-field array from SI (V/m) to requested units via UnitConverter."""
        return np.asarray(
            converter.convert_electric_field(field_array_SI, "V/m", to_units)
        )

    def init_Efield(self) -> "ElectricField":
        """電場をゼロに初期化"""
        self.Efield = np.zeros((len(self.tlist), 2))
        return self

    def get_Efield(self) -> np.ndarray:
        """電場を取得（V/m単位）"""
        return np.asarray(self.Efield)

    def get_Efield_SI(self) -> np.ndarray:
        """SI単位系での電場を取得（常にV/m）"""
        return np.asarray(self.Efield)

    def get_time_SI(self) -> np.ndarray:
        """Return the canonical fs time axis (the method name is legacy)."""
        return np.asarray(self.tlist)

    def get_time_in_units(self, target_units: str) -> np.ndarray:
        """指定単位での時間配列を取得"""
        return self._convert_time_from_fs(self.tlist, target_units)

    def get_Efield_in_units(self, target_units: str) -> np.ndarray:
        """指定単位での電場を取得"""
        field_SI = self.get_Efield_SI()  # Get in V/m
        return self._convert_field_from_SI(field_SI, target_units)

    def set_Efield_from_units(self, field_array: np.ndarray, from_units: str) -> None:
        """指定単位から電場を設定"""
        field_SI = self._convert_field_to_SI(field_array, from_units)
        self.Efield = field_SI.copy()

    def get_field_scale_factor(self) -> float:
        """電場スケールファクターを取得"""
        efield_array = np.asarray(self.Efield)
        return float(np.max(np.abs(efield_array)))

    def get_field_scale_info(self, *, field_units: str) -> dict:
        """Return field scale information in explicitly requested units."""
        if field_units not in converter.get_supported_units("field_amplitude"):
            raise ValueError(
                "field_units must be a supported electric-field amplitude unit"
            )
        scale_V_per_m = self.get_field_scale_factor()

        return {
            "scale_V_per_m": scale_V_per_m,
            "scale_MV_per_cm": scale_V_per_m / 1e8,
            "scale_GV_per_m": scale_V_per_m / 1e9,
            "scale_in_requested_units": self._convert_field_from_SI(
                np.array([scale_V_per_m]), field_units
            )[0],
            "requested_units": field_units,
        }

    def get_pol(self) -> np.ndarray:
        """偏光ベクトルを取得"""
        if self._constant_pol is None:
            raise ValueError("Polarisation is time-dependent (use RK4 path).")
        if isinstance(self._constant_pol, bool):
            raise ValueError("Polarisation is time-dependent (use RK4 path).")
        return self._constant_pol

    def get_scalar_field(self) -> np.ndarray:
        """スカラー電場を取得"""
        if self._scalar_field is None:
            raise ValueError("Scalar field is not set.")
        return self._scalar_field

    def get_scalar_and_pol(self) -> tuple[np.ndarray, np.ndarray]:
        """スカラー電場と偏光を取得"""
        if (
            isinstance(self._constant_pol, np.ndarray)
            and self._scalar_field is not None
        ):
            return self._scalar_field.astype(np.float64, copy=False), self._constant_pol
        raise ValueError("Polarisation is time-dependent (use RK4 path).")

    def get_Efield_spectrum(self):
        """電場のスペクトルを取得"""
        E_freq = rfft(self.Efield, axis=0)
        freq = rfftfreq(len(self.tlist), d=(self.tlist[1] - self.tlist[0]))
        self.freq = freq
        self.Efield_FT = E_freq
        return freq, E_freq

    def add_dispersed_Efield(
        self,
        envelope_func,
        duration: float,
        t_center: float,
        carrier_freq: float,
        *,
        duration_units: str,
        t_center_units: str,
        carrier_freq_units: str,
        amplitude: float,
        amplitude_units: str,
        polarization: np.ndarray = np.array([1.0, 0.0]),
        phase_rad: float = 0.0,
        gdd: float | None = None,
        tod: float | None = None,
        gdd_units: str | None = None,
        tod_units: str | None = None,
        const_polarisation: bool | None = None,
    ) -> None:
        """Add a dispersed pulse after one explicit conversion to fs and V/m.

        ``duration_units``, ``t_center_units``, ``carrier_freq_units``, and
        ``amplitude_units`` are required. ``amplitude_units`` accepts direct
        electric-field amplitude units only, not intensity. GDD and TOD each
        require both their value and unit when present; omitting both members
        of a pair applies the exact zero modifier.
        """
        if amplitude_units not in converter.get_supported_units("field_amplitude"):
            raise ValueError(
                "amplitude_units must be a supported electric-field amplitude unit"
            )
        if (gdd is None) != (gdd_units is None):
            raise ValueError("gdd and gdd_units must be provided together")
        if (tod is None) != (tod_units is None):
            raise ValueError("tod and tod_units must be provided together")

        polarization = np.array(polarization, dtype=np.complex128)
        if polarization.shape != (2,):
            raise ValueError("polarization must be a 2-element vector")
        polarization /= np.linalg.norm(polarization)

        # 偏光の一定性を判定
        if const_polarisation is None:  # 従来の自動判定
            if self._constant_pol is None:
                self._constant_pol = polarization.copy()
            elif isinstance(self._constant_pol, np.ndarray):
                if not np.allclose(polarization, self._constant_pol):
                    self._constant_pol = False
        else:  # 明示指定
            if const_polarisation:  # True → 一定と宣言
                self._constant_pol = polarization.copy()
            else:  # False → 可変
                self._constant_pol = False

        # 履歴に追加
        frame = inspect.currentframe()
        if frame is not None:
            args, _, _, values = inspect.getargvalues(frame)
            self.add_history.append({k: values[k] for k in args if k != "self"})

        # 単位変換
        duration_fs = float(converter.convert_time(duration, duration_units, "fs"))
        t_center_fs = float(converter.convert_time(t_center, t_center_units, "fs"))
        amplitude_v_per_m = float(
            converter.convert_electric_field(amplitude, amplitude_units, "V/m")
        )

        # GDD / TOD unit conversion to fs^2, fs^3
        gdd_fs2 = (
            0.0 if gdd is None else float(converter.convert_gdd(gdd, gdd_units, "fs^2"))
        )
        tod_fs3 = (
            0.0 if tod is None else float(converter.convert_tod(tod, tod_units, "fs^3"))
        )

        if carrier_freq_units == "PHz":  # cycles per fs
            cycles_per_fs = carrier_freq
        else:
            # Convert to rad/fs then → cycles/fs
            rad_per_fs = float(
                converter.convert_frequency(carrier_freq, carrier_freq_units, "rad/fs")
            )
            cycles_per_fs = rad_per_fs / (2 * pi)

        # 包絡線とキャリア波の構築
        envelope = (
            envelope_func(self.tlist, t_center_fs, duration_fs) * amplitude_v_per_m
        )
        carrier = np.exp(
            1j * (2 * pi * cycles_per_fs * (self.tlist - t_center_fs) + phase_rad)
        )
        Efield = envelope * carrier
        Efield_vec = np.real(np.outer(Efield, polarization))
        Efield_vec_disp = apply_dispersion(
            self.tlist, Efield_vec, cycles_per_fs, gdd_fs2, tod_fs3
        )
        if isinstance(Efield_vec_disp, tuple):
            Efield_vec_disp = Efield_vec_disp[0]
        Efield_vec_disp = np.asarray(Efield_vec_disp)
        self.Efield += np.real(Efield_vec_disp)

        # Split-Op用スカラー場を保持
        if const_polarisation is True or (
            const_polarisation is None and isinstance(self._constant_pol, np.ndarray)
        ):
            ef_real = np.real(np.asarray(Efield))  # 1次元配列
            ef_real_2d = ef_real.reshape(-1, 1)
            ef_disp = apply_dispersion(
                self.tlist, ef_real_2d, cycles_per_fs, gdd_fs2, tod_fs3
            )
            if isinstance(ef_disp, tuple):
                ef_disp = ef_disp[0]
            ef_disp = np.asarray(ef_disp)
            self._scalar_field = np.real(ef_disp).flatten()

    def apply_sinusoidal_mod(
        self,
        center_freq: float,
        modulation_depth: float,
        delay_fs: float,
        phase_rad: float = 0.0,
        mode: str = "phase",
    ):
        """正弦波変調を適用"""
        self.Efield = apply_sinusoidal_mod(
            self.tlist,
            self.Efield,
            center_freq,
            modulation_depth,
            delay_fs,
            phase_rad,
            mode,
        )
        if self._scalar_field is not None:
            scalar_2d = self._scalar_field.reshape(-1, 1)
            self._scalar_field = apply_sinusoidal_mod(
                self.tlist,
                scalar_2d,
                center_freq,
                modulation_depth,
                delay_fs,
                phase_rad,
                mode,
            ).reshape(-1)

    def apply_binned_mod(
        self,
        initial_freq: int,
        bin_width: int,
        mod_values: np.ndarray,
        mode: str = "phase",
        window: str | None = None,
    ) -> "ElectricField":
        """ビン幅指定変調を適用"""
        spec = get_mod_spectrum_from_bin_setting(
            initial_freq, bin_width, mod_values, self.tlist
        )
        # 窓関数による移動平均（スペクトル平滑化）
        if window:
            win = _select_window(window, bin_width)
            win = win / win.sum()
            spec = np.convolve(spec, win, mode="same")
        # 各偏光成分に適用
        self.Efield = self.apply_arbitrary_mod(spec, mode)
        return self

    def apply_arbitrary_mod(self, mod_spectrum: np.ndarray, mod_type: str = "phase"):
        """任意変調を適用"""
        efield_array = np.asarray(self.Efield)
        E_freq = np.asarray(rfft(efield_array, axis=0))
        if (
            len(mod_spectrum.shape) != len(E_freq.shape)
            or mod_spectrum.shape[0] != E_freq.shape[0]
        ):
            raise ValueError("mod_spectrum shape mismatch")
        E_freq_mod = None
        if mod_type == "phase":
            mod_spectrum = np.clip(mod_spectrum, -1e4, 1e4)
            E_freq_mod = E_freq * np.exp(-1j * mod_spectrum)
        elif mod_type == "amplitude":
            mod_spectrum = np.abs(mod_spectrum)
            E_freq_mod = E_freq * mod_spectrum
        elif mod_type == "both":
            if mod_spectrum.shape[1] != 2:
                raise ValueError("mod_spectrum.shape[1] must be 2 for 'both' mode")
            mod_spectrum[:, 0] = np.clip(mod_spectrum[:, 0], -1e4, 1e4)
            E_freq_mod = (
                E_freq * np.exp(-1j * mod_spectrum[:, 0]) * np.abs(mod_spectrum[:, 1])
            )

        if E_freq_mod is not None:
            self.Efield = irfft(E_freq_mod, axis=0, n=len(self.tlist))
        return self

    def add_arbitrary_Efield(
        self,
        Efield: np.ndarray,
        *,
        field_units: str,
    ) -> "ElectricField":
        """Add arbitrary field samples after explicit conversion to V/m.

        ``field_units`` must describe an electric-field amplitude. Intensity
        units are rejected because an arbitrary signed field array cannot be
        reconstructed from cycle-averaged intensity samples.
        """
        if field_units not in converter.get_supported_units("field_amplitude"):
            raise ValueError(
                "field_units must be a supported electric-field amplitude unit"
            )
        field_si = self._convert_field_to_SI(np.asarray(Efield), field_units)
        efield_array = np.asarray(self.Efield)
        if field_si.shape != efield_array.shape:
            raise ValueError("Efield shape mismatch")
        self.Efield += field_si
        return self

    def plot(self, use_SI_units: bool = True):
        """電場波形をプロット"""
        import matplotlib.pyplot as plt

        if use_SI_units:
            t_plot = self.get_time_SI()
            E_plot = self.get_Efield_SI()
            time_label = "Time (fs)"
            x_label = r"$E_x$ (V/m)"
            y_label = r"$E_y$ (V/m)"
        else:
            t_plot = np.asarray(self.tlist)
            E_plot = np.asarray(self.Efield)
            time_label = "Time (fs)"
            x_label = r"$E_x$ (V/m)"
            y_label = r"$E_y$ (V/m)"

        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].plot(t_plot, E_plot[:, 0])
        ax[1].plot(t_plot, E_plot[:, 1])
        ax[1].set_xlabel(time_label)
        ax[0].set_ylabel(x_label)
        ax[1].set_ylabel(y_label)
        plt.show()

    def plot_spectrum(
        self,
        remove_linear_phase: bool = True,
        freq_range: tuple | None = None,
        t_center: float | None = None,
        center_freq: float | None = None,
        width_fit: float | None = None,
    ):
        """電場のスペクトルをプロット"""
        import matplotlib.pyplot as plt

        freq, E_freq = self.get_Efield_spectrum()
        E_freq = np.asarray(E_freq)

        if t_center is None:
            phase_x = np.unwrap(np.angle(E_freq[:, 0]))
            phase_y = np.unwrap(np.angle(E_freq[:, 1]))
            if remove_linear_phase:
                phase_x = _remove_linear_phase(
                    freq, phase_x, center_freq=center_freq, width_fit=width_fit
                )
                phase_y = _remove_linear_phase(
                    freq, phase_y, center_freq=center_freq, width_fit=width_fit
                )
        else:
            E_freq_comp = E_freq * (np.exp(1j * 2 * pi * freq * t_center)).reshape(
                (len(freq), 1)
            )
            phase_x = np.unwrap(np.angle(E_freq_comp[:, 0]))
            phase_y = np.unwrap(np.angle(E_freq_comp[:, 1]))

        if freq_range is not None:
            mask = (freq >= freq_range[0]) & (freq <= freq_range[1])
            E_freq = E_freq[mask]
            phase_x = phase_x[mask]
            phase_y = phase_y[mask]
            freq = freq[mask]

        # プロット
        fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True)
        ax0.plot(freq, np.abs(E_freq[:, 0]))
        ax1.plot(freq, np.abs(E_freq[:, 1]))
        ax0_r = ax0.twinx()
        ax1_r = ax1.twinx()
        ax0_r.plot(freq, phase_x, color="red", alpha=0.5)
        ax1_r.plot(freq, phase_y, color="red", alpha=0.5)
        ax0_r.set_ylabel(r"$\phi_x$ (rad)")
        ax1_r.set_ylabel(r"$\phi_y$ (rad)")
        ax0.set_xlim(freq_range)
        ax0.set_ylabel(r"$E_x$ (V/m)")
        ax1.set_ylabel(r"$E_y$ (V/m)")
        ax1.set_xlabel("Frequency (rad/fs)")
        plt.show()


class ZeroField(ElectricField):
    """An explicitly field-free time grid.

    A plain :class:`ElectricField` whose samples happen to be zero is treated
    as an incomplete driven-field setup by strict nondimensionalization.
    ``ZeroField`` records that zero interaction is the caller's intent.
    """

    def __init__(self, tlist: np.ndarray, *, time_units: str):
        super().__init__(tlist, time_units=time_units)
        self._constant_pol = np.array([1.0, 0.0], dtype=np.complex128)
        self._scalar_field = np.zeros(self.tlist.size, dtype=np.float64)

    def add_dispersed_Efield(self, *args, **kwargs) -> None:
        raise TypeError("ZeroField cannot contain a pulse; use ElectricField")

    def add_arbitrary_Efield(
        self,
        Efield: np.ndarray,
        *,
        field_units: str,
    ) -> "ElectricField":
        if np.any(np.asarray(Efield) != 0):
            raise TypeError("ZeroField accepts only identically zero samples")
        return super().add_arbitrary_Efield(Efield, field_units=field_units)
