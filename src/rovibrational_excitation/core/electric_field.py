# 電場波形生成
# electric_field.py
import inspect
from typing import Union, Optional, TYPE_CHECKING

import numpy as np
from numpy import pi

# from scipy.fft import fft, ifft, fftfreq
from scipy.fft import irfft, rfft, rfftfreq
from scipy.special import wofz

if TYPE_CHECKING:
    from rovibrational_excitation.core.nondimensionalize import NondimensionalizationScales

ArrayLike = Union[np.ndarray, float]


class ElectricField:
    """
    電場波形を表現するクラス（偏光、包絡線、GDD/TOD付き）
    
    無次元化システムと統合されており、スケールファクターを保持して
    SI単位系と無次元系の間で適切に変換を行います。
    """

    def __init__(
        self, 
        tlist: np.ndarray, 
        time_units: str = "fs",
        field_units: str = "V/m",
        scales: Optional["NondimensionalizationScales"] = None,
        is_dimensionless_interface: bool = False
    ):
        """
        Parameters
        ----------
        tlist : np.ndarray
            時間軸（指定単位）
        time_units : str, default "fs"
            時間の単位 ("fs", "ps", "ns", "s")
        field_units : str, default "V/m"
            電場の単位 ("V/m", "MV/cm", "kV/cm", "GV/m", etc.)
        scales : NondimensionalizationScales, optional
            無次元化スケールファクター。無次元系インターフェースで使用する場合は必須
        is_dimensionless_interface : bool, optional
            インターフェースが無次元系かどうか。デフォルトはFalse
        """
        # 単位情報を保存
        self.time_units = time_units
        self.field_units = field_units
        
        # 時間配列を内部単位（fs）に変換
        self.tlist = self._convert_time_to_fs(np.asarray(tlist), time_units)
        self.dt = self.tlist[1] - self.tlist[0]  # fs単位
        self.dt_state = self.dt * 2
        self.steps_state = len(self.tlist) // 2
        
        # 電場配列を初期化（内部はV/m単位で保持）
        self.Efield = np.zeros((len(self.tlist), 2))  # V/m単位
        self.add_history = []
        self._constant_pol: np.ndarray | None | bool = None
        self._scalar_field: np.ndarray | None = None
        
        # 無次元化関連
        self.scales = scales
        self.is_dimensionless_interface = is_dimensionless_interface
        
        if is_dimensionless_interface and scales is None:
            raise ValueError("scales must be provided when is_dimensionless_interface=True")

    # ------------------------------------------------------------------
    # Unit conversion helpers
    # ------------------------------------------------------------------
    def _convert_time_to_fs(self, time_array: np.ndarray, from_units: str) -> np.ndarray:
        """Convert time array to fs units."""
        conversions = {
            "fs": 1.0,
            "ps": 1e3,
            "ns": 1e6,
            "s": 1e15,
        }
        
        if from_units not in conversions:
            raise ValueError(f"Unknown time unit: {from_units}. Supported: {list(conversions.keys())}")
        
        return time_array * conversions[from_units]
    
    def _convert_time_from_fs(self, time_array_fs: np.ndarray, to_units: str) -> np.ndarray:
        """Convert time array from fs to specified units."""
        conversions = {
            "fs": 1.0,
            "ps": 1e-3,
            "ns": 1e-6,
            "s": 1e-15,
        }
        
        if to_units not in conversions:
            raise ValueError(f"Unknown time unit: {to_units}. Supported: {list(conversions.keys())}")
        
        return time_array_fs * conversions[to_units]
    
    def _convert_field_to_SI(self, field_array: np.ndarray, from_units: str) -> np.ndarray:
        """Convert electric field array to V/m units."""
        conversions = {
            "V/m": 1.0,
            "V/mm": 1e3,
            "V/cm": 1e2,
            "kV/m": 1e3,
            "kV/mm": 1e6,
            "kV/cm": 1e5,
            "MV/m": 1e6,
            "MV/mm": 1e9,
            "MV/cm": 1e8,
            "GV/m": 1e9,
            "GV/cm": 1e11,
            "TV/m": 1e12,
        }
        
        if from_units not in conversions:
            raise ValueError(f"Unknown field unit: {from_units}. Supported: {list(conversions.keys())}")
        
        return field_array * conversions[from_units]
    
    def _convert_field_from_SI(self, field_array_SI: np.ndarray, to_units: str) -> np.ndarray:
        """Convert electric field array from V/m to specified units."""
        conversions = {
            "V/m": 1.0,
            "V/mm": 1e-3,
            "V/cm": 1e-2,
            "kV/m": 1e-3,
            "kV/mm": 1e-6,
            "kV/cm": 1e-5,
            "MV/m": 1e-6,
            "MV/mm": 1e-9,
            "MV/cm": 1e-8,
            "GV/m": 1e-9,
            "GV/cm": 1e-11,
            "TV/m": 1e-12,
        }
        
        if to_units not in conversions:
            raise ValueError(f"Unknown field unit: {to_units}. Supported: {list(conversions.keys())}")
        
        return field_array_SI * conversions[to_units]

    def init_Efield(self) -> "ElectricField":
        """
        電場をゼロに初期化
        """
        self.Efield = np.zeros((len(self.tlist), 2))
        return self

    def get_Efield(self) -> np.ndarray:
        """
        電場を取得（インターフェースに応じて単位が変わる）
        Returns
        -------
        np.ndarray
            電場。is_dimensionless_interface=Trueの場合は無次元、Falseの場合はV/m
        """
        efield_array = np.asarray(self.Efield)
        if self.is_dimensionless_interface:
            if self.scales is not None:
                # Legacy: 外部スケールファクターを使用
                return efield_array / self.scales.Efield0
            else:
                # New: 自己決定スケールファクターを使用
                return self.get_Efield_dimensionless()
        else:
            # そのまま返す（V/m）
            return efield_array

    def get_Efield_SI(self) -> np.ndarray:
        """
        SI単位系での電場を取得（常にV/m）
        Returns
        -------
        np.ndarray
            電場（V/m）
        """
        return np.asarray(self.Efield)

    def get_time_SI(self) -> np.ndarray:
        """
        SI単位系での時間軸を取得（常にfs）
        Returns
        -------
        np.ndarray
            時間軸（fs）
        """
        return np.asarray(self.tlist)
    
    def get_time_in_units(self, target_units: str) -> np.ndarray:
        """
        Get time array in specified units.
        
        Parameters
        ----------
        target_units : str
            Target time units ("fs", "ps", "ns", "s")
            
        Returns
        -------
        np.ndarray
            Time array in target units
        """
        return self._convert_time_from_fs(self.tlist, target_units)
    
    def get_Efield_in_units(self, target_units: str) -> np.ndarray:
        """
        Get electric field in specified units.
        
        Parameters
        ----------
        target_units : str
            Target field units ("V/m", "MV/cm", "kV/cm", etc.)
            
        Returns
        -------
        np.ndarray
            Electric field array in target units
        """
        field_SI = self.get_Efield_SI()  # Get in V/m
        return self._convert_field_from_SI(field_SI, target_units)
    
    def set_Efield_from_units(self, field_array: np.ndarray, from_units: str) -> None:
        """
        Set electric field from array with specified units.
        
        Parameters
        ----------
        field_array : np.ndarray
            Electric field array in specified units
        from_units : str
            Units of the input field array
        """
        # Convert to SI units and store internally
        field_SI = self._convert_field_to_SI(field_array, from_units)
        self.Efield = field_SI.copy()
    
    # ------------------------------------------------------------------
    # Self-determined scale factor
    # ------------------------------------------------------------------
    def get_field_scale_factor(self) -> float:
        """
        Get electric field scale factor from the maximum absolute value of the field.
        
        Returns
        -------
        float
            Electric field scale factor in V/m (SI units)
        """
        efield_array = np.asarray(self.Efield)
        if np.all(efield_array == 0):
            # If field is zero, return a default scale
            return 1e8  # 1 MV/cm as default
        
        return float(np.max(np.abs(efield_array)))
    
    def get_Efield_dimensionless(self) -> np.ndarray:
        """
        Get electric field in dimensionless units using self-determined scale.
        
        Returns
        -------
        np.ndarray
            Dimensionless electric field (E / E_max)
        """
        scale_factor = self.get_field_scale_factor()
        efield_array = np.asarray(self.Efield)
        return efield_array / scale_factor
    
    def get_field_scale_info(self) -> dict:
        """
        Get comprehensive field scale information.
        
        Returns
        -------
        dict
            Dictionary containing scale information
        """
        scale_V_per_m = self.get_field_scale_factor()
        
        return {
            "scale_V_per_m": scale_V_per_m,
            "scale_MV_per_cm": scale_V_per_m / 1e8,
            "scale_GV_per_m": scale_V_per_m / 1e9,
            "scale_in_original_units": self._convert_field_from_SI(
                np.array([scale_V_per_m]), self.field_units
            )[0],
            "original_units": self.field_units,
            "time_scale_fs": None,  # Will be calculated if needed
        }
        
    def get_time_dimensionless(self) -> np.ndarray:
        """
        無次元時間軸を取得
        Returns
        -------
        np.ndarray
            無次元時間
        """
        if self.scales is None:
            raise ValueError("scales is required for dimensionless time")
        # fs → s → 無次元
        tlist_s = self.tlist * 1e-15  # fs → s
        return tlist_s / self.scales.t0

    def get_scalar_and_pol(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        scalar : (N,) float64   実電場
        pol    : (2,) complex   一定偏光
        Raises
        ------
        ValueError : 偏光が時間依存フラグのとき
        """
        if (
            isinstance(self._constant_pol, np.ndarray)
            and self._scalar_field is not None
        ):
            return self._scalar_field.astype(np.float64, copy=False), self._constant_pol
        raise ValueError("Polarisation is time-dependent (use RK4 path).")

    def get_Efield_spectrum(self):
        """
        電場のスペクトルを取得
        Returns
        -------
        np.ndarray
            電場のスペクトル（V/m）
        """
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
        amplitude: float = 1.0,
        polarization: np.ndarray = np.array([1.0, 0.0]),
        phase_rad: float = 0.0,
        gdd: float = 0.0,
        tod: float = 0.0,
        const_polarisation: bool | None = None,
    ):
        polarization = np.array(polarization, dtype=np.complex128)
        if polarization.shape != (2,):
            raise ValueError("polarization must be a 2-element vector")
        polarization /= np.linalg.norm(polarization)
        # --- constant / variable 判定 ---------------------------
        if const_polarisation is None:  # 従来の自動判定
            if self._constant_pol is None:
                self._constant_pol = polarization.copy()
            elif isinstance(self._constant_pol, np.ndarray):
                if not np.allclose(polarization, self._constant_pol):
                    self._constant_pol = False
        else:  # ★NEW : 明示指定
            if const_polarisation:  # True → 一定と宣言
                self._constant_pol = polarization.copy()
            else:  # False → 可変
                self._constant_pol = False
        if self._constant_pol is None:
            self._constant_pol = polarization.copy()
        elif not np.allclose(polarization, self._constant_pol):
            # 途中で別偏光が来たら可変と見なす
            self._constant_pol = False
        # -------- add args to history -------
        frame = inspect.currentframe()
        if frame is not None:
            args, _, _, values = inspect.getargvalues(frame)
            self.add_history.append({k: values[k] for k in args if k != "self"})
        # -------- set Efield ---------------------
        envelope = envelope_func(self.tlist, t_center, duration) * amplitude
        carrier = np.exp(
            1j * (2 * pi * carrier_freq * (self.tlist - t_center) + phase_rad)
        )
        Efield = envelope * carrier
        Efield_vec = np.real(np.outer(Efield, polarization))
        Efield_vec_disp = apply_dispersion(
            self.tlist, Efield_vec, carrier_freq, gdd, tod
        )
        if isinstance(Efield_vec_disp, tuple):
            Efield_vec_disp = Efield_vec_disp[0]
        Efield_vec_disp = np.asarray(Efield_vec_disp)
        self.Efield += np.real(Efield_vec_disp)
        # --- Split-Op 用スカラー場を保持 ------------------------ ★NEW
        if const_polarisation is True or (
            const_polarisation is None and isinstance(self._constant_pol, np.ndarray)
        ):
            # 保存しておく（実数に変換）
            ef_real = np.real(np.asarray(Efield))  # 1次元配列
            # apply_dispersionは2次元配列を期待するので、reshapeしてから適用
            ef_real_2d = ef_real.reshape(-1, 1)
            ef_disp = apply_dispersion(self.tlist, ef_real_2d, carrier_freq, gdd, tod)
            if isinstance(ef_disp, tuple):
                ef_disp = ef_disp[0]
            ef_disp = np.asarray(ef_disp)
            # 1次元に戻す
            self._scalar_field = np.real(ef_disp).flatten()

    def add_dispersed_Efield_dimensionless(
        self,
        envelope_func,
        duration_dimensionless: float,
        t_center_dimensionless: float,
        carrier_freq_dimensionless: float,
        amplitude_dimensionless: float = 1.0,
        polarization: np.ndarray = np.array([1.0, 0.0]),
        phase_rad: float = 0.0,
        gdd_dimensionless: float = 0.0,
        tod_dimensionless: float = 0.0,
        const_polarisation: bool | None = None,
    ):
        """
        無次元パラメータから有次元電場を生成して内部電場に追加
        
        Parameters
        ----------
        envelope_func : callable
            包絡線関数
        duration_dimensionless : float
            パルス幅（無次元時間）
        t_center_dimensionless : float
            中心時刻（無次元時間）
        carrier_freq_dimensionless : float
            キャリア周波数（無次元周波数）
        amplitude_dimensionless : float
            振幅（無次元電場）
        polarization : np.ndarray
            偏光ベクトル
        phase_rad : float
            位相（rad）
        gdd_dimensionless : float
            群遅延分散（無次元単位）
        tod_dimensionless : float
            三次分散（無次元単位）
        const_polarisation : bool, optional
            偏光が一定かどうか
        """
        if not self.is_dimensionless_interface:
            raise ValueError("This method is only for dimensionless interface ElectricField")
        
        if self.scales is None:
            raise ValueError("scales is required for dimensionless operations")
        
        # 無次元パラメータを有次元に変換
        t0_fs = self.scales.t0 * 1e15  # s → fs
        duration_fs = duration_dimensionless * t0_fs
        t_center_fs = t_center_dimensionless * t0_fs
        
        # 周波数: 無次元 → rad/fs
        carrier_freq_rad_fs = carrier_freq_dimensionless / t0_fs
        
        # 電場振幅: 無次元 → V/m
        amplitude_Vm = amplitude_dimensionless * self.scales.Efield0
        
        # 分散パラメータ: 無次元 → fs^2, fs^3
        gdd_fs2 = gdd_dimensionless * (t0_fs ** 2)
        tod_fs3 = tod_dimensionless * (t0_fs ** 3)
        
        # 有次元パラメータで電場生成（内部は常にfs, V/m）
        self.add_dispersed_Efield(
            envelope_func, duration_fs, t_center_fs, carrier_freq_rad_fs,
            amplitude_Vm, polarization, phase_rad, gdd_fs2, tod_fs3, const_polarisation
        )

    @classmethod
    def create_dimensionless_interface(
        cls,
        tlist_dimensionless: np.ndarray,
        scales: "NondimensionalizationScales"
    ) -> "ElectricField":
        """
        無次元インターフェースでElectricFieldを作成（内部はfs単位で保持）
        
        Parameters
        ----------
        tlist_dimensionless : np.ndarray
            無次元時間配列
        scales : NondimensionalizationScales
            無次元化スケールファクター
            
        Returns
        -------
        ElectricField
            無次元インターフェースElectricFieldインスタンス
        """
        # 無次元時間をfs単位に変換して内部保持
        t0_fs = scales.t0 * 1e15  # s → fs
        tlist_fs = tlist_dimensionless * t0_fs
        return cls(tlist_fs, scales=scales, is_dimensionless_interface=True)

    @classmethod 
    def create_from_SI(
        cls,
        tlist_fs: np.ndarray
    ) -> "ElectricField":
        """
        SI単位系（fs, V/m）でElectricFieldを作成
        
        Parameters
        ----------
        tlist_fs : np.ndarray
            時間配列（fs）
            
        Returns
        -------
        ElectricField
            SI単位系ElectricFieldインスタンス
        """
        return cls(tlist_fs, time_units="fs", field_units="V/m", 
                  scales=None, is_dimensionless_interface=False)
    
    @classmethod
    def create_with_units(
        cls,
        tlist: np.ndarray,
        time_units: str,
        field_units: str
    ) -> "ElectricField":
        """
        指定単位でElectricFieldを作成
        
        Parameters
        ----------
        tlist : np.ndarray
            時間配列（指定単位）
        time_units : str
            時間の単位
        field_units : str
            電場の単位
            
        Returns
        -------
        ElectricField
            単位指定ElectricFieldインスタンス
        """
        return cls(tlist, time_units=time_units, field_units=field_units)

    def convert_to_dimensionless_interface(
        self, 
        scales: "NondimensionalizationScales"
    ) -> "ElectricField":
        """
        SI単位系インターフェースを無次元系インターフェースに変換
        
        Parameters
        ----------
        scales : NondimensionalizationScales
            無次元化スケールファクター
            
        Returns
        -------
        ElectricField
            無次元インターフェースの新しいElectricFieldインスタンス
        """
        if self.is_dimensionless_interface:
            raise ValueError("Already dimensionless interface")
        
        # 新しいインスタンスを作成（内部は同じfs, V/m を保持）
        new_field = ElectricField(
            self.tlist.copy(), 
            scales=scales, 
            is_dimensionless_interface=True
        )
        new_field.Efield = np.asarray(self.Efield).copy()
        new_field._constant_pol = self._constant_pol
        if self._scalar_field is not None:
            new_field._scalar_field = np.asarray(self._scalar_field).copy()
        
        return new_field

    def apply_sinusoidal_mod(
        self,
        center_freq: float,
        amplitude: float,
        carrier_freq: float,
        phase_rad: float = 0.0,
        type_mod: str = "phase",
    ):
        """
        Parameters
        ----------
        center_freq : float
            中心周波数（rad/fs）
        amplitude_ratio : float
            振幅比
        carrier_freq : float
            キャリア周波数（rad/fs）
        phase_rad : float, optional
            位相（rad）, by default 0.0
        type_mod : str, optional
            "phase" or "amplitude", by default "phase"
        """
        self.Efield = apply_sinusoidal_mod(
            self.tlist,
            self.Efield,
            center_freq,
            amplitude,
            carrier_freq,
            phase_rad,
            type_mod,
        )

    def apply_binned_mod(
        self,
        initial_freq: int,
        bin_width: int,
        mod_values: np.ndarray,
        mode: str = "phase",
        window: str | None = None,
    ) -> "ElectricField":
        """
        ビン幅指定変調 + オプションで移動平均(窓関数)を適用

        Parameters
        ----------
        min_index : int
            周波数スペクトル配列の開始インデックス
        bin_width : int
            ビン幅（インデックス単位）
        mod_values : np.ndarray
            1D (nbins,) または (nbins,2)配列
        mode : str
            'phase', 'amp'  または両方('both')
        window : str, optional
            'blackman', 'hamming', 'hann' のいずれかを移動平均窓関数として適用
        """

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
        """
        Parameters
        ----------
        mod_spectrum : np.ndarray
            モジュレーションスペクトル（len(tlist), 2） or (len(tlist),1)
        mod_type : str, optional
            "phase" or "amplitude", by default "phase"
        """
        if (
            len(mod_spectrum.shape) != len(self.Efield.shape)
            or mod_spectrum.shape[0] != self.Efield.shape[0]
        ):
            raise ValueError("mod_spectrum shape mismatch")
        E_freq = rfft(self.Efield, axis=0)
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
        self.Efield = irfft(E_freq_mod, axis=0, n=len(self.tlist))
        return self

    def add_arbitrary_Efield(self, Efield: np.ndarray):
        """
        Parameters
        ----------
        Efield : np.ndarray (len(tlist), 2)
            電場（V/m）
        """
        if Efield.shape != self.Efield.shape:
            raise ValueError("Efield shape mismatch")
        self.Efield += Efield
        return self

    def plot(self, use_SI_units: bool = True):
        """
        電場波形をプロット
        
        Parameters
        ----------
        use_SI_units : bool, optional
            SI単位系でプロットするかどうか。デフォルトはTrue
        """
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
            if self.is_dimensionless:
                time_label = "Time (dimensionless)"
                x_label = r"$E'_x$ (dimensionless)"
                y_label = r"$E'_y$ (dimensionless)"
            else:
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
        """
        電場のスペクトルをプロット
        Parameters
        ----------
        one_sided : bool, optional
            片側スペクトルを表示するかどうか, by default True
        """
        import matplotlib.pyplot as plt

        freq, E_freq = self.get_Efield_spectrum()
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
            E_freq = E_freq[(freq >= freq_range[0]) & (freq <= freq_range[1])]
            phase_x = phase_x[(freq >= freq_range[0]) & (freq <= freq_range[1])]
            phase_y = phase_y[(freq >= freq_range[0]) & (freq <= freq_range[1])]
            freq = freq[(freq >= freq_range[0]) & (freq <= freq_range[1])]
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
        # ax[0].set_xticklabels([])
        ax0.set_xlim(freq_range)
        ax0.set_ylabel(r"$E_x$ (V/m)")
        ax1.set_ylabel(r"$E_y$ (V/m)")
        ax1.set_xlabel("Frequency (rad/fs)")
        plt.show()


def _remove_linear_phase(
    freq_p, phase_p, center_freq=None, width_fit=None, return_t0=False
):
    """
    フーリエスペクトル X の位相から線形成分を除去して返す。

    Args:
        X (np.ndarray): 複素スペクトル（np.fft.fft の出力）
        dt (float): サンプリング間隔（秒）
    Returns:
        phi_corr (np.ndarray): 線形位相を除去したあとの位相（ラジアン）
        t0 (float): 推定された時間シフト [秒]
    """
    # 位相をアンラップ
    phase_fit = phase_p.copy()
    freq_fit = freq_p.copy()
    # 1次フィッティング：phi ≈ a * f + b
    if center_freq is not None and width_fit is not None:
        width_fit = min(width_fit, center_freq)
        # 中心周波数とフィッティング幅を指定
        freq_fit = freq_fit[
            (freq_p >= center_freq - width_fit) & (freq_p <= center_freq + width_fit)
        ]
        phase_fit = phase_fit[
            (freq_p >= center_freq - width_fit) & (freq_p <= center_freq + width_fit)
        ]
    a, b = np.polyfit(freq_fit, phase_fit, 1)
    # 線形位相 φ_lin(f) = a f + b を差し引く
    center_freq = center_freq if center_freq is not None else 0
    phase_corr = phase_p - (a * (freq_p) + b)
    if return_t0:
        # ここで a ≃ -2π t0 なので，時間シフト t0 を推定しておきたい場合：
        t0 = -a / (2 * np.pi)
        return phase_corr, t0
    else:
        return phase_corr


def get_mod_spectrum_from_bin_setting(
    initial_freq: float,
    bin_width: float,
    mod_values: np.ndarray,
    tlist: np.ndarray,
):
    mod_values = np.array(mod_values)
    freq = rfftfreq(len(tlist), d=(tlist[1] - tlist[0]))
    df = freq[1] - freq[0]
    initial_index = int(initial_freq / df)
    bin_width = int(bin_width / df)
    nbins = mod_values.shape[0]
    ndim = len(mod_values.shape)
    # 初期スペクトルをゼロ
    spec = np.zeros_like((len(freq), ndim), dtype=np.complex128)
    # ビンごとに値を設定
    len_freq = len(freq)
    for i in range(nbins):
        start = initial_index + i * bin_width
        end = start + bin_width
        if start > len_freq - 1:
            continue
        if end > len_freq - 1:
            end = len_freq - 1
        val = mod_values[i]
        spec[start:end] = val
    return spec


def _select_window(name: str, length: int) -> np.ndarray:
    name = name.lower()
    if name == "blackman":
        return np.blackman(length)
    if name == "hamming":
        return np.hamming(length)
    if name == "hann":
        return np.hanning(length)
    raise ValueError(f"Unknown window: {name}")


def apply_sinusoidal_mod(
    tlist, Efield, center_freq, amplitude, carrier_freq, phase_rad=0.0, type_mod="phase"
):
    freq = rfftfreq(len(tlist), d=(tlist[1] - tlist[0]))
    E_freq = rfft(Efield, axis=0)
    factor = (
        amplitude * np.sin(carrier_freq * (freq - center_freq) + phase_rad) + amplitude
    )
    factor = factor.reshape((len(freq), 1))
    if type_mod == "phase":
        factor = np.clip(factor, -1e4, 1e4)  # 位相のクリッピング
        E_freq_mod = E_freq * np.exp(-1j * factor)
    else:
        factor = np.abs(factor)
        E_freq_mod = E_freq * factor
    return irfft(E_freq_mod, axis=0, n=len(tlist))


def apply_dispersion(tlist, Efield, center_freq, gdd=0.0, tod=0.0):
    """
    GDDとTODを複素電場に適用

    Parameters
    ----------
    tlist : np.ndarray
    Efield : np.ndarray (real or complex)
    center_freq : float
    gdd : float
    tod : float

    Returns
    -------
    np.ndarray
        分散適用後の電場（complex）
    """
    # 元のEfieldが複素数かどうかをチェック
    is_complex_input = np.iscomplexobj(Efield)

    # Efieldを配列に変換
    Efield = np.asarray(Efield)

    # 複素数の場合はfft、実数の場合はrfftを使用
    if is_complex_input or Efield.dtype.kind == "c":
        from scipy.fft import fft, fftfreq, ifft

        freq = fftfreq(len(tlist), d=(tlist[1] - tlist[0]))
        E_freq = fft(Efield, axis=0)

        # 位相計算
        phase = (
            gdd * (2 * pi * (freq - center_freq)) ** 2
            + tod * (2 * pi * (freq - center_freq)) ** 3
        )

        # Efieldの次元に合わせて位相を調整
        if Efield.ndim == 1:
            # 1次元の場合はそのまま
            pass
        elif Efield.ndim == 2:
            # 2次元の場合は位相をreshape
            phase = phase.reshape((len(freq), 1))
        else:
            raise ValueError("Efield must be 1D or 2D array")

        E_freq_disp = E_freq * np.exp(-1j * phase)
        return ifft(E_freq_disp, axis=0, n=len(tlist))
    else:
        freq = rfftfreq(len(tlist), d=(tlist[1] - tlist[0]))
        E_freq = rfft(Efield, axis=0)

        # 位相計算
        phase = (
            gdd * (2 * pi * (freq - center_freq)) ** 2
            + tod * (2 * pi * (freq - center_freq)) ** 3
        )

        # Efieldの次元に合わせて位相を調整
        if Efield.ndim == 1:
            # 1次元の場合はそのまま
            pass
        elif Efield.ndim == 2:
            # 2次元の場合は位相をreshape
            phase = phase.reshape((len(freq), 1))
        else:
            raise ValueError("Efield must be 1D or 2D array")

        E_freq_disp = E_freq * np.exp(-1j * phase)
        return irfft(E_freq_disp, axis=0, n=len(tlist))


# ===== 包絡線関数群 =====


def gaussian(x: ArrayLike, xc: float, sigma: float) -> ArrayLike:
    return np.exp(-((x - xc) ** 2) / (2 * sigma**2))


def lorentzian(x: ArrayLike, xc: float, gamma: float) -> ArrayLike:
    return gamma**2 / ((x - xc) ** 2 + gamma**2)


def voigt(x: ArrayLike, xc: float, sigma: float, gamma: float) -> ArrayLike:
    z = ((x - xc) + 1j * gamma) / (sigma * np.sqrt(2))
    return np.real(wofz(z)) / (sigma * np.sqrt(2 * pi))


def gaussian_fwhm(x: ArrayLike, xc: float, fwhm: float) -> ArrayLike:
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return gaussian(x, xc, sigma)


def lorentzian_fwhm(x: ArrayLike, xc: float, fwhm: float) -> ArrayLike:
    gamma = fwhm / 2
    return lorentzian(x, xc, gamma)


def voigt_fwhm(x: ArrayLike, xc: float, fwhm_g: float, fwhm_l: float) -> ArrayLike:
    sigma = fwhm_g / (2 * np.sqrt(2 * np.log(2)))
    gamma = fwhm_l / 2
    return voigt(x, xc, sigma, gamma)

"""
使用例：
========

1. 無次元系での電場作成（推奨）:

```python
from rovibrational_excitation.core.nondimensionalize import nondimensionalize_with_SI_base_units
from rovibrational_excitation.core.electric_field import ElectricField, gaussian_fwhm

# 1. 無次元化の実行
H0_prime, mu_x_prime, mu_y_prime, Efield_prime, tlist_prime, dt_prime, scales = \\
    nondimensionalize_with_SI_base_units(H0, mu_x, mu_y, efield_dummy)

# 2. 無次元系での電場作成
efield = ElectricField.create_dimensionless(tlist_prime, scales)

# 3. 無次元系でのパルス追加
efield.add_dispersed_Efield_dimensionless(
    envelope_func=gaussian_fwhm,
    duration=10.0,      # 無次元時間
    t_center=50.0,      # 無次元時間
    carrier_freq=1.5,   # 無次元周波数
    amplitude=0.8,      # 無次元電場強度
    gdd=0.1,           # 無次元GDD
    tod=0.05           # 無次元TOD
)

# 4. プロット（自動的にSI単位系に変換）
efield.plot(use_SI_units=True)   # fs, V/m でプロット
efield.plot(use_SI_units=False)  # 無次元でプロット
```

2. 既存コードからの移行:

```python
# 既存のSI単位系電場
efield_SI = ElectricField.create_from_SI(tlist_fs)
efield_SI.add_dispersed_Efield(...)

# 無次元系に変換
efield_dimensionless = efield_SI.convert_to_dimensionless(scales)
```

3. データ取得:

```python
# 無次元データ
time_dim = efield.tlist          # 無次元時間
field_dim = efield.get_Efield()  # 無次元電場

# SI単位系データ
time_fs = efield.get_time_SI()     # fs単位の時間
field_Vm = efield.get_Efield_SI()  # V/m単位の電場
```

利点：
- スケールファクターが自動管理される
- プロット時に適切な単位で表示される
- 数値計算が安定する
- 無次元系と物理単位系の間で透明な変換が可能
"""
