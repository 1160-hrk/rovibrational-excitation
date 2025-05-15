# 電場波形生成
# electric_field.py
import numpy as np
from numpy import pi
from scipy.fft import fft, ifft, fftfreq
from typing import Union
from scipy.special import erf as scipy_erf, wofz
import inspect

ArrayLike = Union[np.ndarray, float]

class ElectricField:
    """
    電場波形を表現するクラス（偏光、包絡線、GDD/TOD付き）
    """
    def __init__(self, tlist):
        """
        Parameters
        ----------
        tlist : np.ndarray
            時間軸（fs）
        """
        self.tlist = tlist
        self.dt = (tlist[1] - tlist[0])
        self.dt_state = self.dt * 2
        self.steps_state = len(tlist) // 2
        self.Efield = np.zeros((len(tlist), 2))
        self.add_history = []
    
    def add_Efield_disp(
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
    ):
        polarization = np.array(polarization, dtype=np.complex128)
        if polarization.shape != (2,):
            raise ValueError("polarization must be a 2-element vector")
        polarization /= np.linalg.norm(polarization)
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        self.add_history.append({k: values[k] for k in args})
        envelope = envelope_func(self.tlist, t_center, duration) * amplitude
        carrier = np.exp(1j * (2 * pi * carrier_freq * (self.tlist-t_center)+phase_rad))
        Efield = envelope * carrier
        Efield_disp = apply_dispersion(self.tlist, Efield, carrier_freq, gdd, tod)
        Efield_vec = np.real(np.outer(Efield_disp, polarization))
        self.Efield += Efield_vec

    def plot(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].plot(self.tlist, self.Efield[:, 0])
        ax[1].plot(self.tlist, self.Efield[:, 1])
        # ax[0].set_xticklabels([])
        ax[1].set_xlabel("Time (fs)")
        ax[0].set_ylabel(r"$E_x$ (V/m)")
        ax[1].set_ylabel(r"$E_y$ (V/m)")
        plt.show()

def apply_dispersion(tlist, Efield, center_freq, gdd=0.0, tod=0.0):
    """
    GDDとTODを複素電場に適用

    Parameters
    ----------
    tlist : np.ndarray
    Efield : np.ndarray (complex)
    center_freq : float
    gdd : float
    tod : float

    Returns
    -------
    np.ndarray
        分散適用後の電場（complex）
    """
    freq = fftfreq(len(tlist), d=(tlist[1] - tlist[0]))
    E_freq = fft(Efield)
    phase = np.where(
            freq >= 0,
            (gdd * (2*pi*(freq - center_freq))**2 + tod * (2*pi*(freq - center_freq))**3),
            (-gdd * (2*pi*(freq + center_freq))**2 + tod * (2*pi*(freq + center_freq))**3)  
        )
    phase = np.clip(phase, -1e4, 1e4)  # 位相のクリッピング
    E_freq_disp = E_freq * np.exp(-1j * phase)
    return ifft(E_freq_disp)

# ===== 包絡線関数群 =====

def gaussian(x: ArrayLike, xc: float, sigma: float) -> ArrayLike:
    return np.exp(-((x - xc)**2) / (2 * sigma**2))

def lorentzian(x: ArrayLike, xc: float, gamma: float) -> ArrayLike:
    return gamma**2 / ((x - xc)**2 + gamma**2)

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
