"""
変調関数群
==========

電場スペクトルの変調を行うための関数群。
正弦波変調、任意変調、分散適用などを提供。
"""

import numpy as np
from numpy import pi
from scipy.fft import irfft, rfft, rfftfreq
from typing import Union


def apply_sinusoidal_mod(
    tlist: np.ndarray, 
    Efield: np.ndarray, 
    center_freq: float, 
    amplitude: float, 
    carrier_freq: float, 
    phase_rad: float = 0.0, 
    type_mod: str = "phase"
) -> np.ndarray:
    """
    正弦波変調を適用
    
    Parameters
    ----------
    tlist : np.ndarray
        時間配列
    Efield : np.ndarray
        電場配列
    center_freq : float
        中心周波数（rad/fs）
    amplitude : float
        変調振幅
    carrier_freq : float
        キャリア周波数（rad/fs）
    phase_rad : float, optional
        位相（rad）, デフォルト: 0.0
    type_mod : str, optional
        "phase" または "amplitude", デフォルト: "phase"
        
    Returns
    -------
    np.ndarray
        変調後の電場
    """
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
    return np.asarray(irfft(E_freq_mod, axis=0, n=len(tlist)))


def apply_dispersion(
    tlist: np.ndarray, 
    Efield: np.ndarray, 
    center_freq: float, 
    gdd: float = 0.0, 
    tod: float = 0.0
) -> np.ndarray:
    """
    GDDとTODを複素電場に適用

    Parameters
    ----------
    tlist : np.ndarray
        時間配列
    Efield : np.ndarray 
        電場配列（実数または複素数）
    center_freq : float
        中心周波数
    gdd : float
        群遅延分散
    tod : float
        三次分散

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
        return np.asarray(ifft(E_freq_disp, axis=0, n=len(tlist)))
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
        return np.asarray(irfft(E_freq_disp, axis=0, n=len(tlist)))


def get_mod_spectrum_from_bin_setting(
    initial_freq: float,
    bin_width: float,
    mod_values: np.ndarray,
    tlist: np.ndarray,
) -> np.ndarray:
    """
    ビン設定から変調スペクトルを生成
    
    Parameters
    ----------
    initial_freq : float
        初期周波数
    bin_width : float
        ビン幅
    mod_values : np.ndarray
        変調値
    tlist : np.ndarray
        時間配列
        
    Returns
    -------
    np.ndarray
        変調スペクトル
    """
    mod_values = np.array(mod_values)
    freq = rfftfreq(len(tlist), d=(tlist[1] - tlist[0]))
    df = freq[1] - freq[0]
    initial_index = int(initial_freq / df)
    bin_width = int(bin_width / df)
    nbins = mod_values.shape[0]
    ndim = len(mod_values.shape)
    # 初期スペクトルをゼロ
    spec = np.zeros((len(freq), ndim), dtype=np.complex128)
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
    """
    窓関数を選択
    
    Parameters
    ----------
    name : str
        窓関数名
    length : int
        長さ
        
    Returns
    -------
    np.ndarray
        窓関数
    """
    name = name.lower()
    if name == "blackman":
        return np.blackman(length)
    if name == "hamming":
        return np.hamming(length)
    if name == "hann":
        return np.hanning(length)
    raise ValueError(f"Unknown window: {name}")


def _remove_linear_phase(
    freq_p: np.ndarray, 
    phase_p: np.ndarray, 
    center_freq: Union[float, None] = None, 
    width_fit: Union[float, None] = None, 
    return_t0: bool = False
) -> Union[np.ndarray, tuple[np.ndarray, float]]:
    """
    フーリエスペクトルの位相から線形成分を除去して返す。

    Parameters
    ----------
    freq_p : np.ndarray
        周波数配列
    phase_p : np.ndarray
        位相配列
    center_freq : float, optional
        中心周波数
    width_fit : float, optional
        フィッティング幅
    return_t0 : bool, optional
        時間シフトを返すかどうか
        
    Returns
    -------
    Union[np.ndarray, tuple[np.ndarray, float]]
        線形位相を除去した位相、または(位相, 時間シフト)のタプル
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