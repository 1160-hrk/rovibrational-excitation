"""
包絡線関数群
============

電場パルスの包絡線を生成するための関数群。
ガウシアン、ローレンツィアン、フォークト関数などを提供。
"""

from typing import Union
import numpy as np
from scipy.special import wofz

ArrayLike = Union[np.ndarray, float]


def gaussian(x: ArrayLike, xc: float, sigma: float) -> ArrayLike:
    """
    ガウシアン関数
    
    Parameters
    ----------
    x : ArrayLike
        入力変数
    xc : float
        中心位置
    sigma : float
        標準偏差
        
    Returns
    -------
    ArrayLike
        ガウシアン関数の値
    """
    return np.exp(-((x - xc) ** 2) / (2 * sigma**2))


def lorentzian(x: ArrayLike, xc: float, gamma: float) -> ArrayLike:
    """
    ローレンツィアン関数
    
    Parameters
    ----------
    x : ArrayLike
        入力変数
    xc : float
        中心位置
    gamma : float
        半値半幅
        
    Returns
    -------
    ArrayLike
        ローレンツィアン関数の値
    """
    return gamma**2 / ((x - xc) ** 2 + gamma**2)


def voigt(x: ArrayLike, xc: float, sigma: float, gamma: float) -> ArrayLike:
    """
    フォークト関数（ガウシアンとローレンツィアンの畳み込み）
    
    Parameters
    ----------
    x : ArrayLike
        入力変数
    xc : float
        中心位置
    sigma : float
        ガウシアン成分の標準偏差
    gamma : float
        ローレンツィアン成分の半値半幅
        
    Returns
    -------
    ArrayLike
        フォークト関数の値
    """
    z = ((x - xc) + 1j * gamma) / (sigma * np.sqrt(2))
    return np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))


def gaussian_fwhm(x: ArrayLike, xc: float, fwhm: float) -> ArrayLike:
    """
    FWHM指定のガウシアン関数
    
    Parameters
    ----------
    x : ArrayLike
        入力変数
    xc : float
        中心位置
    fwhm : float
        半値全幅（FWHM）
        
    Returns
    -------
    ArrayLike
        ガウシアン関数の値
    """
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return gaussian(x, xc, sigma)


def lorentzian_fwhm(x: ArrayLike, xc: float, fwhm: float) -> ArrayLike:
    """
    FWHM指定のローレンツィアン関数
    
    Parameters
    ----------
    x : ArrayLike
        入力変数
    xc : float
        中心位置
    fwhm : float
        半値全幅（FWHM）
        
    Returns
    -------
    ArrayLike
        ローレンツィアン関数の値
    """
    gamma = fwhm / 2
    return lorentzian(x, xc, gamma)


def voigt_fwhm(x: ArrayLike, xc: float, fwhm_g: float, fwhm_l: float) -> ArrayLike:
    """
    FWHM指定のフォークト関数
    
    Parameters
    ----------
    x : ArrayLike
        入力変数
    xc : float
        中心位置
    fwhm_g : float
        ガウシアン成分のFWHM
    fwhm_l : float
        ローレンツィアン成分のFWHM
        
    Returns
    -------
    ArrayLike
        フォークト関数の値
    """
    sigma = fwhm_g / (2 * np.sqrt(2 * np.log(2)))
    gamma = fwhm_l / 2
    return voigt(x, xc, sigma, gamma) 