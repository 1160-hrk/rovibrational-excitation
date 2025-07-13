"""
電場モジュール
==============

電場波形の生成、変調、解析機能を提供。

主要コンポーネント:
- ElectricField: メインの電場クラス
- envelopes: 包絡線関数群
- modulation: 変調関数群

無次元化機能は nondimensional.converter に統一されています。
"""

from .core import ElectricField
from .envelopes import (
    gaussian, 
    lorentzian, 
    voigt, 
    gaussian_fwhm, 
    lorentzian_fwhm, 
    voigt_fwhm
)
from .modulation import (
    apply_sinusoidal_mod,
    apply_dispersion,
    get_mod_spectrum_from_bin_setting,
)

__all__ = [
    "ElectricField",
    # Envelope functions
    "gaussian",
    "lorentzian", 
    "voigt",
    "gaussian_fwhm",
    "lorentzian_fwhm",
    "voigt_fwhm",
    # Modulation functions
    "apply_sinusoidal_mod",
    "apply_dispersion",
    "get_mod_spectrum_from_bin_setting",
] 