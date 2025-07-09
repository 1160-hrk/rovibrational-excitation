"""
電場モジュール（後方互換性レイヤー）
==================================

このファイルは後方互換性のために残されています。
新しいコードでは以下を使用してください:

from rovibrational_excitation.core.electric_field import ElectricField

無次元化機能は削除され、nondimensional.converter に統一されました。
"""

import warnings

# 新しい構造からインポート
from .electric_field.core import ElectricField
from .electric_field.envelopes import (
    gaussian, 
    lorentzian, 
    voigt, 
    gaussian_fwhm, 
    lorentzian_fwhm, 
    voigt_fwhm
)
from .electric_field.modulation import (
    apply_sinusoidal_mod,
    apply_dispersion,
    get_mod_spectrum_from_bin_setting,
)

# 非推奨警告
warnings.warn(
    "Direct import from electric_field.py is deprecated. "
    "Use 'from rovibrational_excitation.core.electric_field import ElectricField' instead. "
    "Dimensionless functionality has been moved to nondimensional.converter.",
    DeprecationWarning,
    stacklevel=2
)

# 後方互換性のためのエクスポート
ArrayLike = None  # 削除された型エイリアス

__all__ = [
    "ElectricField",
    "gaussian",
    "lorentzian", 
    "voigt",
    "gaussian_fwhm",
    "lorentzian_fwhm",
    "voigt_fwhm",
    "apply_sinusoidal_mod",
    "apply_dispersion",
    "get_mod_spectrum_from_bin_setting",
]
