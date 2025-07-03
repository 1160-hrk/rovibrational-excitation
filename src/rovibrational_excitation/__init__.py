"""
rovibrational_excitation
========================
Package for rovibrational wave-packet simulation.

サブモジュール
--------------
core            … 低レベル数値計算 (Hamiltonian, RK4 propagator など)
dipole          … 双極子モーメント行列の高速生成
plots           … 可視化ユーティリティ
simulation      … バッチ実行・結果管理
spectroscopy    … 線形応答理論による分光計算 (吸収、PFID、放射スペクトルなど)

使用例
------
基本的な波束シミュレーション:
>>> import rovibrational_excitation as rve
>>> basis = rve.LinMolBasis(V_max=2, J_max=4)
>>> dip   = rve.LinMolDipoleMatrix(basis)
>>> H0    = basis.generate_H0(omega_rad_phz=1000.0)  # New API (recommended)

線形応答分光計算:
>>> # Modern API (推奨)
>>> calc = rve.LinearResponseCalculator()
>>> calc.initialize(3, 10, spectroscopy_params=rve.SpectroscopyParameters())
>>> spectrum = rve.calculate_absorption_spectrum(rho_thermal, calc)
>>> 
>>> # Legacy API (後方互換性)
>>> rve.prepare_variables(Nv=3, Nj=10, T2=500)
>>> spectrum = rve.absorbance_spectrum_for_loop(rho_thermal)
"""

from __future__ import annotations

# ------------------------------------------------------------------
# パッケージメタデータ
# ------------------------------------------------------------------
from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version(__name__)
except PackageNotFoundError:  # ソースから直接実行
    __version__ = "0.0.0+dev"

__author__ = "Hiroki Tsusaka"
__all__: list[str] = [
    # Core API (波束シミュレーション)
    "LinMolBasis",
    "Hamiltonian",
    "StateVector",
    "DensityMatrix",
    "ElectricField", 
    "LinMolDipoleMatrix",
    "schrodinger_propagation",
    "liouville_propagation",
    
    # Unit management
    "auto_convert_parameters",
    "create_hamiltonian_from_input_units", 
    "print_unit_help",
    
    # Spectroscopy API (線形応答分光)
    # Modern API
    "LinearResponseCalculator",
    "SpectroscopyParameters", 
    "MolecularParameters",
    "calculate_absorption_spectrum",
    "calculate_pfid_spectrum",
    "calculate_emission_spectrum",
    "calculate_absorption_from_hamiltonian",
    
    # Legacy API (後方互換性)
    "prepare_variables",
    "absorbance_spectrum_for_loop",
    "absorbance_spectrum_w_doppler_broadening",
    "PFID_spectrum_for_loop",
    "radiation_spectrum_for_loop",
    "absorbance_spectrum_from_rho_and_mu",
    
    # Broadening functions
    "doppler",
    "sinc",
    "sinc_square",
    "convolution_w_doppler",
    "convolution_w_sinc",
    "convolution_w_sinc_square",
]

# ------------------------------------------------------------------
# 便利 re-export
# ------------------------------------------------------------------
# core
# ------------------------------------------------------------------
# サブパッケージを名前空間に公開（必要なら）
# ------------------------------------------------------------------
from . import core, dipole, plots, simulation, spectroscopy  # noqa: E402, F401
from .core.basis import LinMolBasis, Hamiltonian, StateVector, DensityMatrix  # noqa: E402, F401
from .core.electric_field import ElectricField  # noqa: E402, F401

# Note: generate_H0_LinMol is deprecated - use basis.generate_H0() instead
from .core.propagator import (  # noqa: E402, F401
    liouville_propagation,
    schrodinger_propagation,
)

# dipole
from .dipole.linmol.cache import LinMolDipoleMatrix  # noqa: E402, F401

# spectroscopy - Modern API (推奨)
from .spectroscopy import (  # noqa: E402, F401
    LinearResponseCalculator,
    SpectroscopyParameters,
    MolecularParameters,
    calculate_absorption_spectrum,
    calculate_pfid_spectrum,
    calculate_emission_spectrum,
    calculate_absorption_from_hamiltonian,
)

# spectroscopy - Legacy API (後方互換性)
from .spectroscopy import (  # noqa: E402, F401
    prepare_variables,
    absorbance_spectrum_for_loop,
    absorbance_spectrum_w_doppler_broadening,
    PFID_spectrum_for_loop,
    radiation_spectrum_for_loop,
    absorbance_spectrum_from_rho_and_mu,
)

# spectroscopy - Broadening functions
from .spectroscopy import (  # noqa: E402, F401
    doppler,
    sinc,
    sinc_square,
    convolution_w_doppler,
    convolution_w_sinc,
    convolution_w_sinc_square,
)

# ------------------------------------------------------------------
# 名前空間のクリーンアップ
# ------------------------------------------------------------------
del version, PackageNotFoundError
