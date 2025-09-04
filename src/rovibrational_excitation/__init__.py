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
    
    
    # Spectroscopy public API (最小限)
    "AbsorbanceCalculator",
    "ExperimentalConditions",
    "create_calculator_from_params",
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

# Note: procedural propagators have been removed from public API in favor of class-based propagators

# dipole
from .dipole.linmol.cache import LinMolDipoleMatrix  # noqa: E402, F401

# spectroscopy - Modern API (推奨)
from .spectroscopy import (
    AbsorbanceCalculator,
    ExperimentalConditions,
    create_calculator_from_params,
)

# ------------------------------------------------------------------
# 名前空間のクリーンアップ
# ------------------------------------------------------------------
del version, PackageNotFoundError
