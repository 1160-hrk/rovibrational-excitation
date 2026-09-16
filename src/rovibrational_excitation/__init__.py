"""
rovibrational_excitation
========================
Package for rovibrational wave-packet simulation.

サブモジュール
--------------
core            … 汎用状態、演算子、時間、単位
fields          … 電場波形、包絡線、変調
dynamics        … 時間発展facade、数値solver、実行契約
dipole          … 双極子モーメント行列の高速生成
visualization   … 可視化ユーティリティ
simulation      … バッチ実行・結果管理
spectroscopy    … 線形応答理論による分光計算 (吸収、PFID、放射スペクトルなど)

使用例
------
基本的な波束シミュレーション:
>>> import rovibrational_excitation as rve
>>> basis = rve.LinMolBasis(
...     V_max=2, J_max=4, omega=0.2, B=0.001,
...     alpha=0.0, delta_omega=0.0,
... )
>>> dip = rve.LinMolDipoleMatrix(
...     basis, mu0=1.0e-30, potential_type="harmonic"
... )
>>> H0 = basis.generate_H0()

線形応答分光計算:
`rovibrational_excitation.spectroscopy` の unit-explicit example を参照。
温度、圧力、光路長、コヒーレンス時間、分子質量、波数にはそれぞれ
明示的な単位が必要。
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
from . import (  # noqa: E402, F401
    core,
    dipole,
    fields,
    simulation,
    spectroscopy,
    visualization,
)
from .core.basis import DensityMatrix, StateVector  # noqa: E402, F401
from .core.operators import Hamiltonian  # noqa: E402, F401

# Note: procedural propagators have been removed from public API in favor of class-based propagators
# dipole
from .fields import ElectricField  # noqa: E402, F401
from .models.linear_molecule import (  # noqa: E402, F401
    LinMolBasis,
    LinMolDipoleMatrix,
)

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
