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

>>> import rovibrational_excitation as rve
>>> basis = rve.LinMolBasis(V_max=2, J_max=4)
>>> dip   = rve.LinMolDipoleMatrix(basis)
>>> H0    = basis.generate_H0(omega_rad_phz=1000.0)  # New API (recommended)
"""

from __future__ import annotations

# ------------------------------------------------------------------
# パッケージメタデータ
# ------------------------------------------------------------------
from importlib.metadata import version, PackageNotFoundError

try:
    __version__: str = version(__name__)
except PackageNotFoundError:    # ソースから直接実行
    __version__ = "0.0.0+dev"

__author__  = "Hiroki Tsusaka"
__all__: list[str] = [
    # re-export される公開 API
    "LinMolBasis",
    "ElectricField",
    "LinMolDipoleMatrix",
    "schrodinger_propagation",
    "liouville_propagation",
    "generate_H0_LinMol",
]

# ------------------------------------------------------------------
# 便利 re-export
# ------------------------------------------------------------------
# core
from .core.basis        import LinMolBasis                       # noqa: E402, F401
from .core.electric_field import ElectricField                   # noqa: E402, F401
from .core.hamiltonian   import generate_H0_LinMol               # noqa: E402, F401  # DEPRECATED: use basis.generate_H0() instead
from .core.propagator    import (                                # noqa: E402, F401
    schrodinger_propagation,
    liouville_propagation,
)

# dipole
from .dipole.linmol.cache import LinMolDipoleMatrix              # noqa: E402, F401

# ------------------------------------------------------------------
# サブパッケージを名前空間に公開（必要なら）
# ------------------------------------------------------------------
from . import core, dipole, plots, simulation                    # noqa: E402, F401

# ------------------------------------------------------------------
# 名前空間のクリーンアップ
# ------------------------------------------------------------------
del version, PackageNotFoundError
