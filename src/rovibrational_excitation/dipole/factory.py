"""
legacy SymTop 双極子行列を構築する移行用ファクトリーモジュール。

統合済みモデルは各 ``models.*`` パッケージから直接構築します。
"""

from typing import Literal

from rovibrational_excitation.core.basis import (
    BasisBase,
    SymTopBasis,
)

from .symtop import SymTopDipoleMatrix


def create_dipole_matrix(
    basis: BasisBase,
    mu0: float,
    *,
    potential_type: Literal["harmonic", "morse"],
    backend: Literal["numpy", "cupy"] = "numpy",
    dense: bool = True,
    units: Literal["C*m", "D", "ea0"] = "C*m",
    units_input: Literal["C*m", "D", "ea0"] = "C*m",
) -> SymTopDipoleMatrix:
    """
    legacy SymTop 基底から双極子行列インスタンスを生成します。

    Parameters
    ----------
    basis : BasisBase
        量子基底クラスのインスタンス
    mu0 : float
        双極子モーメントの大きさ（units_inputで指定した単位）
    potential_type : {"harmonic", "morse"}
        振動ポテンシャルの種類
    backend : {"numpy", "cupy"}, optional
        計算バックエンド
    dense : bool, optional
        密行列形式を使用するかどうか
    units : {"C*m", "D", "ea0"}, optional
        内部で使用する単位系
    units_input : {"C*m", "D", "ea0"}, optional
        mu0の単位

    Returns
    -------
    SymTopDipoleMatrix
        基底に対応する双極子行列クラスのインスタンス

    Raises
    ------
    TypeError
        未知の基底クラスが渡された場合
    """
    if isinstance(basis, SymTopBasis):
        return SymTopDipoleMatrix(
            basis=basis,
            mu0=mu0,
            potential_type=potential_type,
            backend=backend,
            dense=dense,
            units=units,
            units_input=units_input,
        )
    else:
        raise TypeError(
            f"未知の基底クラス: {type(basis).__name__}\n"
            "サポートされている基底クラス:\n"
            "- SymTopBasis（対称コマ分子）"
        )
