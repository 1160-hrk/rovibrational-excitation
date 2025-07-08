"""
双極子行列クラスの自動選択機能を提供するファクトリーモジュール。

基底クラスの型に応じて適切な双極子行列クラスを自動的に選択します。
"""

from typing import Literal, Union, TYPE_CHECKING

from rovibrational_excitation.core.basis import (
    BasisBase,
    LinMolBasis,
    SymTopBasis,
    TwoLevelBasis,
    VibLadderBasis,
)
from rovibrational_excitation.dipole import (
    LinMolDipoleMatrix,
    SymTopDipoleMatrix,
    TwoLevelDipoleMatrix,
    VibLadderDipoleMatrix,
)

def create_dipole_matrix(
    basis: BasisBase,
    mu0: float = 1.0,
    *,
    potential_type: Literal["harmonic", "morse"] = "harmonic",
    backend: Literal["numpy", "cupy"] = "numpy",
    dense: bool = True,
    units: Literal["C*m", "D", "ea0"] = "C*m",
    units_input: Literal["C*m", "D", "ea0"] = "C*m",
) -> Union[LinMolDipoleMatrix, SymTopDipoleMatrix, TwoLevelDipoleMatrix, VibLadderDipoleMatrix]:
    """
    基底クラスの型に応じて適切な双極子行列クラスを自動的に選択し、インスタンスを生成します。

    Parameters
    ----------
    basis : BasisBase
        量子基底クラスのインスタンス
    mu0 : float, optional
        双極子モーメントの大きさ（units_inputで指定した単位）
    potential_type : {"harmonic", "morse"}, optional
        振動ポテンシャルの種類（振動を含む系のみ）
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
    Union[LinMolDipoleMatrix, SymTopDipoleMatrix, TwoLevelDipoleMatrix, VibLadderDipoleMatrix]
        基底に対応する双極子行列クラスのインスタンス

    Examples
    --------
    >>> # 二準位系の例
    >>> basis = TwoLevelBasis(energy_gap=2.35, input_units="eV")
    >>> dipole = create_dipole_matrix(basis, mu0=0.5, units_input="D")
    >>> print(dipole.mu_x)  # x方向の双極子行列を取得

    >>> # 線形分子の例
    >>> basis = LinMolBasis(V_max=2, J_max=10, omega=2350, input_units="cm^-1")
    >>> dipole = create_dipole_matrix(
    ...     basis, 
    ...     mu0=1.0, 
    ...     potential_type="morse",
    ...     backend="numpy",
    ...     dense=True
    ... )
    >>> print(dipole.mu_z)  # z方向の双極子行列を取得

    Raises
    ------
    TypeError
        未知の基底クラスが渡された場合
    """
    # 基底クラスの型に応じて適切な双極子行列クラスを選択
    if isinstance(basis, LinMolBasis):
        return LinMolDipoleMatrix(
            basis=basis,
            mu0=mu0,
            potential_type=potential_type,
            backend=backend,
            dense=dense,
            units=units,
            units_input=units_input,
        )
    elif isinstance(basis, SymTopBasis):
        return SymTopDipoleMatrix(
            basis=basis,
            mu0=mu0,
            potential_type=potential_type,
            backend=backend,
            dense=dense,
            units=units,
            units_input=units_input,
        )
    elif isinstance(basis, TwoLevelBasis):
        return TwoLevelDipoleMatrix(
            basis=basis,
            mu0=mu0,
            backend=backend,
            units=units,
            units_input=units_input,
        )
    elif isinstance(basis, VibLadderBasis):
        return VibLadderDipoleMatrix(
            basis=basis,
            mu0=mu0,
            potential_type=potential_type,
            backend=backend,
            units=units,
            units_input=units_input,
        )
    else:
        raise TypeError(
            f"未知の基底クラス: {type(basis).__name__}\n"
            "サポートされている基底クラス:\n"
            "- LinMolBasis（線形分子）\n"
            "- SymTopBasis（対称コマ分子）\n"
            "- TwoLevelBasis（二準位系）\n"
            "- VibLadderBasis（振動準位系）"
        ) 