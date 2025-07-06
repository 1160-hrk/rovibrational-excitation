"""
rovibrational_excitation/core/propagator.py
------------------------------------------
* axes="xy"  → Ex ↔ μ_x,  Ey ↔ μ_y   (デフォルト)
* axes="zx"  → Ex ↔ μ_z,  Ey ↔ μ_x
"""

from __future__ import annotations

from collections.abc import Iterable, Sized
from typing import TYPE_CHECKING, Literal, Union, cast, Any, Protocol, TypeVar, overload

import numpy as np
import scipy.sparse

# ---------------------------------------------------------------------
# RK4 kernels
from ._rk4_lvne import rk4_lvne, rk4_lvne_traj
from ._rk4_schrodinger import rk4_schrodinger  # type: ignore
from ._splitop_schrodinger import splitop_schrodinger  # type: ignore
from .nondimensionalize import (
    get_physical_time as _get_physical_time,
    NondimensionalizationScales as _BaseScales,
)
from .nondimensional.scales import NondimensionalizationScales

# ---------------------------------------------------------------------
# 単位検証と無次元化
from .units.validators import validator
from .nondimensional.converter import (
    auto_nondimensionalize,
    nondimensionalize_from_objects,
)

# ---------------------------------------------------------------------
# optional CuPy
try:
    import cupy as _cp  # type: ignore
    from cupy.typing import NDArray as CupyArray  # type: ignore
except ImportError:
    _cp = None  # type: ignore
    CupyArray = Any  # type: ignore

# ---------------------------------------------------------------------
# type-hints
if TYPE_CHECKING:
    from numpy.typing import NDArray
    from typing import TypeVar
    from scipy import sparse
    _T = TypeVar("_T")
    _DType = TypeVar("_DType", bound=np.dtype[Any])
    _Shape = TypeVar("_Shape")

    class ArrayProtocol(Protocol[_DType]):
        """Protocol for array-like objects that can be used in numerical computations."""
        
        @property
        def shape(self) -> tuple[int, ...]: ...
        
        @property
        def dtype(self) -> _DType: ...
        
        def __len__(self) -> int: ...
        def __array__(self) -> NDArray[np.dtype[Any]]: ...  # type: ignore
        
        # Basic arithmetic operations
        def __add__(self, other: Union[float, int, "ArrayProtocol[_DType]"]) -> "ArrayProtocol[_DType]": ...
        def __sub__(self, other: Union[float, int, "ArrayProtocol[_DType]"]) -> "ArrayProtocol[_DType]": ...
        def __mul__(self, other: Union[float, int, "ArrayProtocol[_DType]"]) -> "ArrayProtocol[_DType]": ...
        def __truediv__(self, other: Union[float, int, "ArrayProtocol[_DType]"]) -> "ArrayProtocol[_DType]": ...
        def __floordiv__(self, other: Union[float, int, "ArrayProtocol[_DType]"]) -> "ArrayProtocol[_DType]": ...
        def __matmul__(self, other: "ArrayProtocol[_DType]") -> "ArrayProtocol[_DType]": ...
        
        # Reverse arithmetic operations
        def __radd__(self, other: Union[float, int]) -> "ArrayProtocol[_DType]": ...
        def __rsub__(self, other: Union[float, int]) -> "ArrayProtocol[_DType]": ...
        def __rmul__(self, other: Union[float, int]) -> "ArrayProtocol[_DType]": ...
        def __rtruediv__(self, other: Union[float, int]) -> "ArrayProtocol[_DType]": ...
        def __rfloordiv__(self, other: Union[float, int]) -> "ArrayProtocol[_DType]": ...
        def __rmatmul__(self, other: "ArrayProtocol[_DType]") -> "ArrayProtocol[_DType]": ...
        
        # Unary operations
        def __neg__(self) -> "ArrayProtocol[_DType]": ...
        def __pos__(self) -> "ArrayProtocol[_DType]": ...
        
        # Array interface
        def __getitem__(self, key: Union[int, slice, tuple[Union[int, slice], ...], NDArray[np.bool_]]) -> Union["ArrayProtocol[_DType]", Any]: ...  # type: ignore
        def __setitem__(self, key: Union[int, slice, tuple[Union[int, slice], ...], NDArray[np.bool_]], value: Union["ArrayProtocol[_DType]", Any]) -> None: ...  # type: ignore
        
        # Complex operations
        def conj(self) -> "ArrayProtocol[_DType]": ...
        
        @property
        def T(self) -> "ArrayProtocol[_DType]": ...
        
        # NumPy array interface
        def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any: ...
        def __array_function__(self, func: Any, types: Any, args: Any, kwargs: Any) -> Any: ...
        def __array_interface__(self) -> dict[str, Any]: ...
        def __array_struct__(self) -> Any: ...
        def __array_wrap__(self, array: Any) -> Any: ...
        def __array_prepare__(self, array: Any, context: Any = None) -> Any: ...
        def __array_priority__(self) -> float: ...
        def __array_finalize__(self, obj: Any) -> None: ...

    # Define Array type to include sparse matrices and ensure it implements Sized
    Array = Union[NDArray[Any], CupyArray, ArrayProtocol[Any], sparse.spmatrix]  # type: ignore

    from rovibrational_excitation.core.electric_field import ElectricField
    from rovibrational_excitation.core.basis.hamiltonian import Hamiltonian
    from rovibrational_excitation.dipole.base import DipoleMatrixBase
else:
    Array = np.ndarray  # runtime dummy

# ---------------------------------------------------------------------
# constants / helpers
_DIRAC_HBAR = 6.62607015e-019 / (2 * np.pi)  # J fs


def validate_propagation_units(
    H0, 
    dipole_matrix, 
    efield, 
    expected_H0_units="J", 
    expected_dipole_units="C*m"
):
    """
    Validate units before propagation and warn about potential issues.
    
    Parameters
    ----------
    H0 : Array
        Hamiltonian matrix
    dipole_matrix : LinMolDipoleMatrix
        Dipole moment matrices
    efield : ElectricField
        Electric field object
    expected_H0_units : str, default "J"
        Expected units for Hamiltonian ("J", "eV", "rad/fs", "cm^-1")
    expected_dipole_units : str, default "C*m"
        Expected units for dipole moments ("C*m", "D", "ea0")
        
    Returns
    -------
    list
        List of warning messages (empty if no issues)
        
    Examples
    --------
    >>> warnings = validate_propagation_units(H0, dipole_matrix, efield, "eV", "D")
    >>> if warnings:
    ...     for w in warnings:
    ...         print(f"⚠️  {w}")
    """
    return validator.validate_propagation_units(
        H0, dipole_matrix, efield, expected_H0_units, expected_dipole_units
    )


def _cm_to_rad_phz(mu: Array) -> Array:
    """μ (C·m) → rad / (PHz/(V·m⁻¹))."""
    return mu / (_DIRAC_HBAR)  # type: ignore # divide once; xp 対応は呼び出し側で

def _J_to_rad_phz(H0: Array) -> Array:
    """J → rad / fs."""
    return H0 / _DIRAC_HBAR  # type: ignore

def _backend(name: str):
    if name == "cupy":
        if _cp is None:
            raise RuntimeError("CuPy backend requested but CuPy not installed")
        return _cp
    return np


def _pick_mu_SI(dip, axis: str) -> Array:
    """
    Get dipole matrix in SI units (C·m) for specified axis.
    
    Parameters
    ----------
    dip : LinMolDipoleMatrix | TwoLevelDipoleMatrix | VibLadderDipoleMatrix
        Dipole matrix object with unit management
    axis : str
        Dipole axis ('x', 'y', 'z')
        
    Returns
    -------
    Array
        Dipole matrix in SI units (C·m)
    """
    # Try to get SI units first (preferred method)
    si_method = f"get_mu_{axis}_SI"
    if hasattr(dip, si_method):
        return getattr(dip, si_method)()
    
    # Fallback to direct attribute access (for backward compatibility)
    attr = f"mu_{axis}"
    if not hasattr(dip, attr):
        raise AttributeError(f"{type(dip).__name__} has no attribute '{attr}' or '{si_method}'")
    return getattr(dip, attr)


# ---------------------------------------------------------------------
def _prepare_args(
    hamiltonian: Hamiltonian,
    E: ElectricField,
    dip: DipoleMatrixBase,  # Type hint for better clarity
    *,
    axes: str = "xy",
    dt: float | None = None,
    mu_x_override: Array | None = None,
    mu_y_override: Array | None = None,
    nondimensional: bool = False,
    auto_timestep: bool = False,
) -> tuple[Array, Array, Array, Array, Array, float]:
    """
    共通前処理

    Returns (順序は旧バージョンと互換):
        H0, μ_a, μ_b, Ex, Ey, dt, steps
        └─ μ_a: Ex に対応 / μ_b: Ey に対応
    """
    axes = axes.lower()
    if len(axes) != 2 or any(a not in "xyz" for a in axes):
        raise ValueError("axes must be like 'xy', 'zx', ...")

    ax0, ax1 = axes[0], axes[1]
    xp = _cp if _cp is not None else np

    if nondimensional:
        if auto_timestep:
            # 完全自動無次元化（最適時間ステップ自動選択）
            (
                H0_prime,
                mu_x_prime,
                mu_y_prime,
                mu_z_prime,
                Efield_prime,
                tlist_prime,
                dt_prime,
                scales,
            ) = auto_nondimensionalize(
                hamiltonian,
                dip,
                E,
                target_accuracy="standard",
                verbose=False,
            )
        else:
            # オブジェクトベースの無次元化
            (
                H0_prime,
                mu_x_prime,
                mu_y_prime,
                mu_z_prime,
                Efield_prime,
                tlist_prime,
                dt_prime,
                scales,
            ) = nondimensionalize_from_objects(
                hamiltonian,
                dip,
                E,
                verbose=False,
            )
        Ex, Ey = _get_field_components(Efield_prime)
        dt = dt_prime * 2
        steps = (len(cast(Sized, Ex)) - 1) // 2  # type: ignore


    dt = E.dt * 2 if dt is None else dt

    # 電場成分を取得
    Ex, Ey = _get_field_components(E)

    # スパース行列対応: スパース行列の場合はそのまま使用
    H0 = hamiltonian.get_matrix("J")
    if scipy.sparse.issparse(H0):  # type: ignore
        H0_dense = H0.toarray()  # type: ignore
    else:
        H0_dense = H0

    # μ_a, μ_b を取得
    if mu_x_override is not None:
        mu_a = mu_x_override
    else:
        mu_a = _pick_mu_SI(dip, ax0)

    if mu_y_override is not None:
        mu_b = mu_y_override
    else:
        mu_b = _pick_mu_SI(dip, ax1)

    # スパース行列対応
    if scipy.sparse.issparse(mu_a):  # type: ignore
        mu_a = mu_a.toarray()  # type: ignore
    if scipy.sparse.issparse(mu_b):  # type: ignore
        mu_b = mu_b.toarray()  # type: ignore

    # 無次元化
    H0_prime = _J_to_rad_phz(H0_dense)
    mu_a_prime = _cm_to_rad_phz(mu_a)
    mu_b_prime = _cm_to_rad_phz(mu_b)

    return H0_prime, mu_a_prime, mu_b_prime, Ex, Ey, dt


# ---------------------------------------------------------------------
def schrodinger_propagation(
    hamiltonian,  # Type: Hamiltonian
    Efield: ElectricField,
    dipole_matrix,  # Type: LinMolDipoleMatrix | TwoLevelDipoleMatrix | VibLadderDipoleMatrix
    psi0: Array,
    *,
    axes: str = "xy",
    return_traj: bool = True,
    return_time_psi: bool = False,
    sample_stride: int = 1,
    backend: str = "numpy",
    sparse: bool = False,
    nondimensional: bool = False,
    validate_units: bool = True,
    verbose: bool = False,
    renorm: bool = False,
    auto_timestep: bool = False,
    target_accuracy: str = "standard",
) -> Array:
    """
    Time-dependent Schrödinger equation propagator with unit-aware physics objects.
    
    Enhanced version that uses Hamiltonian and LinMolDipoleMatrix objects with
    internal unit management for robust physics calculations.
    
    Parameters
    ----------
    hamiltonian : Hamiltonian
        Hamiltonian object with internal unit management
    Efield : ElectricField
        Electric field object  
    dipole_matrix : LinMolDipoleMatrix | TwoLevelDipoleMatrix | VibLadderDipoleMatrix
        Dipole moment matrices with internal unit management
    psi0 : Array
        Initial wavefunction
    axes : str, default "xy"
        Polarization axes mapping ("xy", "zx", etc.)
    return_traj : bool, default True
        Return full trajectory vs final state only
    return_time_psi : bool, default False
        Return time array along with trajectory
    sample_stride : int, default 1
        Sampling stride for trajectory
    backend : str, default "numpy" 
        Computational backend ("numpy" or "cupy")
    nondimensional : bool, default False
        Use nondimensional propagation
    validate_units : bool, default True
        Perform unit validation before propagation
    verbose : bool, default False
        Print detailed information
    renorm : bool, default False
        Renormalize wavefunction during propagation
    auto_timestep : bool, default False
        Automatically select optimal timestep
    target_accuracy : str, default "standard"
        Target accuracy for auto timestep ("high", "standard", "fast")
        
    Returns
    -------
    Array or tuple
        Propagated wavefunction(s), optionally with time array
        
    Notes
    -----
    This function now uses unit-aware physics objects that handle unit
    conversion internally. All physics quantities automatically use
    SI units (J, C·m, V/m) for calculation consistency.
    
    Examples
    --------
    >>> # Using Hamiltonian object
    >>> h = Hamiltonian(H0_matrix, "eV")  # Created in eV
    >>> dipole = LinMolDipoleMatrix(basis, mu0=0.3, units="D")  # In Debye
    >>> result = schrodinger_propagation(h, efield, dipole, psi0)
    
    >>> # Objects handle unit conversion automatically
    >>> h_matrix_J = h.get_matrix("J")  # Automatic eV → J conversion
    >>> mu_x_SI = dipole.get_mu_x_SI()  # Automatic D → C·m conversion
    """
    
    # Unit validation using SI quantities
    if validate_units:
        warnings = validate_propagation_units(
            hamiltonian, dipole_matrix, Efield
        )
        if warnings and verbose:
            for w in warnings:
                print(f"⚠️  {w}")

    # バックエンドを設定
    xp = _backend(backend)

    # 無次元化
    H0, mu_x, mu_y, Ex, Ey, dt = _prepare_args(
        hamiltonian,
        Efield,
        dipole_matrix,
        axes=axes,
        mu_x_override=None,
        mu_y_override=None,
        nondimensional=nondimensional,
    )

    # スパース行列対応
    if sparse:
        if not scipy.sparse.issparse(H0):  # type: ignore
            H0 = scipy.sparse.csr_matrix(H0)  # type: ignore
        if not scipy.sparse.issparse(mu_x):  # type: ignore
            mu_x = scipy.sparse.csr_matrix(mu_x)  # type: ignore
        if not scipy.sparse.issparse(mu_y):  # type: ignore
            mu_y = scipy.sparse.csr_matrix(mu_y)  # type: ignore
        
        # 通常の行列用のプロパゲータを使用
    result = rk4_schrodinger(  # type: ignore
            H0, mu_x, mu_y, Ex, Ey, psi0, dt,
            stride=sample_stride, renorm=renorm, sparse=sparse, backend=backend,
        )
    if return_traj:
        psi = result
    else:
        psi = result[-1]

    if return_time_psi:
        t = np.arange(0, len(cast(Sized, psi)), dtype=np.float64) * dt * sample_stride  # type: ignore
        return t, psi
    return psi


# ---------------------------------------------------------------------
def mixed_state_propagation(
    H0: Array,
    Efield: ElectricField,
    psi0_array: Iterable[Array],
    dipole_matrix,  # Type: LinMolDipoleMatrix | TwoLevelDipoleMatrix | VibLadderDipoleMatrix
    *,
    axes: str = "xy",
    return_traj: bool = True,
    return_time_rho: bool = False,
    sample_stride: int = 1,
    backend: str = "numpy",
) -> Array:
    xp = _backend(backend)

    # Iterableをリストに変換してインデックスアクセス可能にする
    psi0_list = list(psi0_array)
    dim = psi0_list[0].shape[0]

    steps_out = (len(Efield.tlist) // 2) // sample_stride + 1
    rho_out = (
        xp.zeros((steps_out, dim, dim), dtype=xp.complex128)
        if return_traj
        else xp.zeros((dim, dim), dtype=xp.complex128)
    )

    for psi0 in psi0_list:
        result = schrodinger_propagation(
            H0,
            Efield,
            dipole_matrix,
            psi0,
            axes=axes,
            return_traj=return_traj,
            return_time_psi=False,  # time情報は不要
            sample_stride=sample_stride,
            backend=backend,
        )

        # resultがtupleの場合の処理
        if isinstance(result, tuple):
            psi_t = result[1]
        else:
            psi_t = result

        if return_traj:
            rho_out += xp.einsum("ti, tj -> tij", psi_t, psi_t.conj())
        else:
            rho_out += psi_t[0] @ psi_t[0].conj().T

    if return_traj:
        if return_time_rho:
            dt_rho = Efield.dt_state * sample_stride
            steps = Efield.steps_state
            time_psi = xp.arange(0, (steps + 1) * dt_rho, dt_rho)
            return time_psi, rho_out
        else:
            return rho_out
    else:
        return rho_out


# ---------------------------------------------------------------------
def liouville_propagation(
    H0: Array,
    Efield: ElectricField,
    dipole_matrix,  # Type: LinMolDipoleMatrix | TwoLevelDipoleMatrix | VibLadderDipoleMatrix
    rho0: Array,
    *,
    axes: str = "xy",
    return_traj: bool = True,
    sample_stride: int = 1,
    backend: str = "numpy",
) -> Array:
    xp = _backend(backend)
    H0_, mu_a, mu_b, Ex, Ey, dt, steps = _prepare_args(
        H0, Efield, dipole_matrix, axes=axes
    )

    # 引数の数を修正 - rk4_lvne_trajとrk4_lvneで異なる引数を正しく渡す
    rk4_args = (H0_, mu_a, mu_b, Ex, Ey, xp.asarray(rho0), dt, steps)

    if return_traj:
        return rk4_lvne_traj(*rk4_args, sample_stride)
    else:
        return rk4_lvne(*rk4_args)


def _get_field_components(efield: ElectricField) -> tuple[Array, Array]:
    """Get x and y components of the electric field."""
    field = efield.get_Efield()
    return field[:, 0], field[:, 1]  # type: ignore


def get_physical_time(t_prime: Array, scales: NondimensionalizationScales) -> Array:
    """Convert dimensionless time to physical time."""
    # 型変換を行って基底クラスに合わせる
    base_scales = cast(_BaseScales, scales)
    return _get_physical_time(t_prime, base_scales)  # type: ignore
