"""
rovibrational_excitation/core/propagator.py
------------------------------------------
* axes="xy"  → Ex ↔ μ_x,  Ey ↔ μ_y   (デフォルト)
* axes="zx"  → Ex ↔ μ_z,  Ey ↔ μ_x
"""

from __future__ import annotations

from collections.abc import Iterable, Sized
from typing import TYPE_CHECKING, Literal, Union, cast, Any, Protocol, TypeVar, overload, Callable

import numpy as np
import scipy.sparse

# ---------------------------------------------------------------------

from .propagation.schrodinger import SchrodingerPropagator
from .propagation.mixed_state import MixedStatePropagator
from .propagation.liouville import LiouvillePropagator

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
# Legacy wrapper functions for backward compatibility
# These functions now delegate to the new propagation module classes
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
    algorithm: str = "rk4",
    propagator_func: Union[None, Callable] = None,
) -> Array:
    """
    Time-dependent Schrödinger equation propagator with unit-aware physics objects.
    
    This is a legacy wrapper function for backward compatibility.
    New code should use the SchrodingerPropagator class from the propagation module.
    
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
    sparse : bool, default False
        Use sparse matrix operations
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
    algorithm : str, default "rk4"
        Algorithm to use ("rk4" or "split_operator")
        
    Returns
    -------
    Array or tuple
        Propagated wavefunction(s), optionally with time array
        
    See Also
    --------
    propagation.SchrodingerPropagator : New class-based implementation
    """
    # Create propagator instance
    propagator = SchrodingerPropagator(
        backend=backend,  # type: ignore
        validate_units=validate_units,
        renorm=renorm,
    )
    
    # Call propagate method
    return propagator.propagate(
        hamiltonian,
        Efield,
        dipole_matrix,
        initial_state=psi0,
        axes=axes,
        return_traj=return_traj,
        return_time_psi=return_time_psi,
        sample_stride=sample_stride,
        nondimensional=nondimensional,
        auto_timestep=auto_timestep,
        target_accuracy=target_accuracy,
        verbose=verbose,
        sparse=sparse,
        algorithm=algorithm,
        propagator_func=propagator_func,
    )


# ---------------------------------------------------------------------
def mixed_state_propagation(
    hamiltonian,  # Type: Hamiltonian
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
    """
    Mixed state propagation (legacy wrapper).
    
    This is a legacy wrapper function for backward compatibility.
    New code should use the MixedStatePropagator class from the propagation module.
    
    See Also
    --------
    propagation.MixedStatePropagator : New class-based implementation
    """
    
    # Create propagator instance
    propagator = MixedStatePropagator(
        backend=backend,  # type: ignore
        validate_units=True,
    )
    
    # Call propagate method
    return propagator.propagate(
        hamiltonian,
        Efield,
        dipole_matrix,
        psi0_array,
        axes=axes,
        return_traj=return_traj,
        return_time_rho=return_time_rho,
        sample_stride=sample_stride,
    )


# ---------------------------------------------------------------------
def liouville_propagation(
    hamiltonian,  # Type: Hamiltonian
    Efield: ElectricField,
    dipole_matrix,  # Type: LinMolDipoleMatrix | TwoLevelDipoleMatrix | VibLadderDipoleMatrix
    rho0: Array,
    *,
    axes: str = "xy",
    return_traj: bool = True,
    sample_stride: int = 1,
    backend: str = "numpy",
    nondimensional: bool = False,
    auto_timestep: bool = False,
) -> Array:
    """
    Liouville-von Neumann equation propagation (legacy wrapper).
    
    This is a legacy wrapper function for backward compatibility.
    New code should use the LiouvillePropagator class from the propagation module.
    
    See Also
    --------
    propagation.LiouvillePropagator : New class-based implementation
    """
    
    # Create propagator instance
    propagator = LiouvillePropagator(
        backend=backend,  # type: ignore
        validate_units=True,
    )
    
    # Call propagate method
    return propagator.propagate(
        hamiltonian,
        Efield,
        dipole_matrix,
        rho0,
        axes=axes,
        return_traj=return_traj,
        sample_stride=sample_stride,
        nondimensional=nondimensional,
        auto_timestep=auto_timestep,
    )