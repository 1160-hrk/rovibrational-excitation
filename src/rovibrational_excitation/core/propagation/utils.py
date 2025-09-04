"""
Utility functions for time propagation algorithms.

This module contains common utilities used by various propagator implementations.
"""

from typing import Optional, Tuple, Union, Any, TYPE_CHECKING
import numpy as np
import scipy.sparse
from ..basis.hamiltonian import Hamiltonian
from ..electric_field import ElectricField
from ...dipole.base import DipoleMatrixBase

if TYPE_CHECKING:
    from numpy.typing import NDArray
    Array = Union[NDArray[Any], Any]
else:
    Array = np.ndarray

# Physical constants
DIRAC_HBAR = 6.62607015e-019 / (2 * np.pi)  # J fs

# Optional CuPy support
try:
    import cupy as cp  # type: ignore
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False


def get_backend(name: str):
    """
    Get computational backend by name.
    
    Parameters
    ----------
    name : str
        Backend name ("numpy" or "cupy")
        
    Returns
    -------
    module
        NumPy or CuPy module
        
    Raises
    ------
    RuntimeError
        If CuPy is requested but not installed
    """
    if name == "numpy":
        return np
    elif name == "cupy":
        if cp is None:
            raise RuntimeError("CuPy backend requested but CuPy not installed")
        return cp
    else:
        raise ValueError(f"Unknown backend: {name}")


def cm_to_rad_phz(mu: Array) -> Array:
    """
    Convert dipole moment from C·m to rad / (PHz/(V·m⁻¹)).
    
    Parameters
    ----------
    mu : Array
        Dipole moment in C·m
        
    Returns
    -------
    Array
        Dipole moment in rad / (PHz/(V·m⁻¹))
    """
    return mu / DIRAC_HBAR


def J_to_rad_phz(H0: Array) -> Array:
    """
    Convert energy from J to rad / fs.
    
    Parameters
    ----------
    H0 : Array
        Energy in Joules
        
    Returns
    -------
    Array
        Energy in rad / fs
    """
    return H0 / DIRAC_HBAR


def get_dipole_component_SI(dipole_matrix, axis: str) -> Array:
    """
    Get dipole matrix component in SI units (C·m) for specified axis.
    
    Parameters
    ----------
    dipole_matrix : LinMolDipoleMatrix | TwoLevelDipoleMatrix | VibLadderDipoleMatrix
        Dipole matrix object with unit management
    axis : str
        Dipole axis ('x', 'y', 'z')
        
    Returns
    -------
    Array
        Dipole matrix in SI units (C·m)
        
    Raises
    ------
    AttributeError
        If the dipole matrix doesn't have the requested component
    """
    # Try to get SI units first (preferred method)
    si_method = f"get_mu_{axis}_SI"
    if hasattr(dipole_matrix, si_method):
        return getattr(dipole_matrix, si_method)()
    
    # Fallback to direct attribute access (for backward compatibility)
    attr = f"mu_{axis}"
    if not hasattr(dipole_matrix, attr):
        raise AttributeError(
            f"{type(dipole_matrix).__name__} has no attribute '{attr}' or '{si_method}'"
        )
    return getattr(dipole_matrix, attr)


def get_field_components(efield) -> Tuple[Array, Array]:
    """
    Get x and y components of the electric field.
    
    Parameters
    ----------
    efield : ElectricField
        Electric field object
        
    Returns
    -------
    tuple
        (Ex, Ey) field components
    """
    field = efield.get_Efield()
    return field[:, 0], field[:, 1]


def ensure_dense_matrix(matrix: Array) -> Array:
    """
    Convert sparse matrix to dense if necessary.
    
    Parameters
    ----------
    matrix : Array or sparse matrix
        Input matrix
        
    Returns
    -------
    Array
        Dense matrix
    """
    if scipy.sparse.issparse(matrix):
        return matrix.toarray()  # type: ignore
    return matrix


def ensure_sparse_matrix(matrix: Array) -> scipy.sparse.csr_matrix:
    """
    Convert dense matrix to sparse CSR format if necessary.
    
    Parameters
    ----------
    matrix : Array or sparse matrix
        Input matrix
        
    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse CSR matrix
    """
    if not scipy.sparse.issparse(matrix):
        return scipy.sparse.csr_matrix(matrix)
    return matrix.tocsr()  # type: ignore


def validate_axes(axes: str) -> Tuple[str, str]:
    """
    Validate and parse axes string.
    
    Parameters
    ----------
    axes : str
        Axes string (e.g., "xy", "zx")
        
    Returns
    -------
    tuple
        (axis0, axis1) individual axes
        
    Raises
    ------
    ValueError
        If axes string is invalid
    """
    axes = axes.lower()
    if len(axes) != 2 or any(a not in "xyz" for a in axes):
        raise ValueError("axes must be like 'xy', 'zx', ...")
    return axes[0], axes[1]


def prepare_propagation_args(
    hamiltonian: Hamiltonian,
    efield: ElectricField,
    dipole_matrix: DipoleMatrixBase,
    *,
    axes: str = "xy",
    mu_x_override: Optional[Array] = None,
    mu_y_override: Optional[Array] = None,
    nondimensional: bool = False,
    auto_timestep: bool = False,
) -> Tuple[Array, Array, Array, Array, Array, Array, Array, float, float]:
    """
    Prepare arguments for propagation algorithms.
    
    Parameters
    ----------
    hamiltonian : Hamiltonian
        Hamiltonian object
    efield : ElectricField
        Electric field object
    dipole_matrix : DipoleMatrixBase
        Dipole matrix object
    axes : str
        Polarization axes
    dt : float, optional
        Time step (if None, use field dt * 2)
    mu_x_override : Array, optional
        Override for x-component dipole
    mu_y_override : Array, optional
        Override for y-component dipole
    nondimensional : bool
        Use nondimensional units
    auto_timestep : bool
        Automatically select timestep
        
    Returns
    -------
    tuple
        (H0, mu_a, mu_b, Ex, Ey, dt) prepared for propagation
        mu_a corresponds to Ex, mu_b corresponds to Ey
    """
    from ..nondimensional.converter import (
        auto_nondimensionalize,
        nondimensionalize_from_objects,
    )
    
    ax0, ax1 = validate_axes(axes)
    try:
        pol = efield.get_pol()
    except ValueError:
        pol = np.zeros((2,))
        
    if nondimensional:
        if auto_timestep:
            # Complete auto nondimensionalization
            (
                H0_prime,
                mu_x_prime,
                mu_y_prime,
                mu_z_prime,
                Efield_prime,
                E_scalar,
                tlist_prime,
                dt_prime,
                scales,
            ) = auto_nondimensionalize(
                hamiltonian,
                dipole_matrix,
                efield,
                target_accuracy="standard",
                verbose=False,
            )
        else:
            # Object-based nondimensionalization
            (
                H0_prime,
                mu_x_prime,
                mu_y_prime,
                mu_z_prime,
                Efield_prime,
                E_scalar,
                tlist_prime,
                dt_prime,
                scales,
            ) = nondimensionalize_from_objects(
                hamiltonian,
                dipole_matrix,
                efield,
                verbose=False,
            )
        Ex, Ey = Efield_prime[:, 0], Efield_prime[:, 1]
        dt = dt_prime * 2
        
        # Map dipole components based on axes
        dipole_map = {'x': mu_x_prime, 'y': mu_y_prime, 'z': mu_z_prime}
        mu_a = dipole_map[ax0]
        mu_b = dipole_map[ax1]
        mu_a *= scales.lambda_coupling
        mu_b *= scales.lambda_coupling
        return H0_prime, mu_a, mu_b, Ex, Ey, pol, E_scalar, dt, scales.t0
    
    # Standard dimensional calculation
    dt = efield.dt * 2
    
    # Get field components
    Ex, Ey = get_field_components(efield)
    try:
        E_scalar = efield.get_scalar_and_pol()[0]
    except ValueError:
        E_scalar = np.zeros_like(Ex)
        
    # Get dipole components
    mu_a_prime = None
    mu_b_prime = None
    
    if mu_x_override is not None:
        mu_a_prime = mu_x_override
    else:
        # mu_a = get_dipole_component_SI(dipole_matrix, ax0)
        mu_a_prime = dipole_matrix.get_mu_in_units(ax0, "rad/fs/(V/m)")
    
    if mu_y_override is not None:
        mu_b_prime = mu_y_override
    else:
        # mu_b = get_dipole_component_SI(dipole_matrix, ax1)
        mu_b_prime = dipole_matrix.get_mu_in_units(ax1, "rad/fs/(V/m)")
    
    # Ensure dense matrices
    mu_a_prime = ensure_dense_matrix(mu_a_prime)
    mu_b_prime = ensure_dense_matrix(mu_b_prime)
    
    # Convert to appropriate units
    H0_prime = ensure_dense_matrix(hamiltonian.get_matrix("rad/fs"))
    # mu_a_prime = cm_to_rad_phz(mu_a)
    # mu_b_prime = cm_to_rad_phz(mu_b)
    
    return H0_prime, mu_a_prime, mu_b_prime, Ex, Ey, pol, E_scalar, dt, 1.0


def is_sparse_matrix(dipole_matrix, threshold: float = 0.1) -> bool:
    """
    Check if dipole matrices are sparse.
    
    Parameters
    ----------
    dipole_matrix : DipoleMatrixBase
        Dipole moment matrices to check
    threshold : float
        Sparsity threshold (fraction of non-zero elements)
        
    Returns
    -------
    bool
        True if matrices are considered sparse
    """
    # Get matrices for all axes
    matrices = [
        dipole_matrix.get_matrix(axis)
        for axis in dipole_matrix.available_axes
    ]
    
    # Calculate sparsity for each matrix
    sparsities = []
    for mat in matrices:
        total_elements = mat.size
        nonzero_elements = np.count_nonzero(mat)
        sparsity = nonzero_elements / total_elements
        sparsities.append(sparsity)
    
    # Return True if any matrix is sparse
    return any(sparsity < threshold for sparsity in sparsities) 