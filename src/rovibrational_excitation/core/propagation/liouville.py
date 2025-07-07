"""
Liouville-von Neumann equation propagator implementation.

This module provides the LiouvillePropagator class for density matrix
propagation using the Liouville-von Neumann equation.
"""

from typing import Optional, Literal, Union
import numpy as np

from .base import PropagatorBase
from .utils import (
    get_backend,
    prepare_propagation_args,
    HAS_CUPY,
)
from ..units.validators import validator


class LiouvillePropagator(PropagatorBase):
    """
    Liouville-von Neumann equation propagator for density matrices.
    
    This class implements the propagation of density matrices using
    the Liouville-von Neumann equation.
    """
    
    def __init__(
        self,
        backend: Literal["numpy", "cupy"] = "numpy",
        validate_units: bool = True,
    ):
        """
        Initialize Liouville propagator.
        
        Parameters
        ----------
        backend : {"numpy", "cupy"}
            Computational backend
        validate_units : bool
            Whether to validate physical units
        """
        super().__init__(validate_units)
        self.backend = backend
        
        # Validate backend availability
        if backend == "cupy" and not HAS_CUPY:
            raise RuntimeError("CuPy backend requested but CuPy not installed")
    
    def get_algorithm_name(self) -> str:
        """Get the name of the propagation algorithm."""
        return "Liouville-von-Neumann"
    
    def get_supported_backends(self) -> list:
        """Get list of supported computational backends."""
        backends = ["numpy"]
        if HAS_CUPY:
            backends.append("cupy")
        return backends
    
    def propagate(
        self,
        hamiltonian,
        efield,
        dipole_matrix,
        initial_state: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        Propagate density matrix using Liouville-von Neumann equation.
        
        Parameters
        ----------
        hamiltonian : Hamiltonian
            Hamiltonian object with internal unit management
        efield : ElectricField
            Electric field object
        dipole_matrix : DipoleMatrixBase
            Dipole moment matrices with internal unit management
        initial_state : np.ndarray
            Initial density matrix
        **kwargs
            Additional parameters:
            - axes : str, default "xy"
                Polarization axes mapping
            - return_traj : bool, default True
                Return full trajectory vs final state only
            - sample_stride : int, default 1
                Sampling stride for trajectory
            - verbose : bool, default False
                Print detailed information
            - dt : float, optional
                Override time step
            
        Returns
        -------
        np.ndarray
            Propagated density matrix or trajectory
        """
        # Extract kwargs
        axes = kwargs.get('axes', 'xy')
        return_traj = kwargs.get('return_traj', True)
        sample_stride = kwargs.get('sample_stride', 1)
        verbose = kwargs.get('verbose', False)
        dt = kwargs.get('dt', None)
        
        # Use initial_state as rho0
        rho0 = initial_state
        
        # Unit validation
        if self.validate_units:
            warnings = validator.validate_propagation_units(
                hamiltonian, dipole_matrix, efield
            )
            if warnings:
                self._last_validation_warnings = warnings
                if verbose:
                    self.print_validation_warnings()
        
        # Get backend
        xp = get_backend(self.backend)
        
        # Prepare arguments
        H0, mu_a, mu_b, Ex, Ey, dt_calc = prepare_propagation_args(
            hamiltonian,
            efield,
            dipole_matrix,
            axes=axes,
            dt=dt,
            nondimensional=False,
            auto_timestep=False,
        )
        
        # Calculate number of steps
        steps = (len(Ex) - 1) // 2
        
        # Import and run RK4 Liouville propagator
        from .algorithms.rk4.lvne import rk4_lvne, rk4_lvne_traj
        
        # Prepare arguments for RK4
        rk4_args = (H0, mu_a, mu_b, Ex, Ey, xp.asarray(rho0), dt_calc, steps)
        
        if return_traj:
            return rk4_lvne_traj(*rk4_args, sample_stride)
        else:
            return rk4_lvne(*rk4_args) 