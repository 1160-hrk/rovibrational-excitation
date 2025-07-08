"""
Schrödinger equation propagator implementation.

This module provides the SchrodingerPropagator class for time-dependent
Schrödinger equation propagation.
"""

from typing import Optional, Dict, Any, Literal, Union, cast
import numpy as np
from collections.abc import Sized

from .base import PropagatorBase
from .utils import (
    get_backend,
    prepare_propagation_args,
    ensure_sparse_matrix,
    HAS_CUPY,
)
from ..units.validators import validator


class SchrodingerPropagator(PropagatorBase):
    """
    Time-dependent Schrödinger equation propagator.
    
    This class implements various algorithms for solving the time-dependent
    Schrödinger equation with external fields.
    """
    
    def __init__(
        self,
        backend: Literal["numpy", "cupy"] = "numpy",
        validate_units: bool = True,
        renorm: bool = False,
    ):
        """
        Initialize Schrödinger propagator.
        
        Parameters
        ----------
        algorithm : {"rk4", "split_operator"}
            Propagation algorithm to use
        backend : {"numpy", "cupy"}
            Computational backend
        sparse : bool
            Use sparse matrix operations
        validate_units : bool
            Whether to validate physical units
        renorm : bool
            Renormalize wavefunction during propagation
        """
        super().__init__(validate_units)
        self.backend = backend
        self.renorm = renorm
        
        # Validate backend availability
        if backend == "cupy" and not HAS_CUPY:
            raise RuntimeError("CuPy backend requested but CuPy not installed")
    
    def get_algorithm_name(self) -> str:
        """Get the name of the propagation algorithm."""
        return f"Schrödinger-{self.algorithm}"
    
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
        *,
        axes: str = "xy",
        return_traj: bool = True,
        return_time_psi: bool = False,
        sample_stride: int = 1,
        nondimensional: bool = False,
        auto_timestep: bool = False,
        target_accuracy: str = "standard",
        verbose: bool = False,
        algorithm: Literal["rk4", "split_operator"] = "rk4",
        sparse: bool = False,
    ) -> Union[np.ndarray, tuple]:
        """
        Propagate wavefunction using time-dependent Schrödinger equation.
        
        Parameters
        ----------
        hamiltonian : Hamiltonian
            Hamiltonian object with internal unit management
        efield : ElectricField
            Electric field object
        dipole_matrix : DipoleMatrixBase
            Dipole moment matrices with internal unit management
        psi0 : np.ndarray
            Initial wavefunction
        axes : str
            Polarization axes mapping ("xy", "zx", etc.)
        return_traj : bool
            Return full trajectory vs final state only
        return_time_psi : bool
            Return time array along with trajectory
        sample_stride : int
            Sampling stride for trajectory
        nondimensional : bool
            Use nondimensional propagation
        auto_timestep : bool
            Automatically select optimal timestep
        target_accuracy : str
            Target accuracy for auto timestep
        verbose : bool
            Print detailed information
        dt : float, optional
            Override time step
            
        Returns
        -------
        np.ndarray or tuple
            Propagated wavefunction(s), optionally with time array
        """
        # Unit validation
        if self.validate_units:
            warnings = validator.validate_propagation_units(
                hamiltonian, dipole_matrix, efield
            )
            if warnings:
                self._last_validation_warnings = warnings
                if verbose:
                    self.print_validation_warnings()
        
        # Prepare arguments
        H0, mu_x, mu_y, Ex, Ey, pol, E_scalar, dt_calc = prepare_propagation_args(
            hamiltonian,
            efield,
            dipole_matrix,
            axes=axes,
            nondimensional=nondimensional,
            auto_timestep=auto_timestep,
        )
        
        # Handle sparse matrices if requested
        if sparse:
            H0 = ensure_sparse_matrix(H0)
            mu_x = ensure_sparse_matrix(mu_x)
            mu_y = ensure_sparse_matrix(mu_y)
        
        # Select and run algorithm
        if algorithm == "rk4":
            result = self._propagate_rk4(
                H0, mu_x, mu_y, Ex, Ey, initial_state, dt_calc,
                return_traj, sample_stride, sparse
            )
        elif algorithm == "split_operator":
            result = self._propagate_split_operator(
                H0, mu_x, mu_y, pol, E_scalar, initial_state, dt_calc,
                return_traj, sample_stride, sparse
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Handle return values
        if return_traj:
            psi = result
        else:
            psi = result[-1] if hasattr(result, '__len__') else result
        
        if return_time_psi:
            t = np.arange(0, len(cast(Sized, psi)), dtype=np.float64) * dt_calc * sample_stride
            return t, psi
        
        return psi
    
    def _propagate_rk4(
        self,
        H0: np.ndarray,
        mu_x: np.ndarray,
        mu_y: np.ndarray,
        Ex: np.ndarray,
        Ey: np.ndarray,
        initial_state: np.ndarray,
        dt: float,
        return_traj: bool,
        stride: int,
        sparse: bool,
    ) -> np.ndarray:
        """Run RK4 propagation algorithm."""
        from .algorithms.rk4.schrodinger import rk4_schrodinger
        
        return rk4_schrodinger(
            H0, mu_x, mu_y, Ex, Ey, initial_state, dt,
            return_traj=return_traj,
            stride=stride,
            renorm=self.renorm,
            sparse=sparse,
            backend=self.backend,
        )
    
    def _propagate_split_operator(
        self,
        H0: np.ndarray,
        mu_x: np.ndarray,
        mu_y: np.ndarray,
        pol: np.ndarray,
        E_scalar: np.ndarray,
        initial_state: np.ndarray,
        dt: float,
        return_traj: bool,
        stride: int,
        sparse: bool,
    ) -> np.ndarray:
        """Run split-operator propagation algorithm."""
        from .algorithms.split_operator.schrodinger import splitop_schrodinger
        
        return splitop_schrodinger(
            H0, mu_x, mu_y, pol, E_scalar, initial_state, dt,
            return_traj=return_traj,
            sample_stride=stride,
            backend=self.backend,
            sparse=sparse,
        ) 