"""
Mixed state propagator implementation.

This module provides the MixedStatePropagator class for propagating
statistical mixtures of quantum states.
"""

from typing import Optional, Literal, Union, Iterable
import numpy as np

from .base import PropagatorBase
from .schrodinger import SchrodingerPropagator
from .utils import get_backend, HAS_CUPY
from ..units.validators import validator


class MixedStatePropagator(PropagatorBase):
    """
    Mixed state propagator for statistical ensembles.
    
    This class propagates a statistical mixture of quantum states by
    evolving each pure state component and constructing the density matrix.
    """
    
    def __init__(
        self,
        algorithm: Literal["rk4", "split_operator"] = "rk4",
        backend: Literal["numpy", "cupy"] = "numpy",
        sparse: bool = False,
        validate_units: bool = True,
        renorm: bool = False,
    ):
        """
        Initialize mixed state propagator.
        
        Parameters
        ----------
        algorithm : {"rk4", "split_operator"}
            Propagation algorithm for individual states
        backend : {"numpy", "cupy"}
            Computational backend
        sparse : bool
            Use sparse matrix operations
        validate_units : bool
            Whether to validate physical units
        """
        super().__init__(validate_units)
        self.algorithm = algorithm
        self.backend = backend
        self.sparse = sparse
        
        # Create underlying Schrödinger propagator
        self._schrodinger_prop = SchrodingerPropagator(
            backend=backend,
            validate_units=validate_units,
            renorm=renorm,
        )
    
    def get_algorithm_name(self) -> str:
        """Get the name of the propagation algorithm."""
        return f"MixedState-{self.algorithm}"
    
    def get_supported_backends(self) -> list:
        """Get list of supported computational backends."""
        return self._schrodinger_prop.get_supported_backends()
    
    def propagate(
        self,
        hamiltonian,
        efield,
        dipole_matrix,
        initial_state: Union[np.ndarray, Iterable[np.ndarray]],
        **kwargs,
    ) -> Union[np.ndarray, tuple]:
        """
        Propagate mixed state ensemble.
        
        Parameters
        ----------
        hamiltonian : Hamiltonian
            Hamiltonian object with internal unit management
        efield : ElectricField
            Electric field object
        dipole_matrix : DipoleMatrixBase
            Dipole moment matrices with internal unit management
        initial_state : array or iterable of arrays
            Initial state(s). Can be:
            - Single density matrix (will be returned as-is)
            - Iterable of pure states (will be evolved and summed)
        **kwargs
            Additional parameters:
            - axes : str, default "xy"
                Polarization axes mapping
            - return_traj : bool, default True
                Return full trajectory vs final state only
            - return_time_rho : bool, default False
                Return time array along with trajectory
            - sample_stride : int, default 1
                Sampling stride for trajectory
            - verbose : bool, default False
                Print detailed information
            
        Returns
        -------
        np.ndarray or tuple
            Propagated density matrix or trajectory, optionally with time
        """
        # Extract kwargs
        axes = kwargs.get('axes', 'xy')
        return_traj = kwargs.get('return_traj', True)
        return_time_rho = kwargs.get('return_time_rho', False)
        sample_stride = kwargs.get('sample_stride', 1)
        verbose = kwargs.get('verbose', False)
        
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
        
        # Check if input is already a density matrix
        if isinstance(initial_state, np.ndarray) and initial_state.ndim == 2:
            # Already a density matrix - use Liouville propagator
            from .liouville import LiouvillePropagator
            liouville_prop = LiouvillePropagator(
                backend=self.backend,  # type: ignore
                validate_units=False,
            )
            return liouville_prop.propagate(
                hamiltonian, efield, dipole_matrix, initial_state, **kwargs
            )
        
        # Convert iterable to list for indexing
        psi0_list = list(initial_state)
        dim = psi0_list[0].shape[0]
        
        # Calculate output dimensions
        steps_out = (len(efield.tlist) // 2) // sample_stride + 1
        rho_out = (
            xp.zeros((steps_out, dim, dim), dtype=xp.complex128)
            if return_traj
            else xp.zeros((dim, dim), dtype=xp.complex128)
        )
        
        # Initialize time_psi
        time_psi = None
        
        # Propagate each pure state
        for psi0 in psi0_list:
            result = self._schrodinger_prop.propagate(
                hamiltonian,
                efield,
                dipole_matrix,
                psi0,
                axes=axes,
                return_traj=return_traj,
                return_time_psi=return_time_rho,
                sample_stride=sample_stride,
                verbose=False,
                algorithm=self.algorithm,
            )
            
            # Handle result format
            if isinstance(result, tuple) or return_time_rho:
                psi_t = result[1]
                if return_time_rho:
                    time_psi = result[0]
            else:
                psi_t = result
            
            # Accumulate density matrix
            if return_traj:
                rho_out += xp.einsum("ti, tj -> tij", psi_t, psi_t.conj())
            else:
                rho_out += xp.outer(psi_t, psi_t.conj())
        
        # Return with time if requested
        if return_traj and return_time_rho and time_psi is not None:
            return time_psi, rho_out
        
        return rho_out 