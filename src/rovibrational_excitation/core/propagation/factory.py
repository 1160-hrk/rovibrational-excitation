"""
Propagator factory module.

This module provides the PropagatorFactory class for creating appropriate
propagator instances based on system characteristics and requirements.
"""

from typing import Optional, Literal, Union
import numpy as np

from .base import PropagatorBase
from .schrodinger import SchrodingerPropagator
from .liouville import LiouvillePropagator
from .mixed_state import MixedStatePropagator
from .utils import is_sparse_matrix


class PropagatorFactory:
    """
    Factory class for creating propagator instances.
    
    This class provides methods to create appropriate propagator instances
    based on system characteristics such as state type, polarization
    properties, and sparsity.
    """
    
    @staticmethod
    def create_propagator(
        state_type: Literal["pure", "density", "mixed"] = "pure",
        backend: Literal["numpy", "cupy"] = "numpy",
        algorithm: Optional[Literal["rk4", "split-operator"]] = None,
        const_polarization: bool = False,
        dipole_matrix = None,
        validate_units: bool = True,
    ) -> PropagatorBase:
        """
        Create appropriate propagator instance.
        
        Parameters
        ----------
        state_type : {"pure", "density", "mixed"}
            Type of quantum state to propagate
        backend : {"numpy", "cupy"}
            Computational backend
        algorithm : {"rk4", "split-operator"}, optional
            Explicitly specify propagation algorithm
        const_polarization : bool
            Whether polarization is constant during propagation
        dipole_matrix : DipoleMatrixBase, optional
            Dipole moment matrices for sparsity analysis
        validate_units : bool
            Whether to validate physical units
            
        Returns
        -------
        PropagatorBase
            Appropriate propagator instance
            
        Notes
        -----
        Algorithm selection logic:
        1. If algorithm is explicitly specified, use that
        2. For pure states with constant polarization, prefer split-operator
        3. For sparse systems, prefer RK4
        4. Default to RK4 for all other cases
        """
        # Handle explicit algorithm specification
        if algorithm is not None:
            if algorithm == "split-operator":
                if state_type != "pure":
                    raise ValueError(
                        "Split-operator method only supports pure states"
                    )
                return SchrodingerPropagator(
                    backend=backend,
                    validate_units=validate_units
                )
            elif algorithm == "rk4":
                if state_type == "pure":
                    return SchrodingerPropagator(
                        backend=backend,
                        validate_units=validate_units
                    )
                elif state_type == "density":
                    return LiouvillePropagator(
                        backend=backend,
                        validate_units=validate_units
                    )
                else:  # mixed
                    return MixedStatePropagator(
                        backend=backend,
                        validate_units=validate_units
                    )
        
        # Automatic algorithm selection
        if state_type == "pure":
            # For pure states, prefer split-operator if polarization is constant
            if const_polarization:
                return SchrodingerPropagator(
                    backend=backend,
                    validate_units=validate_units
                )
            # For sparse systems, use RK4
            elif dipole_matrix is not None and is_sparse_matrix(dipole_matrix):
                return SchrodingerPropagator(
                    backend=backend,
                    validate_units=validate_units
                )
            # Default to RK4 for general cases
            else:
                return SchrodingerPropagator(
                    backend=backend,
                    validate_units=validate_units
                )
        
        elif state_type == "density":
            return LiouvillePropagator(
                backend=backend,
                validate_units=validate_units
            )
        
        else:  # mixed
            return MixedStatePropagator(
                backend=backend,
                validate_units=validate_units
            ) 