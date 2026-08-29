"""
Base class for time propagation algorithms.

This module provides the abstract base class that all propagators
should inherit from.
"""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

import numpy as np

from .options import PropagationOptions
from .problem import PropagationProblem
from .result import PropagationResult

InitialStateT = TypeVar("InitialStateT")


class PropagatorBase(ABC, Generic[InitialStateT]):
    """
    Abstract base class for time propagation algorithms.

    This class defines the interface that all propagator implementations
    must follow.
    """

    def __init__(self, validate_units: bool = True):
        """
        Initialize propagator.

        Parameters
        ----------
        validate_units : bool
            Whether to validate physical units before propagation
        """
        self.validate_units = validate_units

    @abstractmethod
    def propagate(
        self,
        problem: PropagationProblem,
        *,
        options: PropagationOptions,
        verbose: bool = False,
    ) -> PropagationResult:
        """
        Propagate the quantum state forward in time.

        Parameters
        ----------
        problem : PropagationProblem
            Complete model, field, time grid, and typed initial state
        options
            Required typed computational policy.

        Returns
        -------
        PropagationResult
            Endpoint-complete typed result
        """
        pass

    @abstractmethod
    def get_algorithm_name(self) -> str:
        """Get the name of the propagation algorithm."""
        pass

    def get_supported_backends(self) -> list[str]:
        """Get list of supported computational backends."""
        return ["numpy"]

    def prepare_units(
        self, H0: np.ndarray, dipole_matrix: Any, efield: Any
    ) -> tuple[np.ndarray, Any, Any]:
        """
        Prepare quantities in appropriate units for calculation.

        This method can be overridden by subclasses to handle
        unit conversion specific to their algorithm.

        Parameters
        ----------
        H0 : np.ndarray
            Hamiltonian
        dipole_matrix : object
            Dipole matrices
        efield : ElectricField
            Electric field

        Returns
        -------
        tuple
            (H0_prepared, dipole_prepared, efield_prepared)
        """
        return H0, dipole_matrix, efield
