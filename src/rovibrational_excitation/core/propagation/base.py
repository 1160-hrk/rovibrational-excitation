"""
Base class for time propagation algorithms.

This module provides the abstract base class that all propagators
should inherit from.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
import numpy as np

from ..units.validators import validator


class PropagatorBase(ABC):
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
        self._last_validation_warnings = []

    @abstractmethod
    def propagate(
        self, hamiltonian, efield, dipole_matrix, initial_state: np.ndarray, **kwargs
    ) -> Any:
        """
        Propagate the quantum state forward in time.

        Parameters
        ----------
        hamiltonian : object
            Hamiltonian object
        efield : ElectricField
            Electric field object
        dipole_matrix : object
            Dipole moment matrices
        initial_state : np.ndarray
            Initial quantum state
        **kwargs
            Additional algorithm-specific parameters

        Returns
        -------
        Any
            Final state or trajectory (can be array or tuple)
        """
        pass

    def validate_inputs(
        self,
        H0: np.ndarray,
        dipole_matrix,
        efield,
        expected_H0_units: str = "J",
        expected_dipole_units: str = "C*m",
    ) -> bool:
        """
        Validate input parameters.

        Parameters
        ----------
        H0 : np.ndarray
            Hamiltonian matrix
        dipole_matrix : object
            Dipole moment matrices
        efield : ElectricField
            Electric field object
        expected_H0_units : str
            Expected units for Hamiltonian
        expected_dipole_units : str
            Expected units for dipole moments

        Returns
        -------
        bool
            True if validation passes, False otherwise
        """
        if not self.validate_units:
            return True

        self._last_validation_warnings = validator.validate_propagation_units(
            H0, dipole_matrix, efield, expected_H0_units, expected_dipole_units
        )

        return len(self._last_validation_warnings) == 0

    def get_validation_warnings(self) -> list:
        """Get the last validation warnings."""
        return self._last_validation_warnings.copy()

    def print_validation_warnings(self):
        """Print validation warnings if any."""
        if self._last_validation_warnings:
            print("⚠️  単位検証で以下の警告が検出されました:")
            for i, warning in enumerate(self._last_validation_warnings, 1):
                print(f"   {i}. {warning}")

            if len(self._last_validation_warnings) >= 3:
                print(
                    "\n🚨 複数の重大な警告があります。計算結果を慎重にご確認ください。"
                )
        else:
            print("✅ 単位検証: すべて正常です")

    @abstractmethod
    def get_algorithm_name(self) -> str:
        """Get the name of the propagation algorithm."""
        pass

    def get_supported_backends(self) -> list:
        """Get list of supported computational backends."""
        return ["numpy"]

    def prepare_units(
        self, H0: np.ndarray, dipole_matrix, efield
    ) -> Tuple[np.ndarray, Any, Any]:
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
