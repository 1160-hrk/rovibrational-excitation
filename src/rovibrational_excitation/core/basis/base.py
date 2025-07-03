"""
Abstract base class for quantum basis sets.
"""

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .hamiltonian import Hamiltonian


class BasisBase(ABC):
    """
    Abstract base class for quantum basis sets.

    All concrete basis classes should inherit from this class and implement
    the required abstract methods.
    """

    @abstractmethod
    def size(self) -> int:
        """
        Return the total number of basis states (dimension).

        Returns
        -------
        int
            The dimension of the basis set.
        """
        pass

    @abstractmethod
    def get_index(self, state: tuple | Any) -> int:
        """
        Get the index corresponding to a quantum state.

        Parameters
        ----------
        state : tuple or any
            Quantum state specification (format depends on the basis type).

        Returns
        -------
        int
            Index of the state in the basis.
        """
        pass

    @abstractmethod
    def get_state(self, index: int) -> Any:
        """
        Get the quantum state corresponding to an index.

        Parameters
        ----------
        index : int
            Index in the basis.

        Returns
        -------
        any
            Quantum state specification.
        """
        pass

    @abstractmethod
    def generate_H0(self, **kwargs) -> "Hamiltonian":
        """
        Generate the free Hamiltonian for this basis.

        Parameters
        ----------
        **kwargs
            System-specific parameters for Hamiltonian construction.

        Returns
        -------
        Hamiltonian
            Hamiltonian object with unit information.
        """
        pass

    def __repr__(self) -> str:
        """Default representation."""
        return f"{self.__class__.__name__}(size={self.size()})"
