"""
Two-level system basis.
"""
import numpy as np
from .base import BasisBase


class TwoLevelBasis(BasisBase):
    """
    Two-level system basis: |0⟩ and |1⟩.
    
    Simple quantum system with ground state |0⟩ and excited state |1⟩.
    """
    
    def __init__(self):
        """Initialize two-level basis."""
        self.basis = np.array([[0], [1]])  # |0⟩, |1⟩
        self.index_map = {(0,): 0, (1,): 1}
    
    def size(self) -> int:
        """Return the dimension (always 2 for two-level system)."""
        return 2
    
    def get_index(self, state) -> int:
        """
        Get index for a two-level state.
        
        Parameters
        ----------
        state : int or tuple
            State specification: 0 or 1, or (0,) or (1,).
            
        Returns
        -------
        int
            Index of the state (0 or 1).
        """
        if isinstance(state, (int, np.integer)):
            if state in [0, 1]:
                return int(state)
            else:
                raise ValueError(f"Invalid state {state}. Must be 0 or 1.")
        
        if hasattr(state, '__iter__'):
            if not isinstance(state, tuple):
                state = tuple(state)
            if state in self.index_map:
                return self.index_map[state]
        
        raise ValueError(f"State {state} not found in two-level basis")
    
    def get_state(self, index: int):
        """
        Get state from index.
        
        Parameters
        ----------
        index : int
            Index (0 or 1).
            
        Returns
        -------
        np.ndarray
            State array [level].
        """
        if index not in [0, 1]:
            raise ValueError(f"Invalid index {index}. Must be 0 or 1.")
        return self.basis[index]
    
    def generate_H0(self, energy_gap=1.0, **kwargs) -> np.ndarray:
        """
        Generate two-level Hamiltonian.
        
        H = |0⟩⟨0| × 0 + |1⟩⟨1| × energy_gap
        
        Parameters
        ----------
        energy_gap : float
            Energy difference between |1⟩ and |0⟩ states.
        **kwargs
            Additional parameters (ignored).
            
        Returns
        -------
        np.ndarray
            2x2 diagonal Hamiltonian matrix.
        """
        return np.diag([0.0, energy_gap])
    
    def __repr__(self) -> str:
        """String representation."""
        return "TwoLevelBasis(|0⟩, |1⟩)" 