"""
Vibrational ladder system basis (rotation-free).
"""
import numpy as np
from .base import BasisBase


class VibLadderBasis(BasisBase):
    """
    Vibrational ladder basis: |v=0⟩, |v=1⟩, ..., |v=V_max⟩.
    
    Pure vibrational system without rotational degrees of freedom.
    """
    
    def __init__(self, V_max: int, omega_rad_phz: float = 1.0, 
                 delta_omega_rad_phz: float = 0.0):
        """
        Initialize vibrational ladder basis.
        
        Parameters
        ----------
        V_max : int
            Maximum vibrational quantum number.
        omega_rad_phz : float
            Vibrational frequency (rad/fs).
        delta_omega_rad_phz : float
            Anharmonicity parameter (rad/fs).
        """
        self.V_max = V_max
        self.omega_rad_phz = omega_rad_phz
        self.delta_omega_rad_phz = delta_omega_rad_phz
        
        self.basis = np.array([[v] for v in range(V_max + 1)])
        self.V_array = self.basis[:, 0]
        self.index_map = {(v,): v for v in range(V_max + 1)}
    
    def size(self) -> int:
        """Return the number of vibrational levels."""
        return self.V_max + 1
    
    def get_index(self, state) -> int:
        """
        Get index for a vibrational state.
        
        Parameters
        ----------
        state : int or tuple
            State specification: v or (v,).
            
        Returns
        -------
        int
            Index of the vibrational state.
        """
        if isinstance(state, (int, np.integer)):
            v = int(state)
            if 0 <= v <= self.V_max:
                return v
            else:
                raise ValueError(f"Invalid vibrational state {v}. Must be 0 <= v <= {self.V_max}.")
        
        if hasattr(state, '__iter__'):
            if not isinstance(state, tuple):
                state = tuple(state)
            if state in self.index_map:
                return self.index_map[state]
        
        raise ValueError(f"State {state} not found in vibrational ladder basis")
    
    def get_state(self, index: int):
        """
        Get state from index.
        
        Parameters
        ----------
        index : int
            Index (0 to V_max).
            
        Returns
        -------
        np.ndarray
            State array [v].
        """
        if not (0 <= index <= self.V_max):
            raise ValueError(f"Invalid index {index}. Must be 0 <= index <= {self.V_max}.")
        return self.basis[index]
    
    def generate_H0(self, omega_rad_phz=None, delta_omega_rad_phz=None, **kwargs) -> np.ndarray:
        """
        Generate vibrational Hamiltonian.
        
        H_vib = ω*(v+1/2) - Δω*(v+1/2)^2
        
        Parameters
        ----------
        omega_rad_phz : float, optional
            Vibrational frequency (rad/fs). If None, use instance value.
        delta_omega_rad_phz : float, optional
            Anharmonicity parameter (rad/fs). If None, use instance value.
        **kwargs
            Additional parameters (ignored).
            
        Returns
        -------
        np.ndarray
            Diagonal Hamiltonian matrix.
        """
        # Use instance values if not provided
        if omega_rad_phz is None:
            omega_rad_phz = self.omega_rad_phz
        if delta_omega_rad_phz is None:
            delta_omega_rad_phz = self.delta_omega_rad_phz
            
        vterm = self.V_array + 0.5
        energy = omega_rad_phz * vterm - delta_omega_rad_phz * vterm**2
        return np.diag(energy)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"VibLadderBasis(V_max={self.V_max}, size={self.size()})" 