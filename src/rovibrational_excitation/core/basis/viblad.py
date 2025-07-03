"""
Vibrational ladder system basis (rotation-free).
"""

import numpy as np

from .base import BasisBase
from .hamiltonian import Hamiltonian


class VibLadderBasis(BasisBase):
    """
    Vibrational ladder basis: |v=0⟩, |v=1⟩, ..., |v=V_max⟩.

    Pure vibrational system without rotational degrees of freedom.
    """

    def __init__(
        self, V_max: int, omega_rad_pfs: float = 1.0, delta_omega_rad_pfs: float = 0.0
    ):
        """
        Initialize vibrational ladder basis.

        Parameters
        ----------
        V_max : int
            Maximum vibrational quantum number.
        omega_rad_pfs : float
            Vibrational frequency (rad/fs).
        delta_omega_rad_pfs : float
            Anharmonicity parameter (rad/fs).
        """
        self.V_max = V_max
        self.omega_rad_pfs = omega_rad_pfs
        self.delta_omega_rad_pfs = delta_omega_rad_pfs

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
        if isinstance(state, int | np.integer):
            v = int(state)
            if 0 <= v <= self.V_max:
                return v
            else:
                raise ValueError(
                    f"Invalid vibrational state {v}. Must be 0 <= v <= {self.V_max}."
                )

        if hasattr(state, "__iter__"):
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
            raise ValueError(
                f"Invalid index {index}. Must be 0 <= index <= {self.V_max}."
            )
        return self.basis[index]

    def generate_H0(
        self, 
        omega_rad_pfs=None, 
        delta_omega_rad_pfs=None, 
        units="J",
        **kwargs
    ) -> Hamiltonian:
        """
        Generate vibrational Hamiltonian.

        H_vib = ω*(v+1/2) - Δω*(v+1/2)^2

        Parameters
        ----------
        omega_rad_pfs : float, optional
            Vibrational frequency (rad/fs). If None, use instance value.
        delta_omega_rad_pfs : float, optional
            Anharmonicity parameter (rad/fs). If None, use instance value.
        units : {"J", "rad/fs"}, optional
            返すハミルトニアンの単位。デフォルトは"J"（エネルギー単位）
        **kwargs
            Additional parameters (ignored).

        Returns
        -------
        Hamiltonian
            Diagonal Hamiltonian object with unit information.
        """
        # Use instance values if not provided
        if omega_rad_pfs is None:
            omega_rad_pfs = self.omega_rad_pfs
        if delta_omega_rad_pfs is None:
            delta_omega_rad_pfs = self.delta_omega_rad_pfs

        vterm = self.V_array + 0.5
        energy_freq = omega_rad_pfs * vterm - delta_omega_rad_pfs * vterm**2
        
        # Create Hamiltonian in frequency units first
        H0_matrix = np.diag(energy_freq)
        
        # Create basis info for debugging
        basis_info = {
            "basis_type": "VibLadder",
            "V_max": self.V_max,
            "size": self.size(),
            "omega_rad_pfs": omega_rad_pfs,
            "delta_omega_rad_pfs": delta_omega_rad_pfs,
        }
        
        # Create Hamiltonian object in rad/fs
        hamiltonian = Hamiltonian(H0_matrix, "rad/fs", basis_info)
        
        # Convert to requested units
        if units == "J":
            return hamiltonian.to_energy_units()
        else:
            return hamiltonian

    def __repr__(self) -> str:
        """String representation."""
        return f"VibLadderBasis(V_max={self.V_max}, size={self.size()})"
