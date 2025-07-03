"""
Two-level system basis.
"""

import numpy as np

from .base import BasisBase
from .hamiltonian import Hamiltonian


class TwoLevelBasis(BasisBase):
    """
    Two-level system basis: |0⟩ and |1⟩.

    Simple quantum system with ground state |0⟩ and excited state |1⟩.
    """

    def __init__(self):
        """Initialize two-level basis."""
        self.omega_rad_pfs = 1.0
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
        if isinstance(state, int | np.integer):
            if state in [0, 1]:
                return int(state)
            else:
                raise ValueError(f"Invalid state {state}. Must be 0 or 1.")

        if hasattr(state, "__iter__"):
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

    def generate_H0(self, energy_gap=None, energy_gap_units="energy", return_energy_units=None, units="J", **kwargs) -> Hamiltonian:
        """
        Generate two-level Hamiltonian.

        H = |0⟩⟨0| × 0 + |1⟩⟨1| × energy_gap

        Parameters
        ----------
        energy_gap : float, optional
            エネルギーギャップの値（単位はenergy_gap_unitsで指定）
        energy_gap_units : str, optional
            'energy'（J）または'frequency'（rad/fs）
        return_energy_units : bool, optional
            TrueならJ、Falseならrad/fsで返す
        units : {"J", "rad/fs"}, optional
            明示的な単位指定
        **kwargs
            Additional parameters (ignored).

        Returns
        -------
        Hamiltonian
            2x2 diagonal Hamiltonian object with unit information.
        """
        # デフォルト値
        gap = self.omega_rad_pfs
        gap_units = energy_gap_units
        if energy_gap is not None:
            gap = energy_gap

        # 単位変換
        if gap_units == "energy":
            # J → rad/fs
            _HBAR = 6.62607015e-034 / (2 * np.pi)
            gap_rad_pfs = gap / _HBAR * 1e-15
        elif gap_units == "frequency":
            gap_rad_pfs = gap
        else:
            raise ValueError(f"Unknown energy_gap_units: {gap_units}")

        H0_matrix = np.diag([0.0, gap_rad_pfs])
        basis_info = {
            "basis_type": "TwoLevel",
            "size": 2,
            "energy_gap_rad_pfs": gap_rad_pfs,
        }
        hamiltonian = Hamiltonian(H0_matrix, "rad/fs", basis_info)

        # 単位指定の優先順位（テストの期待に合わせる）
        if return_energy_units is not None:
            if return_energy_units:
                return hamiltonian.to_energy_units()
            else:
                return hamiltonian
        if units == "J":
            return hamiltonian.to_energy_units()
        else:
            return hamiltonian

    def __repr__(self) -> str:
        """String representation."""
        return "TwoLevelBasis(|0⟩, |1⟩)"
