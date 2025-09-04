"""
Two-level system basis.
"""

import numpy as np

from .base import BasisBase
from .hamiltonian import Hamiltonian
from rovibrational_excitation.core.units.converters import converter


class TwoLevelBasis(BasisBase):
    """
    Two-level system basis: |0⟩ and |1⟩.

    Simple quantum system with ground state |0⟩ and excited state |1⟩.
    
    Parameters
    ----------
    energy_gap : float, optional
        エネルギーギャップ（input_unitsで指定した単位）
    input_units : str, optional
        入力パラメータの単位（"rad/fs", "cm^-1", "THz", "eV"など）
    output_units : str, optional
        出力ハミルトニアンの単位（"J" または "rad/fs"）
    """

    def __init__(
        self,
        energy_gap: float | None = None,
        input_units: str = "rad/fs",
        output_units: str = "J",
    ):
        """Initialize two-level basis with physical parameters."""
        self.input_units = input_units
        self.output_units = output_units
        
        # エネルギーギャップの単位変換と保存
        if energy_gap is None:
            self.gap_rad_pfs = 1.0  # デフォルト値
        else:
            if input_units in converter.get_supported_units("frequency"):
                self.gap_rad_pfs = energy_gap * converter.convert_frequency(1.0, input_units, "rad/fs")
            elif input_units in converter.get_supported_units("energy"):
                # エネルギー単位からrad/fsへの変換
                gap_J = converter.convert_energy(energy_gap, input_units, "J")
                self.gap_rad_pfs = gap_J / Hamiltonian._HBAR * 1e-15
            else:
                raise ValueError(
                    f"Unsupported input_units '{input_units}'.\n"
                    f"Supported frequency units: {list(converter.get_supported_units('frequency'))}\n"
                    f"Supported energy units: {list(converter.get_supported_units('energy'))}"
                )
        
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

    def generate_H0(self) -> Hamiltonian:
        """
        保存されたパラメータから二準位ハミルトニアンを生成

        H = |0⟩⟨0| × 0 + |1⟩⟨1| × energy_gap

        Returns
        -------
        Hamiltonian
            2x2 diagonal Hamiltonian object with unit information.
        """
        H0_matrix = np.diag(np.array([0.0, self.gap_rad_pfs]))
        
        basis_info = {
            "basis_type": "TwoLevel",
            "size": 2,
            "energy_gap_rad_pfs": self.gap_rad_pfs,
            "input_units": self.input_units,
            "output_units": self.output_units,
        }
        
        hamiltonian = Hamiltonian(H0_matrix, "rad/fs", basis_info)
        
        # Convert to requested units
        if self.output_units == "J":
            return hamiltonian.to_energy_units()
        else:
            return hamiltonian

    def generate_H0_with_params(
        self,
        energy_gap=None,
        energy_gap_units=None,
        return_energy_units=None,
        units=None,
        **kwargs,
    ) -> Hamiltonian:
        """
        一時的にパラメータを変更してハミルトニアンを生成（後方互換性）

        Parameters
        ----------
        energy_gap : float, optional
            エネルギーギャップの値（単位はenergy_gap_unitsで指定）
        energy_gap_units : str, optional
            energy_gapの単位
        return_energy_units : bool, optional
            TrueならJ、Falseならrad/fsで返す（非推奨）
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
        if energy_gap is None:
            gap_rad_pfs = self.gap_rad_pfs
        else:
            if energy_gap_units is None:
                energy_gap_units = self.input_units
            
            unit_key = energy_gap_units

            # Decide whether provided unit is frequency or energy and convert to rad/fs
            if unit_key in converter.get_supported_units("frequency"):
                gap_rad_pfs = energy_gap * converter.convert_frequency(1.0, unit_key, "rad/fs")
            elif unit_key in converter.get_supported_units("energy"):
                # Convert to Joule first, then to rad/fs using ℏ
                gap_J = converter.convert_energy(energy_gap, unit_key, "J")
                gap_rad_pfs = gap_J / Hamiltonian._HBAR * 1e-15
            elif unit_key == "energy":  # backward compatibility (J)
                gap_rad_pfs = energy_gap / Hamiltonian._HBAR * 1e-15
            elif unit_key == "frequency":  # backward compatibility (rad/fs)
                gap_rad_pfs = energy_gap
            else:
                raise ValueError(
                    f"Unsupported energy_gap_units '{unit_key}'.\n"
                    f"Supported frequency units: {list(converter.get_supported_units('frequency'))}\n"
                    f"Supported energy units: {list(converter.get_supported_units('energy'))}"
                )

        H0_matrix = np.diag(np.array([0.0, gap_rad_pfs]))
        basis_info = {
            "basis_type": "TwoLevel",
            "size": 2,
            "energy_gap_rad_pfs": gap_rad_pfs,
            "energy_gap_units": energy_gap_units,
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
        return f"TwoLevelBasis(gap={self.gap_rad_pfs:.3f} rad/fs, |0⟩, |1⟩)"
