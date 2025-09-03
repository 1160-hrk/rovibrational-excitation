"""
Vibrational ladder system basis (rotation-free).
"""

import numpy as np

from .base import BasisBase
from .hamiltonian import Hamiltonian
from rovibrational_excitation.core.units.converters import converter

class VibLadderBasis(BasisBase):
    """
    Vibrational ladder basis: |v=0⟩, |v=1⟩, ..., |v=V_max⟩.

    Pure vibrational system without rotational degrees of freedom.
    
    Parameters
    ----------
    V_max : int
        最大振動量子数
    omega : float, optional
        振動周波数（input_unitsで指定した単位）
    delta_omega : float, optional
        非調和性パラメータ（input_unitsで指定した単位）
    input_units : str, optional
        入力パラメータの単位（"rad/fs", "cm^-1", "THz", "eV"など）
    output_units : str, optional
        出力ハミルトニアンの単位（"J" または "rad/fs"）
    """

    def __init__(
        self, 
        V_max: int, 
        omega: float | None = None, 
        delta_omega: float = 0.0,
        input_units: str = "rad/fs",
        output_units: str = "J",
    ):
        """
        Initialize vibrational ladder basis with physical parameters.

        Parameters
        ----------
        V_max : int
            Maximum vibrational quantum number.
        omega : float, optional
            Vibrational frequency (in input_units).
        delta_omega : float, optional
            Anharmonicity parameter (in input_units).
        input_units : str, optional
            Units of input parameters.
        output_units : str, optional
            Units of output Hamiltonian.
        """
        self.V_max = V_max
        self.input_units = input_units
        self.output_units = output_units
        
        # 物理パラメータの単位変換と保存
        if input_units in converter.get_supported_units("frequency"):
            conv = converter.convert_frequency(1.0, input_units, "rad/fs")
            self.omega_rad_pfs = omega * conv if omega is not None else 1.0
            self.delta_omega_rad_pfs = delta_omega * conv
        elif input_units in converter.get_supported_units("energy"):
            # エネルギー単位からrad/fsへの変換
            energy_conv = converter.convert_energy(1.0, input_units, "J")
            self.omega_rad_pfs = (omega * energy_conv / Hamiltonian._HBAR * 1e-15 
                                  if omega is not None else 1.0)
            self.delta_omega_rad_pfs = delta_omega * energy_conv / Hamiltonian._HBAR * 1e-15
        else:
            raise ValueError(
                f"Unsupported input_units '{input_units}'.\n"
                f"Supported frequency units: {list(converter.get_supported_units('frequency'))}\n"
                f"Supported energy units: {list(converter.get_supported_units('energy'))}"
            )

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

    def generate_H0(self) -> Hamiltonian:
        """
        保存されたパラメータから振動ハミルトニアンを生成

        H_vib = ω*(v+1/2) - Δω*(v+1/2)^2

        Returns
        -------
        Hamiltonian
            Diagonal Hamiltonian object with unit information.
        """
        vterm = self.V_array + 0.5
        energy_freq = (self.omega_rad_pfs + self.delta_omega_rad_pfs) * vterm - self.delta_omega_rad_pfs / 2 * vterm**2
        
        # Create Hamiltonian in frequency units first
        H0_matrix = np.diag(energy_freq)
        
        # Create basis info for debugging
        basis_info = {
            "basis_type": "VibLadder",
            "V_max": self.V_max,
            "size": self.size(),
            "omega_rad_pfs": self.omega_rad_pfs,
            "delta_omega_rad_pfs": self.delta_omega_rad_pfs,
            "input_units": self.input_units,
            "output_units": self.output_units,
        }
        
        # Create Hamiltonian object in rad/fs
        hamiltonian = Hamiltonian(H0_matrix, "rad/fs", basis_info)
        
        # Convert to requested units
        if self.output_units == "J":
            return hamiltonian.to_energy_units()
        else:
            return hamiltonian

    def generate_H0_with_params(
        self, 
        omega=None, 
        delta_omega=None, 
        units=None,
        input_units=None,
        **kwargs
    ) -> Hamiltonian:
        """
        一時的にパラメータを変更してハミルトニアンを生成（後方互換性）

        Parameters
        ----------
        omega : float, optional
            Vibrational frequency. If None, use instance value.
        delta_omega : float, optional
            Anharmonicity parameter. If None, use instance value.
        units : {"J", "rad/fs"}, optional
            返すハミルトニアンの単位
        input_units : str, optional
            Units of the input frequency.
        **kwargs
            Additional parameters (ignored).

        Returns
        -------
        Hamiltonian
            Diagonal Hamiltonian object with unit information.
        """
        # 単位の決定
        if input_units is None:
            input_units = self.input_units
        if units is None:
            units = self.output_units
            
        # Determine conversion factor from input_units to internal rad/fs
        if input_units not in converter.get_supported_units("frequency"):
            raise ValueError(
                f"Unsupported input_units '{input_units}'.\n"
                f"Supported units: {list(converter.get_supported_units('frequency'))}"
            )

        conv = converter.convert_frequency(1.0, input_units, "rad/fs")

        # Use instance values if not provided, otherwise convert supplied values
        if omega is None:
            omega_rad_pfs = self.omega_rad_pfs  # already rad/fs
        else:
            omega_rad_pfs = omega * conv

        if delta_omega is None:
            delta_omega_rad_pfs = self.delta_omega_rad_pfs  # already rad/fs
        else:
            delta_omega_rad_pfs = delta_omega * conv

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
            "input_units": input_units,
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
        return f"VibLadderBasis(V_max={self.V_max}, ω={self.omega_rad_pfs:.3f} rad/fs, size={self.size()})"
