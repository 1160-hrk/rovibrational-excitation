"""
Symmetric-top molecule basis (vibration + rotation + K + M).
================================================================

This is a *skeleton implementation* that mirrors the public API of
LinMolBasis so that downstream code (e.g. dipole builders) can already
import and instantiate the class.  The physical formulas (rotational
energy, anharmonic corrections, etc.) follow the simplest symmetric-top
expression and can be improved later.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .base import BasisBase
from .hamiltonian import Hamiltonian
from rovibrational_excitation.core.units.converters import converter


class SymTopBasis(BasisBase):
    """Basis |V J K M⟩ for symmetric-top molecules.

    Parameters
    ----------
    V_max, J_max : int
        Maximum vibrational and rotational quantum numbers.
    use_M : bool, default True
        Whether to include space-fixed magnetic quantum number *M*.
    omega, B, C, alpha, delta_omega : float, optional
        Usual rovibrational spectroscopic parameters.  All are interpreted in
        *input_units*.
    input_units, output_units : str
        Units for input parameters and the generated Hamiltonian.  Supported
        frequency/energy units are identical to LinMolBasis.
    """

    def __init__(
        self,
        V_max: int,
        J_max: int,
        *,
        omega: float | None = None,
        B: float | None = None,
        C: float | None = None,
        alpha: float = 0.0,
        delta_omega: float = 0.0,
        input_units: str = "rad/fs",
        output_units: str = "J",
    ):
        self.V_max = V_max
        self.J_max = J_max
        self.input_units = input_units
        self.output_units = output_units

        # -------------------------------------------------------------
        # Unit conversion (identical logic to LinMolBasis)
        # -------------------------------------------------------------
        if input_units in converter.get_supported_units("frequency"):
            conv = converter.convert_frequency(1.0, input_units, "rad/fs")
            self.omega_rad_pfs = omega * conv if omega is not None else 1.0
            self.B_rad_pfs = B * conv if B is not None else 1.0
            self.C_rad_pfs = C * conv if C is not None else 1.0
            self.alpha_rad_pfs = alpha * conv
            self.delta_omega_rad_pfs = delta_omega * conv
        elif input_units in converter.get_supported_units("energy"):
            econv = converter.convert_energy(1.0, input_units, "J")
            self.omega_rad_pfs = (
                omega * econv / Hamiltonian._HBAR * 1e-15 if omega is not None else 1.0
            )
            self.B_rad_pfs = (
                B * econv / Hamiltonian._HBAR * 1e-15 if B is not None else 1.0
            )
            self.C_rad_pfs = (
                C * econv / Hamiltonian._HBAR * 1e-15 if C is not None else 1.0
            )
            self.alpha_rad_pfs = alpha * econv / Hamiltonian._HBAR * 1e-15
            self.delta_omega_rad_pfs = delta_omega * econv / Hamiltonian._HBAR * 1e-15
        else:
            raise ValueError(
                f"Unsupported input_units '{input_units}'.\n"
                f"Supported frequency units: {list(converter.get_supported_units('frequency'))}\n"
                f"Supported energy units: {list(converter.get_supported_units('energy'))}"
            )

        # -------------------------------------------------------------
        # Generate basis table and index map
        # -------------------------------------------------------------
        self.basis = self._generate_basis()
        self.V_array = self.basis[:, 0]
        self.J_array = self.basis[:, 1]
        self.M_array = self.basis[:, 2]
        self.K_array = self.basis[:, 3]
        self.index_map = {tuple(state): idx for idx, state in enumerate(self.basis)}

    # ------------------------------------------------------------------
    # Basis generation helper
    # ------------------------------------------------------------------
    def _generate_basis(self):
        basis = []
        for V in range(self.V_max + 1):
            for J in range(self.J_max + 1):
                for M in range(-J, J + 1):
                    for K in range(-J, J + 1):
                        basis.append([V, J, M, K])
        return np.asarray(basis, dtype=np.int64)

    # ------------------------------------------------------------------
    # Required BasisBase interface
    # ------------------------------------------------------------------
    def size(self) -> int:  # noqa: D401 (simple name)
        return len(self.basis)

    def get_index(self, state) -> int:  # type: ignore[override]
        if hasattr(state, "__iter__") and not isinstance(state, tuple):
            state = tuple(state)
        idx = self.index_map.get(state)
        if idx is None:
            raise ValueError(f"State {state} not found in basis")
        return idx

    def get_state(self, index: int):  # type: ignore[override]
        return self.basis[index]

    # ------------------------------------------------------------------
    def generate_H0(self, **kwargs) -> Hamiltonian:  # type: ignore[override]
        """Return *placeholder* rotational–vibrational Hamiltonian.

        The expression
            E = ω(v+1/2) - Δω(v+1/2)^2
                +  (B - α(v+1/2)) J(J+1)
                +  (C-B) K²
        is used as a first-order approximation for a symmetric top.  This can
        be refined later.
        """
        vterm = self.V_array + 0.5
        jterm = self.J_array * (self.J_array + 1)
        kterm = self.K_array ** 2

        energy_freq = self.omega_rad_pfs * vterm - self.delta_omega_rad_pfs * vterm**2
        energy_freq += (self.B_rad_pfs - self.alpha_rad_pfs * vterm) * jterm
        energy_freq += (self.C_rad_pfs - self.B_rad_pfs) * kterm

        H0_matrix = np.diag(energy_freq)
        basis_info = {
            "basis_type": "SymTop",
            "V_max": self.V_max,
            "J_max": self.J_max,
            "size": self.size(),
            "omega_rad_pfs": self.omega_rad_pfs,
            "B_rad_pfs": self.B_rad_pfs,
            "C_rad_pfs": self.C_rad_pfs,
            "alpha_rad_pfs": self.alpha_rad_pfs,
            "delta_omega_rad_pfs": self.delta_omega_rad_pfs,
            "input_units": self.input_units,
            "output_units": self.output_units,
        }

        ham = Hamiltonian(H0_matrix, "rad/fs", basis_info)
        return ham.to_energy_units() if self.output_units == "J" else ham

    # ------------------------------------------------------------------
    def __repr__(self):  # noqa: D401
        return (
            f"SymTopBasis(V_max={self.V_max}, J_max={self.J_max}, size={self.size()})"
        ) 