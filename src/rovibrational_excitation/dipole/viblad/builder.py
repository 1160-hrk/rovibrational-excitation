"""
Vibrational ladder system dipole matrix builder.
"""

from typing import Literal, cast

import numpy as np

from rovibrational_excitation.core.basis.viblad import VibLadderBasis
from rovibrational_excitation.dipole.vib.harmonic import tdm_vib_harm
from rovibrational_excitation.dipole.vib.morse import omega01_domega_to_N, tdm_vib_morse


class VibLadderDipoleMatrix:
    """
    Dipole matrix for vibrational ladder systems.

    For vibrational systems without rotation, only the z-component
    of the dipole moment is typically non-zero (parallel transitions).
    """

    def __init__(
        self,
        basis: VibLadderBasis,
        mu0: float = 1.0,
        potential_type: Literal["harmonic", "morse"] = "harmonic",
    ):
        """
        Initialize vibrational ladder dipole matrix.

        Parameters
        ----------
        basis : VibLadderBasis
            Vibrational ladder basis.
        mu0 : float
            Dipole matrix element scaling factor.
        potential_type : {'harmonic', 'morse'}
            Type of vibrational potential.
        """
        if not isinstance(basis, VibLadderBasis):
            raise TypeError("basis must be VibLadderBasis")

        self.basis = basis
        self.mu0 = mu0
        self.potential_type = potential_type.lower()

        if self.potential_type not in ("harmonic", "morse"):
            raise ValueError("potential_type must be 'harmonic' or 'morse'")

        # Set up Morse parameters if needed
        if self.potential_type == "morse":
            omega01_domega_to_N(basis.omega_rad_pfs, basis.delta_omega_rad_pfs)

        # Cache for computed matrices
        self._cache: dict[str, np.ndarray] = {}

    def mu(self, axis: Literal["x", "y", "z"] = "z") -> np.ndarray:
        """
        Get dipole matrix for specified axis.

        Parameters
        ----------
        axis : {'x', 'y', 'z'}
            Dipole axis direction.

        Returns
        -------
        np.ndarray
            Dipole matrix of shape (V_max+1, V_max+1).
        """
        if axis in self._cache:
            return self._cache[axis]

        dim = self.basis.size()
        matrix = np.zeros((dim, dim), dtype=np.complex128)

        if axis == "z":
            # For vibrational transitions, z-component is typically the relevant one
            vib_func = tdm_vib_morse if self.potential_type == "morse" else tdm_vib_harm

            for i in range(dim):
                v1 = self.basis.V_array[i]
                for j in range(dim):
                    v2 = self.basis.V_array[j]
                    vib_element = vib_func(v1, v2)
                    if vib_element != 0.0:
                        matrix[i, j] = self.mu0 * vib_element

        elif axis == "x":
            # For pure vibrational systems, x and y components are typically zero
            # (no rotational mixing)
            matrix = np.diag(np.ones(len(self.basis.V_array)-1), 1)
            matrix += np.diag(np.ones(len(self.basis.V_array)-1), -1)
            # matrix = np.zeros((dim, dim), dtype=np.complex128)
        elif axis == "y":
            matrix = np.zeros((dim, dim), dtype=np.complex128)

        else:
            raise ValueError(f"Invalid axis '{axis}'. Must be 'x', 'y', or 'z'.")

        self._cache[axis] = matrix
        return matrix

    @property
    def mu_x(self) -> np.ndarray:
        """x-component of dipole matrix."""
        return self.mu("x")

    @property
    def mu_y(self) -> np.ndarray:
        """y-component of dipole matrix."""
        return self.mu("y")

    @property
    def mu_z(self) -> np.ndarray:
        """z-component of dipole matrix."""
        return self.mu("z")

    def stacked(self, order: str = "xyz") -> np.ndarray:
        """
        Return stacked dipole matrices.

        Parameters
        ----------
        order : str
            Order of axes (e.g., 'xyz', 'z').

        Returns
        -------
        np.ndarray
            Array of shape (len(order), dim, dim).
        """
        matrices = []
        for ax in order:
            if ax in ["x", "y", "z"]:
                matrices.append(self.mu(cast(Literal["x", "y", "z"], ax)))
            else:
                raise ValueError(f"Invalid axis '{ax}'. Must be 'x', 'y', or 'z'.")
        return np.stack(matrices)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"VibLadderDipoleMatrix(mu0={self.mu0}, "
            f"potential='{self.potential_type}', "
            f"V_max={self.basis.V_max})"
        )
