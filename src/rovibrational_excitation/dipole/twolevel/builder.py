"""
Two-level system dipole matrix builder.
"""

from typing import Literal, cast

import numpy as np

from rovibrational_excitation.core.basis.twolevel import TwoLevelBasis


class TwoLevelDipoleMatrix:
    """
    Dipole matrix for two-level systems.

    For a two-level system, the dipole matrix typically has the form:
    μ = μ₀ * (|0⟩⟨1| + |1⟩⟨0|)  (x-direction)
    μ = μ₀ * i(|1⟩⟨0| - |0⟩⟨1|)  (y-direction)
    μ = 0                         (z-direction, typically)
    """

    def __init__(self, basis: TwoLevelBasis, mu0: float = 1.0):
        """
        Initialize two-level dipole matrix.

        Parameters
        ----------
        basis : TwoLevelBasis
            Two-level basis (must have size=2).
        mu0 : float
            Dipole matrix element magnitude.
        """
        if not isinstance(basis, TwoLevelBasis):
            raise TypeError("basis must be TwoLevelBasis")
        if basis.size() != 2:
            raise ValueError("basis must have exactly 2 states")

        self.basis = basis
        self.mu0 = mu0

        # Cache for computed matrices
        self._cache: dict[str, np.ndarray] = {}

    def mu(self, axis: Literal["x", "y", "z"] = "x") -> np.ndarray:
        """
        Get dipole matrix for specified axis.

        Parameters
        ----------
        axis : {'x', 'y', 'z'}
            Dipole axis direction.

        Returns
        -------
        np.ndarray
            2x2 dipole matrix.
        """
        if axis in self._cache:
            return self._cache[axis]

        if axis == "x":
            # σ_x = |0⟩⟨1| + |1⟩⟨0|
            matrix = self.mu0 * np.array([[0, 1], [1, 0]], dtype=np.complex128)
        elif axis == "y":
            # σ_y = i(|1⟩⟨0| - |0⟩⟨1|)
            matrix = self.mu0 * np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        elif axis == "z":
            # Typically zero for electric dipole transitions in two-level atoms
            matrix = np.zeros((2, 2), dtype=np.complex128)
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
            Order of axes (e.g., 'xyz', 'xy').

        Returns
        -------
        np.ndarray
            Array of shape (len(order), 2, 2).
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
        return f"TwoLevelDipoleMatrix(mu0={self.mu0})"
