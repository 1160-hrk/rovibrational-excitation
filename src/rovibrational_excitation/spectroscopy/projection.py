"""Typed polarization projections for spectroscopy measurements."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rovibrational_excitation.core.model import Axis


@dataclass(frozen=True, slots=True, eq=False)
class CartesianProjection:
    """A normalized interaction Jones ket in an ordered Cartesian basis."""

    axes: tuple[Axis, ...]
    interaction: np.ndarray

    def __post_init__(self) -> None:
        axes = self.axes
        if (
            not isinstance(axes, tuple)
            or not 1 <= len(axes) <= 3
            or not all(isinstance(axis, Axis) for axis in axes)
        ):
            raise TypeError("axes must contain between one and three typed Axis values")
        if len(set(axes)) != len(axes):
            raise ValueError("axes must contain unique Cartesian components")

        vector = np.asarray(self.interaction, dtype=np.complex128)
        expected_shape = (len(axes),)
        if vector.shape != expected_shape:
            raise ValueError(
                f"interaction must have shape {expected_shape} matching axes"
            )
        if not np.all(np.isfinite(vector.real)) or not np.all(np.isfinite(vector.imag)):
            raise ValueError("interaction must be finite nonzero Jones ket")
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError("interaction must be finite nonzero Jones ket")

        normalized = np.array(vector / norm, dtype=np.complex128, copy=True)
        normalized.setflags(write=False)
        object.__setattr__(self, "interaction", normalized)

    @classmethod
    def from_jones(
        cls,
        *,
        axes: str,
        interaction: np.ndarray,
    ) -> CartesianProjection:
        """Construct from exact lowercase Cartesian labels without coercion."""
        if not isinstance(axes, str):
            raise TypeError("axes must be a string")
        if not 1 <= len(axes) <= 3 or any(axis not in "xyz" for axis in axes):
            raise ValueError("axes must be an exact lowercase ordered subset of xyz")
        if len(set(axes)) != len(axes):
            raise ValueError("axes must contain unique Cartesian components")
        return cls(
            axes=tuple(Axis(axis) for axis in axes),
            interaction=interaction,
        )

    @property
    def axes_string(self) -> str:
        """Return the ordered Cartesian labels used by the Jones ket."""
        return "".join(axis.value for axis in self.axes)


__all__ = ["CartesianProjection"]
