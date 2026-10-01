"""Immutable model and coupling contracts shared by model and dynamics layers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

import numpy as np


class Axis(str, Enum):
    """A Cartesian operator axis."""

    X = "x"
    Y = "y"
    Z = "z"


class CouplingMode(str, Enum):
    """Implemented field-operator coupling representations."""

    CARTESIAN = "cartesian"
    SCALAR = "scalar"


@dataclass(frozen=True, slots=True)
class CouplingSpec:
    """Exclusive scalar or two-component Cartesian coupling selection."""

    mode: CouplingMode
    scalar_axis: Axis | None
    cartesian_axes: tuple[Axis, Axis] | None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, CouplingMode):
            raise TypeError("mode must be a CouplingMode")
        if self.mode is CouplingMode.SCALAR:
            if not isinstance(self.scalar_axis, Axis):
                raise ValueError("scalar_axis is required for scalar coupling")
            if self.cartesian_axes is not None:
                raise ValueError("cartesian_axes is not applicable to scalar coupling")
            return

        if self.scalar_axis is not None:
            raise ValueError("scalar_axis is not applicable to Cartesian coupling")
        axes = self.cartesian_axes
        if (
            not isinstance(axes, tuple)
            or len(axes) != 2
            or not all(isinstance(axis, Axis) for axis in axes)
        ):
            raise ValueError(
                "cartesian_axes must contain exactly two typed Cartesian axes"
            )

    @classmethod
    def scalar(cls, axis: Axis) -> CouplingSpec:
        """Construct one explicitly selected scalar coupling axis."""
        return cls(
            mode=CouplingMode.SCALAR,
            scalar_axis=axis,
            cartesian_axes=None,
        )

    @classmethod
    def cartesian(cls, axes: str) -> CouplingSpec:
        """Construct an ordered two-axis Cartesian coupling."""
        if not isinstance(axes, str):
            raise TypeError("axes must be a two-character string")
        axes_normalized = axes.lower()
        if len(axes_normalized) != 2 or any(
            axis not in "xyz" for axis in axes_normalized
        ):
            raise ValueError("axes must be like xy or zx")
        return cls(
            mode=CouplingMode.CARTESIAN,
            scalar_axis=None,
            cartesian_axes=(Axis(axes_normalized[0]), Axis(axes_normalized[1])),
        )

    @property
    def axes(self) -> tuple[str, ...]:
        """Return active operator axes in propagation order."""
        if self.mode is CouplingMode.SCALAR:
            assert self.scalar_axis is not None
            return (self.scalar_axis.value,)
        assert self.cartesian_axes is not None
        return tuple(axis.value for axis in self.cartesian_axes)

    def propagation_kwargs(self) -> dict[str, str]:
        """Project to the temporary explicit public solver fields."""
        if self.mode is CouplingMode.SCALAR:
            assert self.scalar_axis is not None
            return {"coupling_axis": self.scalar_axis.value}
        return {"axes": "".join(self.axes)}


def _basis_dimension(basis: Any) -> int:
    size = getattr(basis, "size", None)
    if not callable(size):
        raise TypeError("model basis must provide size()")
    dimension = size()
    if isinstance(dimension, bool) or not isinstance(dimension, (int, np.integer)):
        raise TypeError("model basis size must be an integer")
    if int(dimension) <= 0:
        raise ValueError("model basis size must be positive")
    return int(dimension)


@dataclass(frozen=True, slots=True, eq=False)
class SystemModel:
    """One dimensionally consistent basis, Hamiltonian, dipole, and coupling."""

    name: str
    basis: Any
    hamiltonian: Any
    dipole: Any
    coupling: CouplingSpec
    metadata: Mapping[str, Any]
    dimension: int = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("model name must be a nonempty string")
        if not isinstance(self.coupling, CouplingSpec):
            raise TypeError("model coupling must be a CouplingSpec")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("model metadata must be a mapping")
        if any(not isinstance(key, str) for key in self.metadata):
            raise TypeError("model metadata keys must be strings")

        dimension = _basis_dimension(self.basis)
        hamiltonian_dimension = getattr(self.hamiltonian, "size", None)
        if isinstance(hamiltonian_dimension, bool) or not isinstance(
            hamiltonian_dimension, (int, np.integer)
        ):
            raise TypeError("model Hamiltonian must expose an integer size")
        if int(hamiltonian_dimension) != dimension:
            raise ValueError(
                "Hamiltonian dimension must equal basis dimension; "
                f"got {hamiltonian_dimension} and {dimension}"
            )

        dipole_basis = getattr(self.dipole, "basis", None)
        if dipole_basis is not None:
            dipole_dimension = _basis_dimension(dipole_basis)
            if dipole_dimension != dimension:
                raise ValueError(
                    "dipole dimension must equal basis dimension; "
                    f"got {dipole_dimension} and {dimension}"
                )

        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


__all__ = ["Axis", "CouplingMode", "CouplingSpec", "SystemModel"]
