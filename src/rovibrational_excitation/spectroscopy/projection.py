"""Typed polarization projections for spectroscopy measurements."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rovibrational_excitation.core.model import Axis


def _validate_axes(axes: tuple[Axis, ...]) -> None:
    if (
        not isinstance(axes, tuple)
        or not 1 <= len(axes) <= 3
        or not all(isinstance(axis, Axis) for axis in axes)
    ):
        raise TypeError("axes must contain between one and three typed Axis values")
    if len(set(axes)) != len(axes):
        raise ValueError("axes must contain unique Cartesian components")


def _normalize_jones(
    vector: np.ndarray,
    *,
    axes: tuple[Axis, ...],
    name: str,
) -> np.ndarray:
    array = np.asarray(vector, dtype=np.complex128)
    expected_shape = (len(axes),)
    if array.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape} matching axes")
    if not np.all(np.isfinite(array.real)) or not np.all(np.isfinite(array.imag)):
        raise ValueError(f"{name} must be a finite nonzero Jones ket")
    norm = float(np.linalg.norm(array))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError(f"{name} must be a finite nonzero Jones ket")

    normalized = np.array(array / norm, dtype=np.complex128, copy=True)
    normalized.setflags(write=False)
    return normalized


@dataclass(frozen=True, slots=True, eq=False)
class CartesianProjection:
    """A normalized interaction Jones ket in an ordered Cartesian basis."""

    axes: tuple[Axis, ...]
    interaction: np.ndarray

    def __post_init__(self) -> None:
        _validate_axes(self.axes)
        object.__setattr__(
            self,
            "interaction",
            _normalize_jones(self.interaction, axes=self.axes, name="interaction"),
        )

    @classmethod
    def from_jones(
        cls,
        *,
        axes: str,
        interaction: np.ndarray,
    ) -> CartesianProjection:
        """Construct from exact lowercase Cartesian labels without coercion."""
        return cls(
            axes=_typed_axes_from_string(axes),
            interaction=interaction,
        )

    @property
    def axes_string(self) -> str:
        """Return the ordered Cartesian labels used by the Jones ket."""
        return "".join(axis.value for axis in self.axes)


@dataclass(frozen=True, slots=True, eq=False)
class CartesianAnalyzerProjection:
    """Normalized interaction and analyzer Jones kets on the same axes."""

    axes: tuple[Axis, ...]
    interaction: np.ndarray
    analyzer: np.ndarray

    def __post_init__(self) -> None:
        _validate_axes(self.axes)
        object.__setattr__(
            self,
            "interaction",
            _normalize_jones(self.interaction, axes=self.axes, name="interaction"),
        )
        object.__setattr__(
            self,
            "analyzer",
            _normalize_jones(self.analyzer, axes=self.axes, name="analyzer"),
        )

    @classmethod
    def from_jones(
        cls,
        *,
        axes: str,
        interaction: np.ndarray,
        analyzer: np.ndarray,
    ) -> CartesianAnalyzerProjection:
        """Construct from exact lowercase Cartesian labels without coercion."""
        return cls(
            axes=_typed_axes_from_string(axes),
            interaction=interaction,
            analyzer=analyzer,
        )

    @property
    def axes_string(self) -> str:
        """Return the ordered Cartesian labels shared by both Jones kets."""
        return "".join(axis.value for axis in self.axes)


def _typed_axes_from_string(axes: str) -> tuple[Axis, ...]:
    if not isinstance(axes, str):
        raise TypeError("axes must be a string")
    if not 1 <= len(axes) <= 3 or any(axis not in "xyz" for axis in axes):
        raise ValueError("axes must be an exact lowercase ordered subset of xyz")
    if len(set(axes)) != len(axes):
        raise ValueError("axes must contain unique Cartesian components")
    return tuple(Axis(axis) for axis in axes)


__all__ = ["CartesianAnalyzerProjection", "CartesianProjection"]
