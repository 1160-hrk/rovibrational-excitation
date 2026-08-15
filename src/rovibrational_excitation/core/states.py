"""Explicit immutable quantum-state kinds for typed propagation boundaries."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..dynamics.algorithms.validation import (
    NUMERICAL_VALIDATION_EPSILON_FACTOR,
    density_matrix_tolerance,
    validate_density_matrix_properties,
)


def _copy_vector(value: Any, *, name: str) -> NDArray[np.complex128]:
    try:
        vector = np.array(value, dtype=np.complex128, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain complex-compatible numbers") from exc
    if vector.ndim != 1 or vector.size < 1:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _normalization_tolerance(dimension: int) -> float:
    return NUMERICAL_VALIDATION_EPSILON_FACTOR * max(1, dimension) * np.finfo(float).eps


@dataclass(frozen=True, slots=True, eq=False)
class PureState:
    """A finite, normalized wavefunction owned by the typed boundary.

    Construction validates but never normalizes caller data. The stored array
    is a defensive read-only complex128 copy.
    """

    amplitudes: NDArray[np.complex128]

    def __post_init__(self) -> None:
        amplitudes = _copy_vector(self.amplitudes, name="pure-state amplitudes")
        norm_squared = float(np.vdot(amplitudes, amplitudes).real)
        if abs(norm_squared - 1.0) > _normalization_tolerance(amplitudes.size):
            raise ValueError(
                "pure-state amplitudes must have norm one; "
                f"got norm squared {norm_squared!r}"
            )
        amplitudes.setflags(write=False)
        object.__setattr__(self, "amplitudes", amplitudes)

    @property
    def dimension(self) -> int:
        """Return the Hilbert-space dimension."""
        return int(self.amplitudes.size)


@dataclass(frozen=True, slots=True, eq=False, init=False)
class IncoherentEnsemble:
    """Normalized statistical mixture using norm-squared raw weights.

    Each input vector carries raw weight ``<psi|psi>``. Zero-norm vectors are
    skipped, nonzero components are normalized, and the weights are normalized
    to sum to one. No coherent cross terms are introduced.
    """

    states: tuple[PureState, ...]
    weights: NDArray[np.float64]

    def __init__(self, weighted_vectors: Iterable[Any]) -> None:
        raw_vectors = list(weighted_vectors)
        if not raw_vectors:
            raise ValueError("incoherent ensemble must not be empty")

        states: list[PureState] = []
        raw_weights: list[float] = []
        dimension: int | None = None
        for index, value in enumerate(raw_vectors):
            vector = _copy_vector(value, name=f"ensemble vector {index}")
            if dimension is None:
                dimension = int(vector.size)
            elif vector.size != dimension:
                raise ValueError(
                    "all ensemble vectors must have the same dimension; "
                    f"vector 0 has {dimension}, vector {index} has {vector.size}"
                )

            weight = float(np.vdot(vector, vector).real)
            if not np.isfinite(weight):
                raise ValueError("ensemble vector weights must be finite")
            if weight == 0.0:
                continue
            states.append(PureState(vector / np.sqrt(weight)))
            raw_weights.append(weight)

        if not raw_weights:
            raise ValueError("at least one ensemble vector must have non-zero norm")

        weights = np.asarray(raw_weights, dtype=np.float64)
        weight_sum = float(weights.sum())
        if not np.isfinite(weight_sum):
            raise ValueError("sum of ensemble vector weights must be finite")
        weights /= weight_sum
        weights.setflags(write=False)
        object.__setattr__(self, "states", tuple(states))
        object.__setattr__(self, "weights", weights)

    @property
    def dimension(self) -> int:
        """Return the shared Hilbert-space dimension."""
        return self.states[0].dimension

    def density_matrix(self) -> NDArray[np.complex128]:
        """Return the incoherent density operator without cross terms."""
        density = np.zeros((self.dimension, self.dimension), dtype=np.complex128)
        for state, weight in zip(self.states, self.weights):
            density += float(weight) * np.outer(
                state.amplitudes, state.amplitudes.conj()
            )
        density.setflags(write=False)
        return density


@dataclass(frozen=True, slots=True, eq=False)
class DensityState:
    """A finite physical density matrix whose trace is already one.

    Construction never normalizes, clips, symmetrizes, or otherwise repairs
    caller data. The stored matrix is a defensive read-only complex128 copy.
    """

    matrix: NDArray[np.complex128]

    def __post_init__(self) -> None:
        try:
            matrix = np.array(self.matrix, dtype=np.complex128, copy=True)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "density-state matrix must contain complex-compatible numbers"
            ) from exc
        validate_density_matrix_properties(matrix)
        trace = np.trace(matrix)
        tolerance = density_matrix_tolerance(matrix)
        if abs(float(trace.real) - 1.0) > tolerance:
            raise ValueError(
                "density-state matrix must have trace one within numerical "
                f"tolerance; got {trace.real!r}"
            )
        matrix.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)

    @classmethod
    def from_pure_state(cls, state: PureState) -> DensityState:
        """Construct ``|psi><psi|`` from an explicitly typed pure state."""
        if not isinstance(state, PureState):
            raise TypeError("state must be a PureState")
        return cls(np.outer(state.amplitudes, state.amplitudes.conj()))

    @property
    def dimension(self) -> int:
        """Return the Hilbert-space dimension."""
        return int(self.matrix.shape[0])


__all__ = ["DensityState", "IncoherentEnsemble", "PureState"]
