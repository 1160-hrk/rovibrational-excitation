"""Typed objective evaluations that preserve optimizer-specific arithmetic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class TargetPopulationEvaluation:
    """Terminal target population and its complementary infidelity."""

    fidelity: float
    infidelity: float


class TargetPopulationEvaluator(Protocol):
    """Evaluate terminal population without prescribing its arithmetic path."""

    def evaluate(self, state: np.ndarray) -> TargetPopulationEvaluation: ...


@dataclass(frozen=True, slots=True)
class IndexedTargetPopulation:
    """Preserve the runner-level ``abs(state[index])**2`` calculation."""

    target_index: int

    def evaluate(self, state: np.ndarray) -> TargetPopulationEvaluation:
        fidelity = float(np.abs(state[self.target_index]) ** 2)
        return TargetPopulationEvaluation(
            fidelity=fidelity,
            infidelity=1.0 - fidelity,
        )


@dataclass(frozen=True, slots=True, eq=False)
class VectorTargetPopulation:
    """Preserve the adjoint-level ``vdot(target, state)`` calculation."""

    target_state: ComplexArray

    def overlap(self, state: np.ndarray) -> np.complex128:
        return np.complex128(np.vdot(self.target_state, state))

    def evaluate_with_overlap(
        self, state: np.ndarray
    ) -> tuple[np.complex128, TargetPopulationEvaluation]:
        overlap = self.overlap(state)
        fidelity = float(np.abs(overlap) ** 2)
        return overlap, TargetPopulationEvaluation(
            fidelity=fidelity,
            infidelity=1.0 - fidelity,
        )

    def evaluate(self, state: np.ndarray) -> TargetPopulationEvaluation:
        return self.evaluate_with_overlap(state)[1]


@dataclass(frozen=True, slots=True)
class DiscreteL2TargetObjective:
    """GRAPE objective ``1-F + lambda_a/2 * sum(E**2)``."""

    lambda_a: float

    def evaluate(
        self,
        target: TargetPopulationEvaluation,
        controls_v_per_m: np.ndarray,
    ) -> float:
        return float(
            target.infidelity
            + 0.5
            * float(self.lambda_a)
            * float(np.sum(controls_v_per_m * controls_v_per_m))
        )


__all__ = [
    "DiscreteL2TargetObjective",
    "IndexedTargetPopulation",
    "TargetPopulationEvaluation",
    "TargetPopulationEvaluator",
    "VectorTargetPopulation",
]
