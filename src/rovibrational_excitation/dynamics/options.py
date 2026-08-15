"""Required immutable options for one propagation calculation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from ..core.execution import ExecutionPolicy
from .capabilities import PropagationAlgorithm


class ScalingMode(str, Enum):
    """Dimensional or explicitly nondimensional propagation."""

    DIMENSIONAL = "dimensional"
    NONDIMENSIONAL = "nondimensional"


class RenormalizationPolicy(str, Enum):
    """Whether a wavefunction is normalized after every integration step."""

    DISABLED = "disabled"
    PER_STEP = "per_step"


@dataclass(frozen=True, slots=True)
class PropagationOptions:
    """Typed source of truth for solver choices shared by all state paths."""

    algorithm: PropagationAlgorithm
    execution: ExecutionPolicy
    return_trajectory: bool
    sample_stride: int
    scaling: ScalingMode
    renormalization: RenormalizationPolicy

    def __post_init__(self) -> None:
        if not isinstance(self.algorithm, PropagationAlgorithm):
            raise TypeError("algorithm must be a PropagationAlgorithm")
        if not isinstance(self.execution, ExecutionPolicy):
            raise TypeError("execution must be an ExecutionPolicy")
        if not isinstance(self.return_trajectory, bool):
            raise TypeError("return_trajectory must be a bool")
        if isinstance(self.sample_stride, bool) or not isinstance(
            self.sample_stride, int
        ):
            raise TypeError("sample_stride must be a positive integer")
        if self.sample_stride <= 0:
            raise ValueError("sample_stride must be a positive integer")
        if not isinstance(self.scaling, ScalingMode):
            raise TypeError("scaling must be a ScalingMode")
        if not isinstance(self.renormalization, RenormalizationPolicy):
            raise TypeError("renormalization must be a RenormalizationPolicy")

    @property
    def algorithm_name(self) -> Literal["rk4", "split_operator"]:
        """Project to the temporary legacy solver algorithm string."""
        return self.algorithm.value

    @property
    def backend_name(self) -> Literal["numpy", "cupy"]:
        """Project to the temporary legacy solver backend string."""
        return self.execution.backend.value

    @property
    def sparse(self) -> bool:
        """Project storage to the temporary legacy sparse flag."""
        return self.execution.sparse

    @property
    def nondimensional(self) -> bool:
        """Project scaling to the temporary legacy nondimensional flag."""
        return self.scaling is ScalingMode.NONDIMENSIONAL

    @property
    def renorm(self) -> bool:
        """Project renormalization to the temporary legacy per-step flag."""
        return self.renormalization is RenormalizationPolicy.PER_STEP


__all__ = [
    "PropagationOptions",
    "RenormalizationPolicy",
    "ScalingMode",
]
