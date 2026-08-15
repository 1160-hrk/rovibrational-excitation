"""Pre-allocation capability checks for explicit execution policies."""

from __future__ import annotations

from enum import Enum

from ..core.execution import ArrayBackend, ExecutionPolicy, MatrixStorage
from .utils import HAS_CUPY


class StatePath(str, Enum):
    """Numerical state path selected before solver construction."""

    PURE = "pure"
    INCOHERENT_ENSEMBLE = "incoherent_ensemble"
    DENSITY = "density"


class PropagationAlgorithm(str, Enum):
    """Implemented propagation algorithms."""

    RK4 = "rk4"
    SPLIT_OPERATOR = "split_operator"


class UnsupportedExecutionPolicyError(ValueError):
    """A requested state/algorithm/backend/storage combination is unsupported."""


def validate_execution_capability(
    *,
    state_path: StatePath,
    algorithm: PropagationAlgorithm,
    policy: ExecutionPolicy,
    cupy_available: bool | None = None,
) -> None:
    """Reject unsupported combinations before operator conversion/allocation."""
    if not isinstance(state_path, StatePath):
        raise TypeError("state_path must be a StatePath")
    if not isinstance(algorithm, PropagationAlgorithm):
        raise TypeError("algorithm must be a PropagationAlgorithm")
    if not isinstance(policy, ExecutionPolicy):
        raise TypeError("policy must be an ExecutionPolicy")

    if policy.backend is ArrayBackend.CUPY and policy.storage is MatrixStorage.CSR:
        raise UnsupportedExecutionPolicyError("CuPy CSR propagation is not supported")

    if state_path is StatePath.DENSITY:
        if algorithm is not PropagationAlgorithm.RK4:
            raise UnsupportedExecutionPolicyError(
                "density-state propagation supports only algorithm='rk4'"
            )
        if policy.backend is not ArrayBackend.NUMPY:
            raise UnsupportedExecutionPolicyError(
                "density-state propagation supports only backend='numpy'"
            )
        if policy.storage is not MatrixStorage.DENSE:
            raise UnsupportedExecutionPolicyError(
                "density-state propagation supports only dense storage"
            )

    available = HAS_CUPY if cupy_available is None else cupy_available
    if policy.backend is ArrayBackend.CUPY and not available:
        raise RuntimeError("CuPy backend requested but CuPy not installed")


__all__ = [
    "PropagationAlgorithm",
    "StatePath",
    "UnsupportedExecutionPolicyError",
    "validate_execution_capability",
]
