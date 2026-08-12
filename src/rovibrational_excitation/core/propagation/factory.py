"""Explicit typed propagator construction without algorithm heuristics."""

from __future__ import annotations

from typing import Any, Literal

from ..execution import ExecutionPolicy
from .base import PropagatorBase
from .capabilities import (
    PropagationAlgorithm,
    StatePath,
    validate_execution_capability,
)
from .liouville import LiouvillePropagator
from .mixed_state import MixedStatePropagator
from .schrodinger import SchrodingerPropagator


class PropagatorFactory:
    """Construct a solver from required typed execution choices."""

    @staticmethod
    def create_propagator(
        *,
        state_path: StatePath,
        algorithm: PropagationAlgorithm,
        execution_policy: ExecutionPolicy,
        renorm: bool,
        validate_units: bool = True,
    ) -> PropagatorBase[Any]:
        """Validate one explicit combination before constructing its solver."""
        if not isinstance(state_path, StatePath):
            raise TypeError("state_path must be a StatePath")
        if not isinstance(algorithm, PropagationAlgorithm):
            raise TypeError("algorithm must be a PropagationAlgorithm")
        if not isinstance(execution_policy, ExecutionPolicy):
            raise TypeError("execution_policy must be an ExecutionPolicy")
        if not isinstance(renorm, bool):
            raise TypeError("renorm must be a bool")

        validate_execution_capability(
            state_path=state_path,
            algorithm=algorithm,
            policy=execution_policy,
        )
        algorithm_name: Literal["rk4", "split_operator"] = algorithm.value
        backend_name: Literal["numpy", "cupy"] = execution_policy.backend.value

        if state_path is StatePath.PURE:
            return SchrodingerPropagator(
                backend=backend_name,
                algorithm=algorithm_name,
                validate_units=validate_units,
                renorm=renorm,
                sparse=execution_policy.sparse,
            )
        if state_path is StatePath.INCOHERENT_ENSEMBLE:
            return MixedStatePropagator(
                backend=backend_name,
                algorithm=algorithm_name,
                validate_units=validate_units,
                renorm=renorm,
                sparse=execution_policy.sparse,
            )
        if renorm:
            raise ValueError("renorm is not applicable to density-state propagation")
        return LiouvillePropagator(
            backend=backend_name,
            validate_units=validate_units,
        )
