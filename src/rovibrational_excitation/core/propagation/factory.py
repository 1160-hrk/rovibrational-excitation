"""Explicit typed propagator construction without algorithm heuristics."""

from __future__ import annotations

from typing import Any

from .base import PropagatorBase
from .capabilities import StatePath, validate_execution_capability
from .liouville import LiouvillePropagator
from .mixed_state import MixedStatePropagator
from .options import PropagationOptions
from .schrodinger import SchrodingerPropagator


class PropagatorFactory:
    """Construct a solver from required typed execution choices."""

    @staticmethod
    def create_propagator(
        *,
        state_path: StatePath,
        options: PropagationOptions,
        validate_units: bool = True,
    ) -> PropagatorBase[Any]:
        """Validate one explicit combination before constructing its solver."""
        if not isinstance(state_path, StatePath):
            raise TypeError("state_path must be a StatePath")
        if not isinstance(options, PropagationOptions):
            raise TypeError("options must be a PropagationOptions")

        validate_execution_capability(
            state_path=state_path,
            algorithm=options.algorithm,
            policy=options.execution,
        )
        algorithm_name = options.algorithm_name
        backend_name = options.backend_name

        if state_path is StatePath.PURE:
            return SchrodingerPropagator(
                backend=backend_name,
                algorithm=algorithm_name,
                validate_units=validate_units,
                renorm=options.renorm,
                sparse=options.sparse,
            )
        if state_path is StatePath.INCOHERENT_ENSEMBLE:
            return MixedStatePropagator(
                backend=backend_name,
                algorithm=algorithm_name,
                validate_units=validate_units,
                renorm=options.renorm,
                sparse=options.sparse,
            )
        if options.renorm:
            raise ValueError("renorm is not applicable to density-state propagation")
        return LiouvillePropagator(
            backend=backend_name,
            validate_units=validate_units,
        )
