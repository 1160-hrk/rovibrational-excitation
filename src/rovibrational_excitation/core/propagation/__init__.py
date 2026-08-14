"""Typed quantum-state propagation facade with cycle-safe lazy exports."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import PropagatorBase as PropagatorBase
    from .direction import PropagationDirection as PropagationDirection
    from .factory import PropagatorFactory as PropagatorFactory
    from .liouville import LiouvillePropagator as LiouvillePropagator
    from .mixed_state import MixedStatePropagator as MixedStatePropagator
    from .options import (
        PropagationOptions as PropagationOptions,
    )
    from .options import (
        RenormalizationPolicy as RenormalizationPolicy,
    )
    from .options import (
        ScalingMode as ScalingMode,
    )
    from .problem import (
        Axis as Axis,
    )
    from .problem import (
        CouplingMode as CouplingMode,
    )
    from .problem import (
        CouplingSpec as CouplingSpec,
    )
    from .problem import (
        PropagationProblem as PropagationProblem,
    )
    from .problem import (
        PropagationState as PropagationState,
    )
    from .problem import (
        SystemModel as SystemModel,
    )
    from .result import PropagationResult as PropagationResult
    from .schrodinger import SchrodingerPropagator as SchrodingerPropagator

_EXPORTS = {
    "PropagatorBase": (".base", "PropagatorBase"),
    "PropagationDirection": (".direction", "PropagationDirection"),
    "SchrodingerPropagator": (".schrodinger", "SchrodingerPropagator"),
    "LiouvillePropagator": (".liouville", "LiouvillePropagator"),
    "MixedStatePropagator": (".mixed_state", "MixedStatePropagator"),
    "PropagatorFactory": (".factory", "PropagatorFactory"),
    "PropagationOptions": (".options", "PropagationOptions"),
    "RenormalizationPolicy": (".options", "RenormalizationPolicy"),
    "ScalingMode": (".options", "ScalingMode"),
    "Axis": (".problem", "Axis"),
    "CouplingMode": (".problem", "CouplingMode"),
    "CouplingSpec": (".problem", "CouplingSpec"),
    "PropagationProblem": (".problem", "PropagationProblem"),
    "PropagationState": (".problem", "PropagationState"),
    "SystemModel": (".problem", "SystemModel"),
    "PropagationResult": (".result", "PropagationResult"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load public names on first access without importing solvers eagerly."""
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__} has no attribute {name}") from None
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value
