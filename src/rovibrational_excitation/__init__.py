"""Minimal typed public API for rovibrational excitation simulations.

The package root exposes only the stable v0.3 entry points. Model,
optimization, spectroscopy, advanced-field, and low-level core APIs remain
available from their explicit subpackages. Public objects are resolved lazily
so importing this package does not load workflow or optional dependencies.
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .core.execution import ExecutionPolicy as ExecutionPolicy
    from .core.time import TimeGrid as TimeGrid
    from .dynamics.options import PropagationOptions as PropagationOptions
    from .dynamics.problem import PropagationProblem as PropagationProblem
    from .dynamics.result import PropagationResult as PropagationResult
    from .fields.field import ElectricField as ElectricField
    from .simulation.runner import run_simulation_case as run_simulation_case

try:
    __version__: str = version(__name__)
except PackageNotFoundError:
    __version__ = "0.0.0+dev"

__author__ = "Hiroki Tsusaka"

_EXPORTS = {
    "ElectricField": (".fields.field", "ElectricField"),
    "TimeGrid": (".core.time", "TimeGrid"),
    "ExecutionPolicy": (".core.execution", "ExecutionPolicy"),
    "PropagationProblem": (".dynamics.problem", "PropagationProblem"),
    "PropagationOptions": (".dynamics.options", "PropagationOptions"),
    "PropagationResult": (".dynamics.result", "PropagationResult"),
    "run_simulation_case": (".simulation.runner", "run_simulation_case"),
}

__all__ = [
    "__version__",
    "ElectricField",
    "TimeGrid",
    "ExecutionPolicy",
    "PropagationProblem",
    "PropagationOptions",
    "PropagationResult",
    "run_simulation_case",
]


def __getattr__(name: str) -> Any:
    """Resolve one supported root export without eager subpackage imports."""
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy public names in interactive discovery."""
    return sorted(set(globals()) | set(__all__))
