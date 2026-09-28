"""Optimization algorithm registry and helpers."""

from .grape import run_grape_optimization
from .krotov import run_krotov_optimization
from .legacy_batch_overlap import run_legacy_batch_overlap_optimization
from .local import run_local_optimization
from .objective import (
    DiscreteL2TargetObjective,
    IndexedTargetPopulation,
    TargetPopulationEvaluation,
    TargetPopulationEvaluator,
    VectorTargetPopulation,
)
from .result import ControlLayout, OptimizationResult

ALGO_REGISTRY = {
    "local": run_local_optimization,
    "krotov": run_krotov_optimization,
    "legacy_batch_overlap": run_legacy_batch_overlap_optimization,
    "grape": run_grape_optimization,
}

__all__ = [
    "run_local_optimization",
    "run_krotov_optimization",
    "run_legacy_batch_overlap_optimization",
    "run_grape_optimization",
    "ALGO_REGISTRY",
    "ControlLayout",
    "OptimizationResult",
    "DiscreteL2TargetObjective",
    "IndexedTargetPopulation",
    "TargetPopulationEvaluation",
    "TargetPopulationEvaluator",
    "VectorTargetPopulation",
]
