"""
Optimization algorithms registry and helpers.
"""

from .local import run_local_optimization
from .krotov import run_krotov_optimization
from .grape import run_grape_optimization

ALGO_REGISTRY = {
    "local": run_local_optimization,
    "krotov": run_krotov_optimization,
    "grape": run_grape_optimization,
}

__all__ = [
    "run_local_optimization",
    "run_krotov_optimization",
    "run_grape_optimization",
    "ALGO_REGISTRY",
]

