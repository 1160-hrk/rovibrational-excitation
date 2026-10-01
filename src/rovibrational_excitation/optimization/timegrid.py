"""Time-layout contracts for optimization algorithms.

GRAPE and Krotov use the canonical :class:`core.time.TimeGrid`.  The local
optimizer deliberately does not: it has a historically different endpoint and
segment-index contract, and changing that contract changes its calculation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rovibrational_excitation.core.time import TimeGrid


@dataclass(frozen=True, slots=True, eq=False)
class LocalOptimizerLegacyGridV1:
    """Expose the existing local-optimizer layout without rebuilding it.

    ``segments`` and ``tlist`` must be the objects returned by the existing
    ``_build_segments_and_tlist`` helper.  They are intentionally neither
    rounded, copied, nor made read-only here.

    The legacy RK4 kernel used ``(field_length - 1) // 2`` propagation steps.
    Consequently, it consumed every sample for odd field lengths and ignored
    only the final sample for even field lengths.  ``full_rk4_slice`` preserves
    exactly that rule while satisfying the current odd-length RK4 boundary.
    """

    segments: list[tuple[int, int]]
    tlist: NDArray[np.float64]

    @staticmethod
    def field_write_slice(start: int, end: int) -> slice:
        """Indices assigned to the newly calculated segment field."""
        return slice(start + 1, end + 1)

    @staticmethod
    def segment_propagation_slice(start: int, end: int) -> slice:
        """Inclusive endpoint view passed to each segment propagation."""
        return slice(start, end + 1)

    @staticmethod
    def segment_mid_index(start: int, end: int) -> int:
        """Existing integer midpoint used to sample the shape function."""
        return (start + end) // 2

    @property
    def full_rk4_slice(self) -> slice:
        """Prefix historically consumed by the floor-based legacy RK4 loop."""
        propagation_steps = (self.tlist.size - 1) // 2
        effective_length = 2 * propagation_steps + 1
        return slice(0, effective_length)

    @property
    def full_rk4_times_fs(self) -> NDArray[np.float64]:
        """Odd-length time prefix used only for the final RK4 propagation."""
        return self.tlist[self.full_rk4_slice]


@dataclass(frozen=True, slots=True)
class OptimizationTimeSettings:
    """Canonical GRAPE/Krotov grid plus output-only sampling policy."""

    grid: TimeGrid
    output_stride: int


def build_optimization_time_settings(
    time_cfg: Mapping[str, Any],
) -> OptimizationTimeSettings:
    """Build a strict GRAPE/Krotov time contract without implicit resampling."""
    if "dt_fs" in time_cfg:
        raise ValueError("dt_fs was removed; provide field_dt_fs")
    if "sample_stride" in time_cfg:
        raise ValueError(
            "sample_stride was removed; provide output_stride for output-only thinning"
        )

    allowed = {"total_fs", "field_dt_fs", "output_stride"}
    unknown = sorted(set(time_cfg) - allowed)
    if unknown:
        raise ValueError("unsupported optimization time options: " + ", ".join(unknown))
    missing = sorted({"total_fs", "field_dt_fs"} - set(time_cfg))
    if missing:
        raise ValueError(
            "missing required optimization time options: " + ", ".join(missing)
        )

    output_stride = time_cfg.get("output_stride", 1)
    if (
        isinstance(output_stride, (bool, np.bool_))
        or not isinstance(output_stride, (int, np.integer))
        or int(output_stride) < 1
    ):
        raise ValueError("output_stride must be a positive integer")

    grid = TimeGrid.from_bounds(
        0.0,
        time_cfg["total_fs"],
        time_cfg["field_dt_fs"],
    )
    return OptimizationTimeSettings(grid=grid, output_stride=int(output_stride))


def sample_optimization_output(
    times_fs: np.ndarray,
    trajectory: np.ndarray,
    *,
    output_stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Thin a complete optimizer trajectory while retaining its endpoint."""
    if (
        isinstance(output_stride, (bool, np.bool_))
        or not isinstance(output_stride, (int, np.integer))
        or int(output_stride) < 1
    ):
        raise ValueError("output_stride must be a positive integer")

    times = np.asarray(times_fs)
    states = np.asarray(trajectory)
    if times.ndim != 1 or times.size < 1:
        raise ValueError("times_fs must be a nonempty one-dimensional array")
    if states.ndim < 1 or states.shape[0] != times.size:
        raise ValueError("trajectory first dimension must match times_fs")

    indices = np.arange(0, times.size, int(output_stride), dtype=np.int64)
    if indices[-1] != times.size - 1:
        indices = np.append(indices, times.size - 1)
    return times[indices], states[indices]


__all__ = [
    "LocalOptimizerLegacyGridV1",
    "OptimizationTimeSettings",
    "build_optimization_time_settings",
    "sample_optimization_output",
]
