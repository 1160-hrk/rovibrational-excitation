"""Versioned time-layout contract for the local optimizer.

This module deliberately does not use :class:`core.time.TimeGrid`.  The local
optimizer has a historically different endpoint and segment-index contract;
changing that contract changes its numerical calculation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


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


__all__ = ["LocalOptimizerLegacyGridV1"]
