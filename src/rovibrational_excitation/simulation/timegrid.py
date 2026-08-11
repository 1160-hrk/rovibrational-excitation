"""Time-grid construction for midpoint-sampled propagation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from rovibrational_excitation.core.time import TimeGrid


def build_time_grid(t_start: float, t_end: float, dt: float) -> NDArray[np.float64]:
    """Build an exact field grid whose endpoints are both propagated.

    ``dt`` is the electric-field sampling interval. One RK4 or split-operator
    propagation step spans two such intervals, because the algorithms consume
    the left endpoint, midpoint, and right endpoint.
    """
    return np.array(
        TimeGrid.from_bounds(t_start, t_end, dt).field_times_fs,
        copy=True,
    )
