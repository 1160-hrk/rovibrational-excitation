"""Time-grid construction for midpoint-sampled propagation."""

from __future__ import annotations

import numpy as np

from rovibrational_excitation.core.propagation.utils import (
    FIELD_INTERVALS_PER_PROPAGATION_STEP,
)


def build_time_grid(t_start: float, t_end: float, dt: float) -> np.ndarray:
    """Build an exact field grid whose endpoints are both propagated.

    ``dt`` is the electric-field sampling interval. One RK4 or split-operator
    propagation step spans two such intervals, because the algorithms consume
    the left endpoint, midpoint, and right endpoint.
    """
    values = np.asarray([t_start, t_end, dt], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("t_start, t_end, and dt must be finite")
    if dt <= 0:
        raise ValueError("dt must be positive")
    span = t_end - t_start
    if span <= 0:
        raise ValueError("t_end must be greater than t_start")

    propagation_dt = FIELD_INTERVALS_PER_PROPAGATION_STEP * dt
    propagation_steps_float = span / propagation_dt
    propagation_steps = int(round(propagation_steps_float))
    if propagation_steps < 1 or not np.isclose(
        propagation_steps_float,
        propagation_steps,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError(
            "t_end - t_start must be an integer multiple of 2 * dt; "
            f"got span={span!r}, dt={dt!r}"
        )

    field_intervals = FIELD_INTERVALS_PER_PROPAGATION_STEP * propagation_steps
    return np.linspace(t_start, t_end, field_intervals + 1, dtype=float)
