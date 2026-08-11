"""Validated time grids for midpoint-sampled propagation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
from numpy.typing import NDArray

FIELD_INTERVALS_PER_PROPAGATION_STEP: Final = 2

_UNIFORM_GRID_RTOL: Final = 1e-10
_UNIFORM_GRID_ATOL_FS: Final = 1e-12

__all__ = ["FIELD_INTERVALS_PER_PROPAGATION_STEP", "TimeGrid"]


@dataclass(frozen=True, slots=True, eq=False)
class TimeGrid:
    """Immutable field-sampling grid and its derived propagation timing.

    The field grid contains the left endpoint, midpoint, and right endpoint
    needed by each RK4 or split-operator update. Consequently, one propagation
    step spans exactly two field-grid intervals.
    """

    field_times_fs: NDArray[np.float64]
    field_dt_fs: float = field(init=False)
    propagation_dt_fs: float = field(init=False)
    propagation_steps: int = field(init=False)

    def __post_init__(self) -> None:
        raw_times = np.asarray(self.field_times_fs)
        if np.iscomplexobj(raw_times) or np.issubdtype(raw_times.dtype, np.bool_):
            raise ValueError("field_times_fs must be real-valued")
        try:
            times = np.array(raw_times, dtype=np.float64, copy=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("field_times_fs must contain real numbers") from exc

        if times.ndim != 1:
            raise ValueError("field_times_fs must be one-dimensional")
        if times.size < 3:
            raise ValueError("field_times_fs must contain at least three samples")
        if times.size % FIELD_INTERVALS_PER_PROPAGATION_STEP == 0:
            raise ValueError(
                "field_times_fs must contain an odd number of samples "
                "(2 * propagation_steps + 1)"
            )
        if not np.all(np.isfinite(times)):
            raise ValueError("field_times_fs must contain only finite values")

        intervals = np.diff(times)
        if np.any(intervals <= 0.0):
            raise ValueError("field_times_fs must be strictly increasing")

        field_dt_fs = float(intervals[0])
        if not np.allclose(
            intervals,
            field_dt_fs,
            rtol=_UNIFORM_GRID_RTOL,
            atol=_UNIFORM_GRID_ATOL_FS,
        ):
            raise ValueError("field_times_fs must be uniformly spaced")

        propagation_steps = (times.size - 1) // FIELD_INTERVALS_PER_PROPAGATION_STEP
        times.setflags(write=False)

        object.__setattr__(self, "field_times_fs", times)
        object.__setattr__(self, "field_dt_fs", field_dt_fs)
        object.__setattr__(
            self,
            "propagation_dt_fs",
            FIELD_INTERVALS_PER_PROPAGATION_STEP * field_dt_fs,
        )
        object.__setattr__(self, "propagation_steps", propagation_steps)

    @classmethod
    def from_bounds(
        cls,
        t_start_fs: float,
        t_end_fs: float,
        field_dt_fs: float,
    ) -> TimeGrid:
        """Construct a grid without rounding or extending the requested span."""
        raw_values = (t_start_fs, t_end_fs, field_dt_fs)
        if any(isinstance(value, (bool, np.bool_)) for value in raw_values):
            raise ValueError("t_start, t_end, and dt must be finite real scalars")
        try:
            values = np.asarray(
                raw_values,
                dtype=np.float64,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "t_start, t_end, and dt must be finite real scalars"
            ) from exc
        if values.shape != (3,):
            raise ValueError("t_start, t_end, and dt must be finite real scalars")
        if not np.all(np.isfinite(values)):
            raise ValueError("t_start, t_end, and dt must be finite")

        start = float(values[0])
        end = float(values[1])
        dt = float(values[2])
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        span = end - start
        if span <= 0.0:
            raise ValueError("t_end must be greater than t_start")

        propagation_dt = FIELD_INTERVALS_PER_PROPAGATION_STEP * dt
        propagation_steps_float = span / propagation_dt
        propagation_steps = int(round(propagation_steps_float))
        if propagation_steps < 1 or not np.isclose(
            propagation_steps_float,
            propagation_steps,
            rtol=_UNIFORM_GRID_RTOL,
            atol=_UNIFORM_GRID_ATOL_FS,
        ):
            raise ValueError(
                "t_end - t_start must be an integer multiple of 2 * dt; "
                f"got span={span!r}, dt={dt!r}"
            )

        field_intervals = FIELD_INTERVALS_PER_PROPAGATION_STEP * propagation_steps
        return cls(
            np.linspace(
                start,
                end,
                field_intervals + 1,
                dtype=np.float64,
            )
        )

    @property
    def t_start_fs(self) -> float:
        """Configured first field time in femtoseconds."""
        return float(self.field_times_fs[0])

    @property
    def t_end_fs(self) -> float:
        """Configured final field time in femtoseconds."""
        return float(self.field_times_fs[-1])

    @property
    def propagation_times_fs(self) -> NDArray[np.float64]:
        """All propagation-state times, including both endpoints."""
        return self.field_times_fs[::FIELD_INTERVALS_PER_PROPAGATION_STEP]
