"""Strict piecewise-constant interval grid for standard Krotov control."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True, eq=False)
class KrotovIntervalGrid:
    """State endpoints and interval-midpoint controls in femtoseconds."""

    state_times_fs: NDArray[np.float64]
    control_times_fs: NDArray[np.float64]
    control_dt_fs: float
    interval_count: int
    output_stride: int

    @classmethod
    def from_config(cls, time_cfg: Mapping[str, Any]) -> KrotovIntervalGrid:
        if "field_dt_fs" in time_cfg:
            raise ValueError(
                "standard Krotov uses piecewise-constant interval controls; "
                "provide control_dt_fs, not field_dt_fs"
            )
        if "sample_stride" in time_cfg:
            raise ValueError(
                "sample_stride was removed; provide output_stride for output-only thinning"
            )
        allowed = {"total_fs", "control_dt_fs", "output_stride"}
        unknown = sorted(set(time_cfg) - allowed)
        if unknown:
            raise ValueError(
                "unsupported standard Krotov time options: " + ", ".join(unknown)
            )
        missing = sorted({"total_fs", "control_dt_fs"} - set(time_cfg))
        if missing:
            raise ValueError(
                "missing required standard Krotov time options: " + ", ".join(missing)
            )

        total = _positive_real(time_cfg["total_fs"], label="total_fs")
        dt = _positive_real(time_cfg["control_dt_fs"], label="control_dt_fs")
        ratio = total / dt
        interval_count = int(round(ratio))
        tolerance = 64.0 * np.finfo(float).eps * max(1.0, abs(ratio))
        if interval_count < 1 or abs(ratio - interval_count) > tolerance:
            raise ValueError(
                "total_fs must be exactly divisible by control_dt_fs for standard Krotov"
            )

        raw_stride = time_cfg.get("output_stride", 1)
        if (
            isinstance(raw_stride, (bool, np.bool_))
            or not isinstance(raw_stride, Integral)
            or int(raw_stride) < 1
        ):
            raise ValueError("output_stride must be a positive integer")

        state_times = np.linspace(0.0, total, interval_count + 1, dtype=np.float64)
        control_times = (state_times[:-1] + state_times[1:]) * 0.5
        state_times.setflags(write=False)
        control_times.setflags(write=False)
        return cls(
            state_times_fs=state_times,
            control_times_fs=control_times,
            control_dt_fs=dt,
            interval_count=interval_count,
            output_stride=int(raw_stride),
        )


def _positive_real(value: Any, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a positive finite real number")
    converted = float(value)
    if not np.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{label} must be a positive finite real number")
    return converted


__all__ = ["KrotovIntervalGrid"]
