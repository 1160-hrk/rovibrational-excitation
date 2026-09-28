"""Typed result boundary shared by all optimization algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from rovibrational_excitation.fields import ElectricField


class ControlLayout(str, Enum):
    """Exact time ownership of the returned two-component control array."""

    RK4_FIELD_SAMPLES = "rk4_field_samples"
    LOCAL_LEGACY_FIELD_SAMPLES = "local_legacy_field_samples"
    PIECEWISE_CONSTANT_INTERVALS = "piecewise_constant_intervals"


@dataclass(frozen=True, slots=True, eq=False)
class OptimizationResult:
    """One explicit optimizer result without grid or field reinterpretation.

    Arrays are the exact objects produced by the algorithm.  This boundary does
    not copy, normalize, resample, make read-only, or otherwise repair them.
    """

    trajectory_times_fs: np.ndarray
    trajectory: np.ndarray
    control_times_fs: np.ndarray
    controls_v_per_m: np.ndarray
    target_index: int | None
    metrics: dict[str, Any]
    control_layout: ControlLayout
    electric_field: ElectricField | None

    def __post_init__(self) -> None:
        if not isinstance(self.control_layout, ControlLayout):
            raise TypeError("control_layout must be a ControlLayout")
        if not isinstance(self.trajectory_times_fs, np.ndarray):
            raise TypeError("trajectory_times_fs must be a numpy.ndarray")
        if (
            self.trajectory_times_fs.ndim != 1
            or self.trajectory_times_fs.size < 1
            or not np.all(np.isfinite(self.trajectory_times_fs))
        ):
            raise ValueError(
                "trajectory_times_fs must be a nonempty finite one-dimensional array"
            )
        if not isinstance(self.trajectory, np.ndarray):
            raise TypeError("trajectory must be a numpy.ndarray")
        if (
            self.trajectory.ndim != 2
            or self.trajectory.shape[0] != self.trajectory_times_fs.size
            or self.trajectory.shape[1] < 1
            or not np.all(np.isfinite(self.trajectory))
        ):
            raise ValueError(
                "trajectory must be a finite two-dimensional array aligned with "
                "trajectory_times_fs"
            )
        if not isinstance(self.control_times_fs, np.ndarray):
            raise TypeError("control_times_fs must be a numpy.ndarray")
        if (
            self.control_times_fs.ndim != 1
            or self.control_times_fs.size < 1
            or not np.all(np.isfinite(self.control_times_fs))
        ):
            raise ValueError(
                "control_times_fs must be a nonempty finite one-dimensional array"
            )
        if not isinstance(self.controls_v_per_m, np.ndarray):
            raise TypeError("controls_v_per_m must be a numpy.ndarray")
        if (
            self.controls_v_per_m.shape != (self.control_times_fs.size, 2)
            or np.iscomplexobj(self.controls_v_per_m)
            or not np.all(np.isfinite(self.controls_v_per_m))
        ):
            raise ValueError(
                "controls_v_per_m must be a finite real array with shape "
                "(len(control_times_fs), 2)"
            )
        if self.target_index is not None and (
            isinstance(self.target_index, (bool, np.bool_))
            or not isinstance(self.target_index, (int, np.integer))
            or not 0 <= int(self.target_index) < self.trajectory.shape[1]
        ):
            raise ValueError(
                "target_index must be None or index the trajectory state dimension"
            )
        if not isinstance(self.metrics, dict):
            raise TypeError("metrics must be a dict")

        interval_layout = (
            self.control_layout is ControlLayout.PIECEWISE_CONSTANT_INTERVALS
        )
        if interval_layout and self.electric_field is not None:
            raise ValueError(
                "piecewise-constant interval controls do not have an ElectricField"
            )
        if not interval_layout and not isinstance(self.electric_field, ElectricField):
            raise ValueError("field-sample controls require an ElectricField")


__all__ = ["ControlLayout", "OptimizationResult"]
