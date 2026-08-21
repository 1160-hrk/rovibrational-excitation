"""Typed time and output-sampling contracts for GRAPE and Krotov."""

from __future__ import annotations

import numpy as np
import pytest

from rovibrational_excitation.optimization.timegrid import (
    build_optimization_time_settings,
    sample_optimization_output,
)


def test_optimization_time_settings_use_explicit_field_spacing() -> None:
    settings = build_optimization_time_settings(
        {"total_fs": 0.8, "field_dt_fs": 0.1, "output_stride": 3}
    )

    np.testing.assert_array_equal(
        settings.grid.field_times_fs,
        np.linspace(0.0, 0.8, 9),
    )
    assert settings.grid.field_dt_fs == pytest.approx(0.1)
    assert settings.grid.propagation_dt_fs == pytest.approx(0.2)
    assert settings.output_stride == 3


@pytest.mark.parametrize(
    ("time_cfg", "message"),
    [
        (
            {"total_fs": 0.8, "dt_fs": 0.2},
            "dt_fs was removed.*field_dt_fs",
        ),
        (
            {"total_fs": 0.8, "field_dt_fs": 0.1, "sample_stride": 1},
            "sample_stride was removed.*output_stride",
        ),
        (
            {"total_fs": 0.8, "field_dt_fs": 0.1, "mystery": 1},
            "unsupported optimization time options: mystery",
        ),
        (
            {"total_fs": 0.7, "field_dt_fs": 0.1},
            r"integer multiple of 2 \* dt",
        ),
        (
            {"total_fs": 0.8, "field_dt_fs": 0.1, "output_stride": 0},
            "output_stride must be a positive integer",
        ),
        (
            {"total_fs": 0.8, "field_dt_fs": 0.1, "output_stride": True},
            "output_stride must be a positive integer",
        ),
    ],
)
def test_optimization_time_settings_reject_implicit_changes(
    time_cfg: dict,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_optimization_time_settings(time_cfg)


def test_optimization_plotting_uses_explicit_result_times_without_examples_dependency() -> (
    None
):
    from inspect import signature

    from rovibrational_excitation.visualization.plot_all import plot_all

    assert "trajectory_times_fs" in signature(plot_all).parameters


def test_output_sampling_appends_endpoint_without_changing_internal_steps() -> None:
    times = np.arange(5, dtype=float) * 0.2
    trajectory = np.column_stack((np.arange(5), -np.arange(5)))

    sampled_times, sampled_trajectory = sample_optimization_output(
        times,
        trajectory,
        output_stride=3,
    )

    np.testing.assert_array_equal(sampled_times, times[[0, 3, 4]])
    np.testing.assert_array_equal(sampled_trajectory, trajectory[[0, 3, 4]])
