"""Contracts for the legacy time-grid construction boundary."""

import numpy as np
import pytest

from rovibrational_excitation.simulation.timegrid import build_time_grid


@pytest.mark.parametrize(
    ("t_start", "t_end", "field_dt", "propagation_steps"),
    [
        (0.0, 0.6, 0.1, 3),
        (-0.3, 0.3, 0.05, 6),
        (1.0, 1.5, 0.05, 5),
    ],
)
def test_legacy_builder_returns_exact_uniform_midpoint_grid(
    t_start: float,
    t_end: float,
    field_dt: float,
    propagation_steps: int,
) -> None:
    grid = build_time_grid(t_start, t_end, field_dt)
    expected = np.linspace(
        t_start,
        t_end,
        2 * propagation_steps + 1,
        dtype=float,
    )

    np.testing.assert_array_equal(grid, expected)
    assert grid.dtype == np.dtype(float)
    assert grid.flags.writeable
    assert grid.size == 2 * propagation_steps + 1


@pytest.mark.parametrize(
    ("t_start", "t_end", "field_dt"),
    [
        (np.nan, 1.0, 0.1),
        (0.0, np.inf, 0.1),
        (0.0, 1.0, np.nan),
    ],
)
def test_legacy_builder_rejects_nonfinite_values(
    t_start: float,
    t_end: float,
    field_dt: float,
) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        build_time_grid(t_start, t_end, field_dt)


@pytest.mark.parametrize("field_dt", [0.0, -0.1])
def test_legacy_builder_rejects_nonpositive_field_dt(field_dt: float) -> None:
    with pytest.raises(ValueError, match="dt must be positive"):
        build_time_grid(0.0, 1.0, field_dt)


@pytest.mark.parametrize("t_end", [0.0, -1.0])
def test_legacy_builder_rejects_nonpositive_span(t_end: float) -> None:
    with pytest.raises(ValueError, match="t_end must be greater than t_start"):
        build_time_grid(0.0, t_end, 0.1)
