"""Contracts for the legacy time-grid construction boundary."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from rovibrational_excitation.core.electric_field import ElectricField
from rovibrational_excitation.core.time import (
    FIELD_INTERVALS_PER_PROPAGATION_STEP,
    TimeGrid,
)
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


def test_typed_grid_derives_propagation_timing_and_exact_endpoints() -> None:
    grid = TimeGrid.from_bounds(-1.0, 1.0, 0.1)

    assert grid.t_start_fs == -1.0
    assert grid.t_end_fs == 1.0
    assert grid.field_dt_fs == pytest.approx(0.1)
    assert grid.propagation_dt_fs == pytest.approx(0.2)
    assert grid.propagation_steps == 10
    assert not grid.field_times_fs.flags.writeable
    np.testing.assert_array_equal(
        grid.propagation_times_fs,
        grid.field_times_fs[::FIELD_INTERVALS_PER_PROPAGATION_STEP],
    )


def test_typed_grid_owns_a_read_only_copy() -> None:
    source = np.linspace(0.0, 0.4, 5)
    grid = TimeGrid(source)
    source[0] = -1.0

    assert grid.t_start_fs == 0.0
    with pytest.raises(ValueError, match="read-only"):
        grid.field_times_fs[0] = -1.0
    with pytest.raises(FrozenInstanceError):
        grid.field_dt_fs = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("times", "message"),
    [
        (np.array([0.0, 0.1]), "at least three"),
        (np.linspace(0.0, 0.3, 4), "odd number"),
        (np.array([0.0, 0.1, 0.21]), "uniformly spaced"),
        (np.array([0.0, 0.1, 0.1]), "strictly increasing"),
        (np.array([0.0, np.nan, 0.2]), "finite"),
        (np.array([0.0 + 0.0j, 0.1 + 0.0j, 0.2 + 0.0j]), "real-valued"),
        (np.array([False, True, True]), "real-valued"),
    ],
)
def test_typed_grid_rejects_invalid_sample_arrays(
    times: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        TimeGrid(times)


@pytest.mark.parametrize(
    "bounds",
    [
        (False, 1.0, 0.1),
        ([0.0], 1.0, 0.1),
        ("not-a-time", 1.0, 0.1),
    ],
)
def test_typed_grid_rejects_non_scalar_real_bounds(bounds: tuple) -> None:
    with pytest.raises(ValueError, match="finite real scalars"):
        TimeGrid.from_bounds(*bounds)


def test_electric_field_can_be_constructed_from_typed_grid() -> None:
    grid = TimeGrid.from_bounds(-0.2, 0.2, 0.05)

    field = ElectricField.from_time_grid(grid)

    np.testing.assert_array_equal(field.tlist, grid.field_times_fs)
    assert field.dt == grid.field_dt_fs
    assert field.dt_state == grid.propagation_dt_fs
    assert field.steps_state == grid.propagation_steps
