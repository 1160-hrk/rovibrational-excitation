"""Exact time and endpoint contracts for the local optimizer."""

from __future__ import annotations

import numpy as np
import pytest

from rovibrational_excitation.optimization.local import _build_segments_and_tlist
from rovibrational_excitation.optimization.timegrid import (
    LocalOptimizerLegacyGridV1,
)


@pytest.mark.parametrize(
    (
        "time_total",
        "expected_segments",
        "expected_tlist",
        "expected_full_rk4_stop",
    ),
    [
        (
            0.4,
            [(0, 4)],
            np.arange(0.0, 0.7, 0.1),
            7,
        ),
        (
            0.8,
            [(0, 4), (4, 8)],
            np.arange(0.0, 1.0, 0.1),
            9,
        ),
    ],
)
def test_legacy_local_grid_preserves_arange_tail_and_rk4_prefix(
    time_total: float,
    expected_segments: list[tuple[int, int]],
    expected_tlist: np.ndarray,
    expected_full_rk4_stop: int,
) -> None:
    segments, tlist = _build_segments_and_tlist(
        time_total,
        0.1,
        None,
        0.5,
    )
    grid = LocalOptimizerLegacyGridV1(segments=segments, tlist=tlist)

    assert segments == expected_segments
    np.testing.assert_array_equal(tlist, expected_tlist)
    assert grid.full_rk4_slice == slice(0, expected_full_rk4_stop)
    assert grid.full_rk4_times_fs.size % 2 == 1
    np.testing.assert_array_equal(
        grid.full_rk4_times_fs,
        tlist[:expected_full_rk4_stop],
    )


def test_legacy_local_grid_preserves_segment_endpoint_ownership() -> None:
    segments, tlist = _build_segments_and_tlist(0.8, 0.1, None, 0.5)
    grid = LocalOptimizerLegacyGridV1(segments=segments, tlist=tlist)
    field = np.zeros(tlist.size)

    first, second = segments
    field[grid.field_write_slice(*first)] = 1.0
    field[grid.field_write_slice(*second)] = 2.0

    assert grid.segment_propagation_slice(*first) == slice(0, 5)
    assert grid.segment_propagation_slice(*second) == slice(4, 9)
    np.testing.assert_array_equal(
        field,
        np.array([0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 0.0]),
    )
    assert field[first[1]] == 1.0
    assert field[second[0]] == 1.0


def test_legacy_local_grid_preserves_step_precedence_and_even_flooring() -> None:
    segments, tlist = _build_segments_and_tlist(0.6, 0.1, 5, 0.2)
    grid = LocalOptimizerLegacyGridV1(segments=segments, tlist=tlist)

    assert segments == [(0, 4), (4, 8)]
    assert grid.field_write_slice(0, 4) == slice(1, 5)
    assert grid.segment_mid_index(0, 4) == 2
