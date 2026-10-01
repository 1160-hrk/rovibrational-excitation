"""Explicit initialization contracts for local control."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from rovibrational_excitation.optimization.local_initialization import (
    LocalNoInitialization,
    LocalSeedFieldInitialization,
    parse_local_initialization,
)


def test_seed_field_initialization_converts_direct_amplitude_once() -> None:
    initialization = parse_local_initialization(
        {
            "method": "seed_field",
            "amplitude": 0.001,
            "amplitude_units": "MV/m",
            "max_segments": 5,
        }
    )

    assert isinstance(initialization, LocalSeedFieldInitialization)
    assert initialization.amplitude == 0.001
    assert initialization.amplitude_units == "MV/m"
    assert initialization.amplitude_v_per_m == pytest.approx(1000.0)
    assert initialization.max_segments == 5


def test_none_initialization_has_no_hidden_seed_parameters() -> None:
    initialization = parse_local_initialization({"method": "none"})

    assert isinstance(initialization, LocalNoInitialization)


@pytest.mark.parametrize(
    ("value", "match"),
    [
        (None, "initialization must be a mapping"),
        ({}, "missing required local initialization option: method"),
        ({"method": "automatic"}, "method must be one of: none, seed_field"),
        (
            {"method": "none", "amplitude": 1.0},
            "not applicable when initialization.method='none': amplitude",
        ),
        (
            {
                "method": "seed_field",
                "amplitude": 1.0,
                "amplitude_units": "V/m",
            },
            "missing required seed-field initialization options: max_segments",
        ),
        (
            {
                "method": "seed_field",
                "amplitude": 1.0,
                "amplitude_units": "W/cm^2",
                "max_segments": 1,
            },
            "supported direct electric-field amplitude unit",
        ),
        (
            {
                "method": "seed_field",
                "amplitude": 0.0,
                "amplitude_units": "V/m",
                "max_segments": 1,
            },
            "amplitude must be positive",
        ),
        (
            {
                "method": "seed_field",
                "amplitude": 1.0,
                "amplitude_units": "V/m",
                "max_segments": 0,
            },
            "max_segments must be a positive integer",
        ),
    ],
)
def test_local_initialization_rejects_ambiguous_or_inert_input(
    value: Any,
    match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        parse_local_initialization(value)


@pytest.mark.parametrize("value", [True, 1.5, "2", np.int64(0)])
def test_seed_max_segments_is_an_exact_positive_integer(value: Any) -> None:
    with pytest.raises(ValueError, match="max_segments must be a positive integer"):
        parse_local_initialization(
            {
                "method": "seed_field",
                "amplitude": 1.0,
                "amplitude_units": "V/m",
                "max_segments": value,
            }
        )
