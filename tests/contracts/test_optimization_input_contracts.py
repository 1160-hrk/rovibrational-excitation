"""Input validation for optimization workflows."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from rovibrational_excitation.core.time import TimeGrid
from rovibrational_excitation.core.units import converter
from rovibrational_excitation.optimization.krotov import run_krotov_optimization
from rovibrational_excitation.optimization.krotov_initial_field import (
    KrotovGeneratedInitialField,
    KrotovSampledInitialField,
    parse_krotov_initial_field,
)


def _generated_params(**overrides: Any) -> dict[str, Any]:
    params: dict[str, Any] = {
        "initial_field_kind": "generated",
        "initial_duration": 200.0,
        "initial_duration_units": "fs",
        "initial_center": 250.0,
        "initial_center_units": "fs",
        "initial_carrier_frequency": 2300.0,
        "initial_carrier_frequency_units": "cm^-1",
        "initial_amplitude": 1.0e9,
        "initial_amplitude_units": "V/m",
        "initial_polarization": [1.0, 0.0],
    }
    params.update(overrides)
    return params


def test_krotov_requires_explicit_initial_field_kind_before_model_work():
    with pytest.raises(ValueError, match="initial_field_kind"):
        run_krotov_optimization(
            basis=None,
            hamiltonian=None,
            dipole=None,
            states={},
            time_cfg={},
            params={},
        )


def test_krotov_rejects_legacy_initial_field_keys_with_migration_hint() -> None:
    with pytest.raises(ValueError, match="legacy.*duration_initial.*value/unit"):
        parse_krotov_initial_field({"duration_initial": 200.0})


def test_krotov_generated_requires_primary_value_unit_pairs() -> None:
    params = _generated_params()
    del params["initial_center_units"]

    with pytest.raises(ValueError, match="initial_center_units"):
        parse_krotov_initial_field(params)


@pytest.mark.parametrize("key", ["initial_gdd", "initial_gdd_units"])
def test_krotov_generated_initial_field_requires_complete_dispersion_pair(
    key: str,
) -> None:
    params = _generated_params(**{key: 1.0 if key == "initial_gdd" else "fs^2"})

    with pytest.raises(ValueError, match="must be supplied together"):
        parse_krotov_initial_field(params)


def test_krotov_generated_initial_field_rejects_intensity_amplitude() -> None:
    with pytest.raises(ValueError, match="direct electric-field amplitude"):
        parse_krotov_initial_field(_generated_params(initial_amplitude_units="W/cm^2"))


def test_krotov_initial_field_rejects_unknown_and_inapplicable_branch_keys() -> None:
    with pytest.raises(ValueError, match="unsupported.*initial_duration_unit"):
        parse_krotov_initial_field(_generated_params(initial_duration_unit="fs"))
    with pytest.raises(ValueError, match="not applicable.*initial_field_samples"):
        parse_krotov_initial_field(
            _generated_params(initial_field_samples=np.zeros((3, 2)))
        )
    with pytest.raises(ValueError, match="not applicable.*initial_duration"):
        parse_krotov_initial_field(
            {
                "initial_field_kind": "sampled",
                "initial_field_samples": np.zeros((3, 2)),
                "initial_field_units": "V/m",
                "initial_duration": 1.0,
            }
        )


def test_krotov_generated_initial_field_is_cross_unit_equivalent() -> None:
    carrier_phz = converter.convert_frequency(2300.0, "cm^-1", "PHz")
    canonical = parse_krotov_initial_field(
        _generated_params(
            initial_gdd=100.0,
            initial_gdd_units="fs^2",
            initial_tod=20.0,
            initial_tod_units="fs^3",
        )
    )
    alternate = parse_krotov_initial_field(
        _generated_params(
            initial_duration=0.2,
            initial_duration_units="ps",
            initial_center=0.25,
            initial_center_units="ps",
            initial_carrier_frequency=carrier_phz,
            initial_carrier_frequency_units="PHz",
            initial_amplitude=10.0,
            initial_amplitude_units="MV/cm",
            initial_gdd=1.0e-4,
            initial_gdd_units="ps^2",
            initial_tod=2.0e-8,
            initial_tod_units="ps^3",
        )
    )
    assert isinstance(canonical, KrotovGeneratedInitialField)
    assert isinstance(alternate, KrotovGeneratedInitialField)
    grid = TimeGrid.from_bounds(0.0, 500.0, 0.5)
    np.testing.assert_allclose(
        canonical.samples_on(grid),
        alternate.samples_on(grid),
        rtol=2e-14,
        atol=1e-7,
    )


def test_krotov_sampled_initial_field_converts_once_and_copies_input() -> None:
    source = np.arange(10, dtype=float).reshape(5, 2)
    parsed = parse_krotov_initial_field(
        {
            "initial_field_kind": "sampled",
            "initial_field_samples": source,
            "initial_field_units": "MV/m",
        }
    )
    assert isinstance(parsed, KrotovSampledInitialField)
    source[:] = -1.0

    actual = parsed.samples_on(TimeGrid.from_bounds(0.0, 0.4, 0.1))
    np.testing.assert_array_equal(actual, np.arange(10).reshape(5, 2) * 1.0e6)
    assert actual.flags.writeable


@pytest.mark.parametrize(
    ("samples", "match"),
    [
        (np.zeros((5, 1)), "shape"),
        (np.zeros((5, 2), dtype=complex), "real-valued"),
        (np.full((5, 2), np.nan), "finite"),
    ],
)
def test_krotov_sampled_initial_field_rejects_invalid_values(
    samples: np.ndarray,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        parse_krotov_initial_field(
            {
                "initial_field_kind": "sampled",
                "initial_field_samples": samples,
                "initial_field_units": "V/m",
            }
        )


def test_krotov_sampled_initial_field_requires_exact_grid_length() -> None:
    parsed = parse_krotov_initial_field(
        {
            "initial_field_kind": "sampled",
            "initial_field_samples": np.zeros((4, 2)),
            "initial_field_units": "V/m",
        }
    )
    with pytest.raises(ValueError, match="length must exactly match time_grid"):
        parsed.samples_on(TimeGrid.from_bounds(0.0, 0.4, 0.1))


def test_krotov_sampled_initial_field_rejects_intensity_units() -> None:
    with pytest.raises(ValueError, match="direct electric-field amplitude"):
        parse_krotov_initial_field(
            {
                "initial_field_kind": "sampled",
                "initial_field_samples": np.zeros((5, 2)),
                "initial_field_units": "W/cm^2",
            }
        )
