"""Strict value contracts for optimizer options and spectral constraints."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from rovibrational_excitation.optimization.options import (
    validate_algorithm_options,
    validate_local_time_options,
)
from rovibrational_excitation.optimization.spectral_constraints import (
    solve_update_in_frequency,
)


def _local_options(**overrides: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "control_axes": "xy",
        "gain": 1.0,
        "gain_units": "(GV/m)^2 fs",
        "initialization": {
            "method": "seed_field",
            "amplitude": 1000.0,
            "amplitude_units": "V/m",
            "max_segments": 5,
        },
    }
    result.update(overrides)
    return result


@pytest.mark.parametrize("algorithm", ["local", "krotov", "grape"])
@pytest.mark.parametrize("axes", ["xx", "yy", "zz"])
def test_control_axes_must_be_distinct(algorithm: str, axes: str) -> None:
    params = (
        _local_options(control_axes=axes)
        if algorithm == "local"
        else {"control_axes": axes}
    )
    with pytest.raises(ValueError, match="distinct"):
        validate_algorithm_options(algorithm, params)


@pytest.mark.parametrize(
    ("params", "missing"),
    [
        ({"gain_units": "(GV/m)^2 fs"}, "gain"),
        ({"gain": 1.0}, "gain_units"),
    ],
)
def test_local_gain_value_and_unit_are_both_required(params, missing):
    with pytest.raises(ValueError, match=rf"missing required.*{missing}"):
        validate_algorithm_options(
            "local",
            {
                "control_axes": "xy",
                "initialization": {"method": "none"},
                **params,
            },
        )


@pytest.mark.parametrize(
    ("gain", "units", "match"),
    [
        (0.0, "(GV/m)^2 fs", "positive"),
        (-1.0, "(GV/m)^2 fs", "positive"),
        (True, "(GV/m)^2 fs", "finite scalar"),
        ("1.0", "(GV/m)^2 fs", "finite scalar"),
        (np.nan, "(GV/m)^2 fs", "finite"),
        (np.inf, "(GV/m)^2 fs", "finite"),
        (1.0e300, "(TV/m)^2 fs", "converted local control gain must be finite"),
        (1.0, "GV/m", "invalid local control gain unit"),
        (1.0, None, "unit must be a string"),
    ],
)
def test_local_gain_value_and_unit_are_strict(gain, units, match):
    with pytest.raises(ValueError, match=match):
        validate_algorithm_options(
            "local",
            {
                "control_axes": "xy",
                "gain": gain,
                "gain_units": units,
                "initialization": {"method": "none"},
            },
        )


def test_local_initialization_is_required() -> None:
    with pytest.raises(ValueError, match="missing required.*initialization"):
        validate_algorithm_options(
            "local",
            {
                "control_axes": "xy",
                "gain": 1.0,
                "gain_units": "(GV/m)^2 fs",
            },
        )


@pytest.mark.parametrize("value", [True, 1.5, "2"])
def test_iteration_count_is_an_exact_nonnegative_integer(value: Any) -> None:
    with pytest.raises(ValueError, match="max_iter.*nonnegative integer"):
        validate_algorithm_options(
            "grape",
            {"control_axes": "xy", "max_iter": value},
        )


@pytest.mark.parametrize("key", ["convergence_tol", "lambda_a", "learning_rate"])
@pytest.mark.parametrize("value", ["1e-3", np.nan, np.inf])
def test_iterative_real_options_do_not_coerce_or_accept_nonfinite_values(
    key: str,
    value: Any,
) -> None:
    with pytest.raises(ValueError, match=rf"{key}.*finite real number"):
        validate_algorithm_options(
            "grape",
            {"control_axes": "xy", key: value},
        )


@pytest.mark.parametrize("value", [-0.1, 1.1, np.nan, "0.5"])
def test_target_fidelity_is_a_finite_probability(value: Any) -> None:
    with pytest.raises(ValueError, match="target_fidelity.*between 0 and 1"):
        validate_algorithm_options(
            "grape",
            {"control_axes": "xy", "target_fidelity": value},
        )


@pytest.mark.parametrize("algorithm", ["local", "krotov", "grape"])
def test_propagator_override_must_be_callable_or_none(algorithm: str) -> None:
    params = (
        _local_options(propagator_func="rk4")
        if algorithm == "local"
        else {"control_axes": "xy", "propagator_func": "rk4"}
    )
    with pytest.raises(ValueError, match="propagator_func.*callable"):
        validate_algorithm_options(algorithm, params)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("use_sin2_shape", "false"),
        ("lookahead_enable", 0),
        ("normalize_weights", "true"),
        ("weight_reverse", 1),
        ("use_one_hot_target_in_weights", None),
    ],
)
def test_local_boolean_options_require_actual_booleans(key: str, value: Any) -> None:
    with pytest.raises(ValueError, match=rf"{key}.*bool"):
        validate_algorithm_options(
            "local",
            _local_options(**{key: value}),
        )


@pytest.mark.parametrize(
    "key",
    [
        "field_max_v_per_m",
        "segment_size_fs",
        "c_abs_min",
        "shape_floor",
        "lookahead_fraction",
        "weight_v_power",
        "weight_target_factor",
        "drive_abs_min",
    ],
)
@pytest.mark.parametrize("value", ["1.0", np.nan, np.inf])
def test_local_real_options_require_finite_numbers(key: str, value: Any) -> None:
    with pytest.raises(ValueError, match=rf"{key}.*finite real number"):
        validate_algorithm_options(
            "local",
            _local_options(**{key: value}),
        )


def test_only_local_segment_size_may_use_none_as_an_alternative_source() -> None:
    validate_algorithm_options(
        "local",
        _local_options(
            segment_size_steps=4,
            segment_size_fs=None,
        ),
    )
    with pytest.raises(ValueError, match="finite scalar"):
        validate_algorithm_options(
            "local",
            _local_options(gain=None),
        )


@pytest.mark.parametrize("value", ["WEIGHTS", "weight", "anything"])
def test_local_evaluation_mode_is_an_exact_enum(value: str) -> None:
    with pytest.raises(ValueError, match="eval_mode.*target, weights"):
        validate_algorithm_options(
            "local",
            _local_options(eval_mode=value),
        )


@pytest.mark.parametrize("value", ["unknown", "by_v_power_reverse"])
def test_local_weight_mode_has_no_fallback_or_reverse_suffix(value: str) -> None:
    with pytest.raises(ValueError, match="weight_mode.*by_v, by_v_power, custom"):
        validate_algorithm_options(
            "local",
            _local_options(weight_mode=value),
        )


@pytest.mark.parametrize("key", ["segment_size_steps"])
@pytest.mark.parametrize("value", [True, 1.5, "2"])
def test_local_integer_options_do_not_truncate_or_parse(key: str, value: Any) -> None:
    with pytest.raises(ValueError, match=rf"{key}.*integer"):
        validate_algorithm_options(
            "local",
            _local_options(**{key: value}),
        )


@pytest.mark.parametrize(
    ("time", "match"),
    [
        ({"total_fs": "1.0", "field_dt_fs": 0.1}, "total_fs.*finite"),
        ({"total_fs": 1.0, "field_dt_fs": np.nan}, "field_dt_fs.*finite"),
        ({"total_fs": 0.0, "field_dt_fs": 0.1}, "total_fs.*positive"),
        ({"total_fs": 1.0, "field_dt_fs": -0.1}, "field_dt_fs.*positive"),
        (
            {"total_fs": 1.0, "field_dt_fs": 0.1, "sample_stride": 1.5},
            "sample_stride.*positive integer",
        ),
    ],
)
def test_local_time_values_do_not_coerce_or_accept_invalid_values(
    time: dict[str, Any],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        validate_local_time_options(time)


def _spectrum(**overrides: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "method": "monotonic_kernel",
        "bands": [[2300.0, 100.0]],
        "units": "cm^-1",
        "mode": "pass",
        "combine": "max",
        "fwhm": True,
        "alpha_scale": 10.0,
    }
    result.update(overrides)
    return result


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"bands": []}, "nonempty"),
        ({"bands": [[2300.0, 0.0]]}, "width.*positive"),
        ({"bands": [[np.nan, 100.0]]}, "finite"),
        ({"units": "not-a-unit"}, "units"),
        ({"mode": "anything"}, "mode.*pass, stop"),
        ({"combine": "anything"}, "combine.*max, sum"),
        ({"fwhm": "true"}, "fwhm.*bool"),
        ({"alpha_scale": -1.0}, "alpha_scale.*nonnegative"),
        ({"alpha_scale": np.inf}, "alpha_scale.*finite"),
        ({"weights": [1.0]}, "weights.*combine='sum'"),
    ],
)
def test_spectrum_constraint_values_are_strict(
    overrides: dict[str, Any],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        validate_algorithm_options(
            "krotov",
            {
                "control_axes": "xy",
                "spectrum_constraints": _spectrum(**overrides),
            },
        )


@pytest.mark.parametrize(
    "weights",
    [[1.0, 2.0], [np.nan], [-1.0]],
)
def test_spectrum_sum_weights_match_bands_and_are_finite_nonnegative(
    weights: list[float],
) -> None:
    with pytest.raises(ValueError, match="weights"):
        validate_algorithm_options(
            "krotov",
            {
                "control_axes": "xy",
                "spectrum_constraints": _spectrum(
                    combine="sum",
                    weights=weights,
                ),
            },
        )


def test_spectral_update_matches_direct_nonnegative_denominator() -> None:
    source = np.array(
        [[0.0, 1.0], [1.0, -2.0], [0.5, 0.25], [-0.5, 3.0]],
        dtype=float,
    )
    alpha = np.array([0.0, 0.5, 2.0])
    expected = np.column_stack(
        [np.fft.irfft(np.fft.rfft(source[:, i]) / (1.0 + alpha), n=4) for i in range(2)]
    )

    actual = solve_update_in_frequency(source, alpha)

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "alpha",
    [
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, np.nan, 0.0]),
        np.zeros((3, 1)),
    ],
)
def test_spectral_update_rejects_invalid_alpha_instead_of_repairing(
    alpha: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="alpha_mask"):
        solve_update_in_frequency(np.zeros((4, 2)), alpha)
