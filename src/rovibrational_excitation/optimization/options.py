"""Strict structural option schemas for optimization algorithms."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from .krotov_initial_field import KROTOV_INITIAL_FIELD_OPTION_KEYS

OptimizationAlgorithm = Literal["local", "krotov", "grape"]

_COMMON_ITERATIVE_KEYS = {
    "max_iter",
    "convergence_tol",
    "lambda_a",
    "target_fidelity",
    "control_axes",
    "propagator_func",
}

GRAPE_OPTION_KEYS = frozenset(_COMMON_ITERATIVE_KEYS | {"learning_rate"})
KROTOV_OPTION_KEYS = frozenset(
    _COMMON_ITERATIVE_KEYS
    | set(KROTOV_INITIAL_FIELD_OPTION_KEYS)
    | {"spectrum_constraints"}
)
LOCAL_OPTION_KEYS = frozenset(
    {
        "control_axes",
        "gain",
        "field_max_v_per_m",
        "use_sin2_shape",
        "segment_size_steps",
        "segment_size_fs",
        "seed_amplitude_v_per_m",
        "seed_max_segments",
        "c_abs_min",
        "shape_floor",
        "lookahead_enable",
        "lookahead_fraction",
        "eval_mode",
        "weight_mode",
        "weight_v_power",
        "weight_target_factor",
        "normalize_weights",
        "weight_reverse",
        "use_one_hot_target_in_weights",
        "drive_abs_min",
        "custom_weights",
        "custom_weights_dict",
        "propagator_func",
    }
)

_OPTION_KEYS = {
    "grape": GRAPE_OPTION_KEYS,
    "krotov": KROTOV_OPTION_KEYS,
    "local": LOCAL_OPTION_KEYS,
}
_REMOVED_LOCAL_KEYS = {
    "field_max": "field_max_v_per_m",
    "seed_amplitude": "seed_amplitude_v_per_m",
}
_SPECTRUM_REQUIRED = {
    "method",
    "bands",
    "units",
    "mode",
    "combine",
    "fwhm",
    "alpha_scale",
}
_SPECTRUM_OPTIONAL = {"weights"}


def _names(values: set[Any]) -> str:
    return ", ".join(
        sorted(value if isinstance(value, str) else repr(value) for value in values)
    )


def _validate_control_axes(algorithm: str, value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError(f"algorithms.{algorithm}.control_axes must be a string")
    if len(value) != 2 or any(axis not in "xyz" for axis in value):
        raise ValueError(
            f"algorithms.{algorithm}.control_axes must contain exactly two "
            "lowercase axes from x, y, z"
        )
    if algorithm == "grape" and value != "xy":
        raise ValueError(
            "GRAPE currently implements only control_axes='xy'; other axes "
            "must not be accepted and ignored"
        )
    return value


def _validate_spectrum_constraints(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("spectrum_constraints must be a mapping")
    keys = set(value)
    unknown = keys - (_SPECTRUM_REQUIRED | _SPECTRUM_OPTIONAL)
    if unknown:
        raise ValueError("unsupported spectrum_constraints options: " + _names(unknown))
    missing = _SPECTRUM_REQUIRED - keys
    if missing:
        raise ValueError(
            "missing required spectrum_constraints options: " + _names(missing)
        )
    if value["method"] != "monotonic_kernel":
        raise ValueError("spectrum_constraints.method must be 'monotonic_kernel'")


def validate_algorithm_options(
    algorithm: OptimizationAlgorithm,
    params: Mapping[str, Any],
) -> str:
    """Validate structural options and return the required control-axis pair."""
    if not isinstance(params, Mapping):
        raise TypeError(f"algorithms.{algorithm} must be a mapping")
    if algorithm == "local":
        removed = set(params) & set(_REMOVED_LOCAL_KEYS)
        if removed:
            key = sorted(removed)[0]
            raise ValueError(
                f"{key} was removed; provide {_REMOVED_LOCAL_KEYS[key]} in V/m"
            )
    unknown = set(params) - set(_OPTION_KEYS[algorithm])
    if unknown:
        raise ValueError(
            f"unsupported {algorithm} optimization options: " + _names(unknown)
        )
    if "control_axes" not in params:
        raise ValueError(
            f"missing required {algorithm} optimization option: control_axes"
        )
    axes = _validate_control_axes(algorithm, params["control_axes"])
    if algorithm == "krotov" and "spectrum_constraints" in params:
        _validate_spectrum_constraints(params["spectrum_constraints"])
    return axes


def validate_local_time_options(time_cfg: Mapping[str, Any]) -> None:
    """Reject local time typos without rebuilding its frozen legacy grid."""
    allowed = {"total_fs", "field_dt_fs", "sample_stride"}
    unknown = set(time_cfg) - allowed
    if unknown:
        raise ValueError(
            "unsupported local optimization time options: " + _names(unknown)
        )


__all__ = [
    "GRAPE_OPTION_KEYS",
    "KROTOV_OPTION_KEYS",
    "LOCAL_OPTION_KEYS",
    "OptimizationAlgorithm",
    "validate_algorithm_options",
    "validate_local_time_options",
]
