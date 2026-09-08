"""Strict structural option schemas for optimization algorithms."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from numbers import Integral, Real
from typing import Any, Literal

import numpy as np

from rovibrational_excitation.core.units import LocalControlGain, converter

from .krotov_initial_field import KROTOV_INITIAL_FIELD_OPTION_KEYS
from .local_initialization import parse_local_initialization

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
        "gain_units",
        "field_max_v_per_m",
        "use_sin2_shape",
        "segment_size_steps",
        "segment_size_fs",
        "initialization",
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
    "field_max": "provide field_max_v_per_m in V/m",
    "seed_amplitude": (
        "provide initialization.amplitude and initialization.amplitude_units"
    ),
    "seed_amplitude_v_per_m": (
        "provide initialization.amplitude and initialization.amplitude_units"
    ),
    "seed_max_segments": "provide initialization.max_segments",
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
_LOCAL_BOOL_KEYS = {
    "use_sin2_shape",
    "lookahead_enable",
    "normalize_weights",
    "weight_reverse",
    "use_one_hot_target_in_weights",
}
_LOCAL_REAL_KEYS = {
    "field_max_v_per_m",
    "segment_size_fs",
    "c_abs_min",
    "shape_floor",
    "lookahead_fraction",
    "weight_v_power",
    "weight_target_factor",
    "drive_abs_min",
}


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
    if value[0] == value[1]:
        raise ValueError(
            f"algorithms.{algorithm}.control_axes must contain two distinct axes"
        )
    if algorithm == "grape" and value != "xy":
        raise ValueError(
            "GRAPE currently implements only control_axes='xy'; other axes "
            "must not be accepted and ignored"
        )
    return value


def _finite_real(value: Any, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite real number")
    converted = float(value)
    if not isfinite(converted):
        raise ValueError(f"{label} must be a finite real number")
    return converted


def _integer(value: Any, *, label: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        qualifier = "nonnegative" if minimum == 0 else "positive"
        raise ValueError(f"{label} must be a {qualifier} integer")
    converted = int(value)
    if converted < minimum:
        qualifier = "nonnegative" if minimum == 0 else "positive"
        raise ValueError(f"{label} must be a {qualifier} integer")
    return converted


def _validate_common_options(params: Mapping[str, Any]) -> None:
    if "max_iter" in params:
        _integer(params["max_iter"], label="max_iter", minimum=0)
    for key in ("convergence_tol", "lambda_a"):
        if key in params:
            _finite_real(params[key], label=key)
    if "target_fidelity" in params:
        value = params["target_fidelity"]
        try:
            fidelity = _finite_real(value, label="target_fidelity")
        except ValueError as exc:
            raise ValueError(
                "target_fidelity must be a finite real number between 0 and 1"
            ) from exc
        if not 0.0 <= fidelity <= 1.0:
            raise ValueError(
                "target_fidelity must be a finite real number between 0 and 1"
            )
    if "propagator_func" in params:
        value = params["propagator_func"]
        if value is not None and not callable(value):
            raise ValueError("propagator_func must be callable or None")


def _sequence(value: Any, *, label: str) -> list[Any]:
    if isinstance(value, (str, bytes, Mapping)):
        raise ValueError(f"{label} must be a sequence")
    try:
        return list(value)
    except TypeError as exc:
        raise ValueError(f"{label} must be a sequence") from exc


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

    bands = _sequence(value["bands"], label="spectrum_constraints.bands")
    if not bands:
        raise ValueError("spectrum_constraints.bands must be nonempty")
    units = value["units"]
    if not isinstance(units, str):
        raise ValueError(
            "spectrum_constraints.units must be a supported frequency unit"
        )
    try:
        converter.convert_frequency(1.0, units, "PHz")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "spectrum_constraints.units must be a supported frequency unit"
        ) from exc
    for index, raw_band in enumerate(bands):
        band = _sequence(
            raw_band,
            label=f"spectrum_constraints.bands[{index}]",
        )
        if len(band) != 2:
            raise ValueError(
                f"spectrum_constraints.bands[{index}] must be [center, width]"
            )
        center = _finite_real(
            band[0],
            label=f"spectrum_constraints.bands[{index}] center",
        )
        width = _finite_real(
            band[1],
            label=f"spectrum_constraints.bands[{index}] width",
        )
        try:
            converter.convert_frequency(center, units, "PHz")
            converted_width = float(converter.convert_frequency(width, units, "PHz"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid spectrum_constraints.bands[{index}] frequency"
            ) from exc
        if converted_width <= 0.0:
            raise ValueError(
                f"spectrum_constraints.bands[{index}] width must be positive"
            )

    mode = value["mode"]
    if not isinstance(mode, str) or mode not in {"pass", "stop"}:
        raise ValueError("spectrum_constraints.mode must be one of: pass, stop")
    combine = value["combine"]
    if not isinstance(combine, str) or combine not in {"max", "sum"}:
        raise ValueError("spectrum_constraints.combine must be one of: max, sum")
    if not isinstance(value["fwhm"], bool):
        raise ValueError("spectrum_constraints.fwhm must be a bool")
    alpha_scale = _finite_real(
        value["alpha_scale"],
        label="spectrum_constraints.alpha_scale",
    )
    if alpha_scale < 0.0:
        raise ValueError("spectrum_constraints.alpha_scale must be nonnegative")

    if "weights" in value:
        if combine != "sum":
            raise ValueError(
                "spectrum_constraints.weights is accepted only with combine='sum'"
            )
        if value["weights"] is not None:
            weights = _sequence(
                value["weights"],
                label="spectrum_constraints.weights",
            )
            if len(weights) != len(bands):
                raise ValueError("spectrum_constraints.weights length must match bands")
            for raw_weight in weights:
                weight = _finite_real(
                    raw_weight,
                    label="spectrum_constraints.weights entries",
                )
                if weight < 0.0:
                    raise ValueError(
                        "spectrum_constraints.weights entries must be nonnegative"
                    )


def _validate_local_options(params: Mapping[str, Any]) -> None:
    missing = {"gain", "gain_units", "initialization"} - set(params)
    if missing:
        raise ValueError(
            "missing required local optimization options: " + _names(missing)
        )
    try:
        LocalControlGain(params["gain"], params["gain_units"])
    except (TypeError, ValueError) as exc:
        raise ValueError(str(exc)) from exc
    parse_local_initialization(params["initialization"])

    for key in _LOCAL_BOOL_KEYS & params.keys():
        if not isinstance(params[key], bool):
            raise ValueError(f"{key} must be a bool")
    for key in _LOCAL_REAL_KEYS & params.keys():
        value = params[key]
        if key == "segment_size_fs" and value is None:
            continue
        _finite_real(value, label=key)

    if "segment_size_steps" in params and params["segment_size_steps"] is not None:
        _integer(params["segment_size_steps"], label="segment_size_steps", minimum=1)
    if "propagator_func" in params:
        value = params["propagator_func"]
        if value is not None and not callable(value):
            raise ValueError("propagator_func must be callable or None")

    if "eval_mode" in params:
        eval_mode = params["eval_mode"]
        if not isinstance(eval_mode, str) or eval_mode not in {"target", "weights"}:
            raise ValueError("eval_mode must be one of: target, weights")
    if "weight_mode" in params:
        configured_weight_mode = params["weight_mode"]
        if not isinstance(
            configured_weight_mode, str
        ) or configured_weight_mode not in {
            "by_v",
            "by_v_power",
            "custom",
        }:
            raise ValueError("weight_mode must be one of: by_v, by_v_power, custom")

    weight_mode = params.get("weight_mode", "by_v")
    custom = params.get("custom_weights")
    custom_dict = params.get("custom_weights_dict")
    if weight_mode == "custom":
        if (custom is None) == (custom_dict is None):
            raise ValueError(
                "weight_mode='custom' requires exactly one of custom_weights "
                "or custom_weights_dict"
            )
    elif custom is not None or custom_dict is not None:
        raise ValueError(
            "custom_weights and custom_weights_dict require weight_mode='custom'"
        )

    if custom is not None:
        raw = np.asarray(custom)
        if np.iscomplexobj(raw) or np.issubdtype(raw.dtype, np.bool_):
            raise ValueError(
                "custom_weights must be a finite real one-dimensional array"
            )
        try:
            weights = np.asarray(raw, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "custom_weights must be a finite real one-dimensional array"
            ) from exc
        if weights.ndim != 1 or not np.all(np.isfinite(weights)):
            raise ValueError(
                "custom_weights must be a finite real one-dimensional array"
            )
    if custom_dict is not None:
        if not isinstance(custom_dict, Mapping):
            raise ValueError("custom_weights_dict must be a mapping")
        for raw_weight in custom_dict.values():
            _finite_real(raw_weight, label="custom_weights_dict values")


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
            raise ValueError(f"{key} was removed; {_REMOVED_LOCAL_KEYS[key]}")
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
    if algorithm in {"grape", "krotov"}:
        _validate_common_options(params)
    if algorithm == "grape" and "learning_rate" in params:
        _finite_real(params["learning_rate"], label="learning_rate")
    if algorithm == "local":
        _validate_local_options(params)
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
    for key in ("total_fs", "field_dt_fs"):
        if key in time_cfg:
            value = _finite_real(time_cfg[key], label=key)
            if value <= 0.0:
                raise ValueError(f"{key} must be positive")
    if "sample_stride" in time_cfg:
        _integer(time_cfg["sample_stride"], label="sample_stride", minimum=1)


__all__ = [
    "GRAPE_OPTION_KEYS",
    "KROTOV_OPTION_KEYS",
    "LOCAL_OPTION_KEYS",
    "OptimizationAlgorithm",
    "validate_algorithm_options",
    "validate_local_time_options",
]
