"""Strict document schema for configured optimization runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from .krotov_initial_field import (
    KrotovSampledInitialField,
    parse_krotov_initial_field,
)
from .options import (
    OptimizationAlgorithm,
    validate_algorithm_options,
    validate_local_time_options,
)
from .timegrid import build_optimization_time_settings


class OptimizationConfigurationError(ValueError):
    """Raised when an optimization document is structurally invalid."""


@dataclass(frozen=True, slots=True)
class OptimizationRunConfiguration:
    """Validated workflow options without changing submitted physical values."""

    algorithm: OptimizationAlgorithm
    algorithm_params: dict[str, Any]
    time: dict[str, Any]
    plot_enabled: bool
    plot_spectrum: bool
    plot_spectrogram: bool
    output_dir: str


_ALGORITHMS = {"local", "krotov", "grape"}
_ROOT_KEYS = {"system", "states", "time", "algorithm", "algorithms", "plot", "output"}


def _names(values: set[Any]) -> str:
    return ", ".join(
        sorted(value if isinstance(value, str) else repr(value) for value in values)
    )


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OptimizationConfigurationError(f"{label} must be a mapping")
    return cast(Mapping[str, Any], value)


def _exact_keys(
    value: Mapping[str, Any],
    *,
    label: str,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    allowed = required | (optional or set())
    unknown = set(value) - allowed
    if unknown:
        if label == "plot" and "save_fig" in unknown:
            raise OptimizationConfigurationError(
                "plot.save_fig was removed; use required plot.enabled"
            )
        raise OptimizationConfigurationError(
            f"Unknown {label} keys: " + _names(unknown)
        )
    missing = required - set(value)
    if missing:
        raise OptimizationConfigurationError(
            f"Missing required {label} keys: " + _names(missing)
        )


def _algorithm_name(value: Any, *, label: str) -> OptimizationAlgorithm:
    if not isinstance(value, str) or value not in _ALGORITHMS:
        raise OptimizationConfigurationError(
            f"{label} must be one of: grape, krotov, local"
        )
    return cast(OptimizationAlgorithm, value)


def _validate_time(algorithm: OptimizationAlgorithm, value: Mapping[str, Any]) -> None:
    try:
        if algorithm == "local":
            if "dt_fs" in value:
                raise ValueError("dt_fs was removed; provide field_dt_fs")
            missing = {"total_fs", "field_dt_fs"} - set(value)
            if missing:
                raise ValueError(
                    "missing required local optimization time options: "
                    + _names(missing)
                )
            validate_local_time_options(value)
            sample_stride = value.get("sample_stride", 1)
            if (
                isinstance(sample_stride, (bool, np.bool_))
                or not isinstance(sample_stride, (int, np.integer))
                or int(sample_stride) < 1
            ):
                raise ValueError("sample_stride must be a positive integer")
            return
        build_optimization_time_settings(value)
    except (TypeError, ValueError) as exc:
        raise OptimizationConfigurationError(str(exc)) from exc


def validate_optimization_config(
    config: Mapping[str, Any],
    *,
    algorithm_override: str | None,
) -> OptimizationRunConfiguration:
    """Validate one optimization document before model or field allocation."""
    if not isinstance(config, Mapping):
        raise OptimizationConfigurationError("optimization config must be a mapping")
    _exact_keys(config, label="optimization config", required=_ROOT_KEYS)

    algorithm_section = _mapping(config["algorithm"], label="algorithm")
    _exact_keys(algorithm_section, label="algorithm", required={"selected"})
    if algorithm_override is None or algorithm_override == "":
        selected_value = algorithm_section["selected"]
    elif not isinstance(algorithm_override, str):
        raise OptimizationConfigurationError(
            "algorithm override must be one of: grape, krotov, local"
        )
    else:
        selected_value = algorithm_override
    selected = _algorithm_name(
        selected_value,
        label="selected optimization algorithm",
    )

    algorithms = _mapping(config["algorithms"], label="algorithms")
    unknown_algorithms = set(algorithms) - _ALGORITHMS
    if unknown_algorithms:
        raise OptimizationConfigurationError(
            "Unknown algorithms keys: " + _names(unknown_algorithms)
        )
    if selected not in algorithms:
        raise OptimizationConfigurationError(
            f"Missing required algorithms.{selected} mapping"
        )
    for name, raw_params in algorithms.items():
        algorithm = _algorithm_name(name, label="algorithms key")
        params = _mapping(raw_params, label=f"algorithms.{algorithm}")
        try:
            validate_algorithm_options(algorithm, params)
            if algorithm == "krotov":
                parse_krotov_initial_field(params)
        except (TypeError, ValueError) as exc:
            raise OptimizationConfigurationError(str(exc)) from exc

    time = _mapping(config["time"], label="time")
    _validate_time(selected, time)
    if selected == "krotov":
        selected_initial_field = parse_krotov_initial_field(
            _mapping(algorithms[selected], label="algorithms.krotov")
        )
        if isinstance(selected_initial_field, KrotovSampledInitialField):
            expected_length = build_optimization_time_settings(
                time
            ).grid.field_times_fs.size
            actual_length = selected_initial_field.samples_v_per_m.shape[0]
            if actual_length != expected_length:
                raise OptimizationConfigurationError(
                    "initial_field_samples length must exactly match time_grid "
                    f"({actual_length} != {expected_length})"
                )

    plot = _mapping(config["plot"], label="plot")
    _exact_keys(
        plot,
        label="plot",
        required={"enabled", "spectrum", "spectrogram"},
    )
    for key in ("enabled", "spectrum", "spectrogram"):
        if not isinstance(plot[key], bool):
            raise OptimizationConfigurationError(f"plot.{key} must be a bool")

    output = _mapping(config["output"], label="output")
    _exact_keys(output, label="output", required={"dir"})
    output_dir = output["dir"]
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise OptimizationConfigurationError("output.dir must be a nonempty string")

    return OptimizationRunConfiguration(
        algorithm=selected,
        algorithm_params=dict(
            _mapping(algorithms[selected], label=f"algorithms.{selected}")
        ),
        time=dict(time),
        plot_enabled=plot["enabled"],
        plot_spectrum=plot["spectrum"],
        plot_spectrogram=plot["spectrogram"],
        output_dir=output_dir,
    )


__all__ = [
    "OptimizationConfigurationError",
    "OptimizationRunConfiguration",
    "validate_optimization_config",
]
