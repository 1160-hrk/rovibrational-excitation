"""Strict document contracts for configured optimization workflows."""

from __future__ import annotations

import inspect
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

import rovibrational_excitation.visualization.plot_all as plot_all_module
from rovibrational_excitation.optimization.config import (
    OptimizationConfigurationError,
    validate_optimization_config,
)
from rovibrational_excitation.optimization.model import build_optimization_model
from rovibrational_excitation.simulation import optimize_runner

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"
ACTIVE_CONFIG_NAMES = {
    "example_krotov_spectral_viblad_v3.yaml",
    "example_local_viblad_v3.yaml",
    "reference_krotov_viblad_v3.yaml",
}


def _load(name: str = "reference_krotov_viblad_v3.yaml") -> dict[str, Any]:
    with (CONFIG_DIR / name).open(encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    assert isinstance(loaded, dict)
    return loaded


def _fake_result(*, complete_plot_data: bool = False, **_: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"metrics": {"fidelity": 0.0}}
    if complete_plot_data:
        result.update(
            {
                "efield": object(),
                "time": np.array([0.0]),
                "psi_traj": object(),
                "tlist": np.array([0.0]),
                "field_data": object(),
                "target_idx": 0,
            }
        )
    return result


def test_exactly_three_current_schema_optimization_configs_are_active() -> None:
    assert {path.name for path in CONFIG_DIR.glob("*.yaml")} == ACTIVE_CONFIG_NAMES


@pytest.mark.parametrize("name", sorted(ACTIVE_CONFIG_NAMES))
def test_active_optimization_configs_validate_through_model_and_states(
    name: str,
) -> None:
    config = _load(name)

    validated = validate_optimization_config(config, algorithm_override=None)
    model = build_optimization_model(config["system"])
    states = optimize_runner._build_states(model.basis, config["states"])

    assert validated.algorithm == config["algorithm"]["selected"]
    assert states == {
        "initial": tuple(config["states"]["initial"]),
        "target": tuple(config["states"]["target"]),
    }


@pytest.mark.parametrize("algorithm", ["grape", "krotov", "local"])
def test_control_axes_are_required_for_every_algorithm(algorithm: str) -> None:
    config = _load()
    config["algorithm"]["selected"] = algorithm
    config["algorithms"] = {algorithm: {}}
    if algorithm == "krotov":
        config["algorithms"][algorithm] = deepcopy(_load()["algorithms"]["krotov"])
        del config["algorithms"][algorithm]["control_axes"]
    if algorithm == "local":
        config["time"] = {
            "total_fs": 1.0,
            "field_dt_fs": 0.1,
            "sample_stride": 1,
        }

    with pytest.raises(
        OptimizationConfigurationError,
        match=rf"missing required {algorithm} optimization option: control_axes",
    ):
        validate_optimization_config(config, algorithm_override=None)


@pytest.mark.parametrize("missing", ["gain", "gain_units"])
def test_local_gain_value_and_unit_are_required_by_document_schema(
    missing: str,
) -> None:
    config = _load("example_local_viblad_v3.yaml")
    del config["algorithms"]["local"][missing]

    with pytest.raises(
        OptimizationConfigurationError,
        match=rf"missing required.*{missing}",
    ):
        validate_optimization_config(config, algorithm_override=None)


def test_local_example_preserves_the_historical_canonical_gain() -> None:
    config = _load("example_local_viblad_v3.yaml")
    params = config["algorithms"]["local"]

    assert params["gain"] == 1000.0
    assert params["gain_units"] == "(GV/m)^2 fs"


def test_local_example_requires_explicit_seed_field_initialization() -> None:
    config = _load("example_local_viblad_v3.yaml")
    initialization = config["algorithms"]["local"]["initialization"]

    assert initialization == {
        "method": "seed_field",
        "amplitude": 1000.0,
        "amplitude_units": "V/m",
        "max_segments": 5,
    }

    del config["algorithms"]["local"]["initialization"]
    with pytest.raises(OptimizationConfigurationError, match="initialization"):
        validate_optimization_config(config, algorithm_override=None)


@pytest.mark.parametrize("axes", ["x", "xyz", "Xy", "x1"])
def test_control_axes_require_exactly_two_lowercase_cartesian_axes(axes: str) -> None:
    config = _load()
    config["algorithms"]["krotov"]["control_axes"] = axes

    with pytest.raises(OptimizationConfigurationError, match="exactly two lowercase"):
        validate_optimization_config(config, algorithm_override=None)


def test_grape_rejects_axes_that_its_numerical_path_does_not_implement() -> None:
    config = _load()
    config["algorithm"]["selected"] = "grape"
    config["algorithms"] = {"grape": {"control_axes": "zx"}}

    with pytest.raises(OptimizationConfigurationError, match="only control_axes='xy'"):
        validate_optimization_config(config, algorithm_override=None)


def test_grape_document_requires_an_explicit_initial_field() -> None:
    config = _load()
    params = deepcopy(config["algorithms"]["krotov"])
    params["control_axes"] = "xy"
    del params["initial_field_kind"]
    config["algorithm"]["selected"] = "grape"
    config["algorithms"] = {"grape": params}

    with pytest.raises(
        OptimizationConfigurationError,
        match="missing required GRAPE parameter: initial_field_kind",
    ):
        validate_optimization_config(config, algorithm_override=None)


def test_grape_document_requires_exact_sampled_seed_length() -> None:
    config = _load()
    config["algorithm"]["selected"] = "grape"
    config["time"] = {
        "total_fs": 0.4,
        "field_dt_fs": 0.1,
        "output_stride": 1,
    }
    config["algorithms"] = {
        "grape": {
            "control_axes": "xy",
            "initial_field_kind": "sampled",
            "initial_field_samples": np.zeros((4, 2)),
            "initial_field_units": "V/m",
        }
    }

    with pytest.raises(
        OptimizationConfigurationError,
        match=r"length must exactly match time_grid \(4 != 5\)",
    ):
        validate_optimization_config(config, algorithm_override=None)


def test_unknown_algorithm_and_spectrum_options_are_rejected() -> None:
    config = _load()
    config["algorithms"]["krotov"]["max_iterations"] = 10
    with pytest.raises(OptimizationConfigurationError, match="max_iterations"):
        validate_optimization_config(config, algorithm_override=None)

    spectral = _load("example_krotov_spectral_viblad_v3.yaml")
    spectral["algorithms"]["krotov"]["spectrum_constraints"]["alpha_sacle"] = 1.0
    with pytest.raises(OptimizationConfigurationError, match="alpha_sacle"):
        validate_optimization_config(spectral, algorithm_override=None)


def test_krotov_document_requires_initial_field_units_and_exact_sample_count() -> None:
    config = _load()
    del config["algorithms"]["krotov"]["initial_amplitude_units"]
    with pytest.raises(OptimizationConfigurationError, match="initial_amplitude_units"):
        validate_optimization_config(config, algorithm_override=None)

    config = _load()
    config["time"] = {
        "total_fs": 0.4,
        "field_dt_fs": 0.1,
        "output_stride": 1,
    }
    config["algorithms"]["krotov"] = {
        "control_axes": "xy",
        "initial_field_kind": "sampled",
        "initial_field_samples": np.zeros((4, 2)),
        "initial_field_units": "V/m",
    }
    with pytest.raises(OptimizationConfigurationError, match="4 != 5"):
        validate_optimization_config(config, algorithm_override=None)


def test_root_plot_and_output_schemas_do_not_accept_fallback_keys() -> None:
    config = _load()
    config["unexpected"] = True
    with pytest.raises(OptimizationConfigurationError, match="unexpected"):
        validate_optimization_config(config, algorithm_override=None)

    config = _load()
    config["plot"]["save_fig"] = False
    del config["plot"]["enabled"]
    with pytest.raises(
        OptimizationConfigurationError, match="plot.save_fig was removed"
    ):
        validate_optimization_config(config, algorithm_override=None)

    config = _load()
    config["output"]["dir"] = ""
    with pytest.raises(OptimizationConfigurationError, match="nonempty string"):
        validate_optimization_config(config, algorithm_override=None)


def test_runner_has_no_unrestricted_keyword_argument_boundary() -> None:
    parameters = inspect.signature(optimize_runner.run_from_config).parameters.values()
    assert all(
        parameter.kind is not inspect.Parameter.VAR_KEYWORD for parameter in parameters
    )


def test_output_directory_config_is_used_and_api_argument_takes_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _load()
    config["output"]["dir"] = str(tmp_path / "configured")
    monkeypatch.setitem(optimize_runner.ALGO_REGISTRY, "krotov", _fake_result)

    configured = optimize_runner.run_from_config(config, do_plot=False)
    explicit = optimize_runner.run_from_config(
        config,
        out_dir=tmp_path / "explicit",
        do_plot=False,
    )

    assert Path(configured["out_dir"]).parent == tmp_path / "configured"
    assert Path(explicit["out_dir"]).parent == tmp_path / "explicit"


def test_plot_config_is_used_and_explicit_false_takes_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _load()
    config["plot"]["enabled"] = True
    monkeypatch.setitem(optimize_runner.ALGO_REGISTRY, "krotov", _fake_result)

    with pytest.raises(ValueError, match="missing data required for plotting"):
        optimize_runner.run_from_config(config, out_dir=tmp_path, do_plot=None)

    optimize_runner.run_from_config(config, out_dir=tmp_path, do_plot=False)


def test_requested_plot_failure_is_not_reported_as_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _load()
    monkeypatch.setitem(
        optimize_runner.ALGO_REGISTRY,
        "krotov",
        lambda **kwargs: _fake_result(complete_plot_data=True, **kwargs),
    )

    def fail_plot(**_: Any) -> None:
        raise RuntimeError("plot failed")

    monkeypatch.setattr(plot_all_module, "plot_all", fail_plot)
    with pytest.raises(RuntimeError, match="plot failed"):
        optimize_runner.run_from_config(config, out_dir=tmp_path, do_plot=True)


def test_runner_does_not_mutate_caller_config_when_applying_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _load()
    original = deepcopy(config)
    monkeypatch.setitem(optimize_runner.ALGO_REGISTRY, "krotov", _fake_result)

    optimize_runner.run_from_config(
        config,
        overrides=["algorithms.krotov.max_iter=0"],
        out_dir=tmp_path,
        do_plot=False,
    )

    assert config == original
