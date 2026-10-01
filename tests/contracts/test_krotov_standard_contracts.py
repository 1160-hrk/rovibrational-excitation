"""Strict public contracts for standard interval-control Krotov."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from rovibrational_excitation.optimization import ALGO_REGISTRY
from rovibrational_excitation.optimization.config import (
    OptimizationConfigurationError,
    validate_optimization_config,
)
from rovibrational_excitation.optimization.krotov_controls import (
    KrotovSampledControl,
    parse_krotov_initial_control,
)
from rovibrational_excitation.optimization.krotov_timegrid import KrotovIntervalGrid
from rovibrational_excitation.optimization.options import validate_algorithm_options
from rovibrational_excitation.simulation.optimize_runner import run_from_config

ROOT = Path(__file__).resolve().parents[2]


def _document() -> dict:
    with (ROOT / "configs" / "reference_legacy_batch_overlap_viblad_v3.yaml").open(
        encoding="utf-8"
    ) as stream:
        document = yaml.safe_load(stream)
    document["algorithm"]["selected"] = "krotov"
    document["time"] = {
        "total_fs": 0.4,
        "control_dt_fs": 0.1,
        "output_stride": 1,
    }
    document["algorithms"] = {
        "krotov": {
            "control_axes": "zx",
            "lambda_a": 0.01,
            "lambda_a_units": "1 / ((GV/m)^2 fs)",
            "max_iter": 1,
            "target_fidelity": 1.0,
            "initial_control_kind": "sampled",
            "initial_control_samples": np.zeros((4, 2)),
            "initial_control_units": "V/m",
        }
    }
    return document


def test_standard_and_legacy_routes_are_both_explicitly_registered():
    assert "krotov" in ALGO_REGISTRY
    assert "legacy_batch_overlap" in ALGO_REGISTRY
    assert ALGO_REGISTRY["krotov"] is not ALGO_REGISTRY["legacy_batch_overlap"]


def test_interval_grid_stores_state_endpoints_and_midpoint_controls():
    grid = KrotovIntervalGrid.from_config(
        {"total_fs": 0.4, "control_dt_fs": 0.1, "output_stride": 3}
    )

    np.testing.assert_array_equal(grid.state_times_fs, np.linspace(0.0, 0.4, 5))
    np.testing.assert_allclose(
        grid.control_times_fs, np.array([0.05, 0.15, 0.25, 0.35])
    )
    assert grid.interval_count == 4
    assert grid.output_stride == 3


@pytest.mark.parametrize(
    ("time_cfg", "match"),
    [
        (
            {"total_fs": 0.4, "field_dt_fs": 0.05},
            "control_dt_fs, not field_dt_fs",
        ),
        (
            {"total_fs": 0.45, "control_dt_fs": 0.1},
            "exactly divisible",
        ),
        (
            {"total_fs": 0.4, "control_dt_fs": 0.1, "output_stride": True},
            "positive integer",
        ),
    ],
)
def test_interval_grid_rejects_implicit_time_changes(time_cfg, match):
    with pytest.raises(ValueError, match=match):
        KrotovIntervalGrid.from_config(time_cfg)


def test_standard_krotov_penalty_value_and_unit_are_both_required():
    base = {"control_axes": "xy"}
    for missing in ("lambda_a", "lambda_a_units"):
        params = {
            **base,
            "lambda_a": 0.01,
            "lambda_a_units": "1 / ((GV/m)^2 fs)",
        }
        del params[missing]
        with pytest.raises(ValueError, match=missing):
            validate_algorithm_options("krotov", params)


def test_standard_krotov_rejects_spectral_constraints_until_independent_reference():
    with pytest.raises(ValueError, match="unsupported krotov.*spectrum_constraints"):
        validate_algorithm_options(
            "krotov",
            {
                "control_axes": "xy",
                "lambda_a": 0.01,
                "lambda_a_units": "1 / ((GV/m)^2 fs)",
                "spectrum_constraints": {},
            },
        )


def test_sampled_interval_control_requires_exact_interval_count():
    parsed = parse_krotov_initial_control(
        {
            "initial_control_kind": "sampled",
            "initial_control_samples": np.zeros((3, 2)),
            "initial_control_units": "MV/m",
        }
    )
    assert isinstance(parsed, KrotovSampledControl)
    with pytest.raises(ValueError, match=r"3 != 4"):
        parsed.samples_on(
            KrotovIntervalGrid.from_config({"total_fs": 0.4, "control_dt_fs": 0.1})
        )


def test_standard_document_validates_without_reinterpreting_legacy_fields():
    document = _document()
    validated = validate_optimization_config(document, algorithm_override=None)
    assert validated.algorithm == "krotov"

    legacy_field_document = deepcopy(document)
    legacy_field_document["time"] = {
        "total_fs": 0.4,
        "field_dt_fs": 0.05,
        "output_stride": 1,
    }
    with pytest.raises(OptimizationConfigurationError, match="control_dt_fs"):
        validate_optimization_config(legacy_field_document, algorithm_override=None)


def test_standard_document_rejects_plotting_without_interval_plot_adapter():
    document = _document()
    document["plot"]["enabled"] = True
    with pytest.raises(OptimizationConfigurationError, match="plotting is unavailable"):
        validate_optimization_config(document, algorithm_override=None)


def test_standard_runner_rejects_explicit_plot_override_before_calculation(tmp_path):
    with pytest.raises(ValueError, match="plotting is unavailable"):
        run_from_config(_document(), out_dir=tmp_path, do_plot=True)
