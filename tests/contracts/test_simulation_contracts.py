"""Regression tests for simulation-level contracts."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from rovibrational_excitation.simulation.checkpoint import CheckpointManager
from rovibrational_excitation.simulation.runner import _run_one, run_all_with_checkpoint
from rovibrational_excitation.simulation.timegrid import build_time_grid
from rovibrational_excitation.simulation.validation import (
    SimulationConfigurationError,
    validate_simulation_case,
)


def _base_case(**overrides):
    case = {
        "basis_type": "twolevel",
        "energy_gap": 0.2,
        "energy_gap_units": "rad/fs",
        "mu0_Cm": 3.0e-30,
        "t_start": -0.5,
        "t_end": 0.5,
        "dt": 0.05,
        "duration": 0.3,
        "t_center": 0.0,
        "carrier_freq": 0.1,
        "amplitude": 1.0e8,
        "polarization": [1.0, 0.0],
        "initial_states": [0],
        "save": False,
    }
    case.update(overrides)
    return case


def test_time_grid_includes_exact_endpoints_and_rk_midpoints():
    grid = build_time_grid(-1.0, 1.0, 0.1)

    assert grid.size == 21
    assert grid[0] == -1.0
    assert grid[-1] == 1.0
    np.testing.assert_allclose(np.diff(grid), 0.1)


def test_time_grid_rejects_span_that_solver_would_truncate():
    with pytest.raises(ValueError, match=r"integer multiple of 2 \* dt"):
        build_time_grid(-1.0, 1.0, 0.3)


@pytest.mark.parametrize(
    "model_overrides",
    [
        {},
        {
            "basis_type": "vibladder",
            "V_max": 2,
            "omega_rad_phz": 0.2,
            "delta_omega_rad_phz": 0.0,
            "potential_type": "harmonic",
        },
    ],
)
@pytest.mark.parametrize("nondimensional", [False, True])
def test_scalar_models_are_independent_of_input_polarization(
    model_overrides, nondimensional
):
    x_polarized = _base_case(nondimensional=nondimensional, **model_overrides)
    y_polarized = {**x_polarized, "polarization": [0.0, 1.0]}

    population_x = _run_one(x_polarized)
    population_y = _run_one(y_polarized)

    np.testing.assert_allclose(population_x, population_y, rtol=1e-12, atol=1e-12)


def test_final_state_only_uses_final_physical_time_and_state_axis(tmp_path):
    params = _base_case(
        amplitude=0.0,
        return_traj=False,
        save=True,
        outdir=str(tmp_path),
    )

    population = _run_one(params)

    assert population.shape == (1, 2)
    with np.load(tmp_path / "result.npz") as data:
        np.testing.assert_array_equal(data["t_p"], np.array([params["t_end"]]))
        assert data["t_E"][0] == params["t_start"]
        assert data["t_E"][-1] == params["t_end"]


def test_validation_rejects_missing_physical_parameter_before_building():
    params = _base_case()
    del params["mu0_Cm"]

    with pytest.raises(SimulationConfigurationError, match="mu0_Cm"):
        validate_simulation_case(params)


def test_validation_rejects_missing_duration():
    params = _base_case()
    del params["duration"]

    with pytest.raises(SimulationConfigurationError, match="duration"):
        validate_simulation_case(params)


def test_validation_rejects_unknown_potential_before_model_construction():
    params = _base_case(
        basis_type="vibladder",
        V_max=2,
        omega_rad_phz=0.2,
        delta_omega_rad_phz=0.0,
        potential_type="quadratic",
    )

    with pytest.raises(SimulationConfigurationError, match="potential_type"):
        validate_simulation_case(params)


def test_validation_rejects_removed_pulse_duration_alias():
    params = _base_case(pulse_duration=0.3)
    params.pop("duration")

    with pytest.raises(
        SimulationConfigurationError,
        match="pulse_duration was removed",
    ):
        validate_simulation_case(params)


def test_checkpoint_deduplicates_cases_and_ignores_runtime_error(tmp_path):
    manager = CheckpointManager(tmp_path)
    case = {"amplitude": 1.0, "save": True, "outdir": "first"}
    duplicate = {**case, "outdir": "second"}
    failed = {**case, "error": "old failure"}

    manager.save_checkpoint([case, duplicate], [failed], 1, 0.0)
    checkpoint = manager.load_checkpoint()

    assert checkpoint is not None
    assert checkpoint["completed_cases"] == 1
    assert checkpoint["failed_cases"] == 0
    assert manager.failed_cases_file.exists()


def test_summary_keeps_each_result_with_its_original_case(tmp_path):
    params = {
        "description": "summary_mapping",
        "amplitude": [1.0, 2.0],
    }
    successful = np.array([[0.25, 0.75]])

    with (
        patch(
            "rovibrational_excitation.simulation.runner._make_root",
            return_value=Path(tmp_path),
        ),
        patch(
            "rovibrational_excitation.simulation.runner._run_one",
            side_effect=[ValueError("invalid first case"), successful],
        ),
    ):
        results = run_all_with_checkpoint(
            params,
            save=True,
            checkpoint_interval=2,
        )

    assert len(results) == 1
    summary = pd.read_csv(tmp_path / "summary.csv")
    first = summary.loc[summary["amplitude"] == 1.0].iloc[0]
    second = summary.loc[summary["amplitude"] == 2.0].iloc[0]
    assert first["status"] == "failed"
    assert second["status"] == "success"
    assert second["pop_0"] == pytest.approx(0.25)
