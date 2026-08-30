"""Contracts for explicit, report-only simulation convergence assessment."""

from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest

from rovibrational_excitation.core.time import TimeGrid
from rovibrational_excitation.fields import ScalarField
from rovibrational_excitation.simulation.convergence import (
    ConvergenceConfigurationError,
    assess_simulation_convergence,
)


def _generated_case(*, dt: float) -> dict:
    return {
        "basis_type": "twolevel",
        "energy_gap": 0.2,
        "energy_gap_units": "rad/fs",
        "dipole_scale": 3.0e-30,
        "dipole_scale_units": "C*m",
        "t_start": -0.4,
        "t_end": 0.4,
        "dt": dt,
        "duration": 0.3,
        "t_center": 0.0,
        "envelope_kind": "gaussian_fwhm",
        "modulation_kind": "none",
        "carrier_frequency": 0.1,
        "carrier_frequency_units": "PHz",
        "amplitude": 1.0e8,
        "initial_states": [0],
        "backend": "numpy",
        "storage": "dense",
        "algorithm": "rk4",
        "return_traj": False,
        "sample_stride": 1,
        "nondimensional": False,
        "renorm": False,
        "save": False,
    }


_GENERATED_KEYS = {
    "t_start",
    "t_end",
    "dt",
    "duration",
    "t_center",
    "envelope_kind",
    "modulation_kind",
    "carrier_frequency",
    "carrier_frequency_units",
    "amplitude",
}


def _external_case() -> dict:
    return {
        key: value
        for key, value in _generated_case(dt=0.1).items()
        if key not in _GENERATED_KEYS
    }


@patch("rovibrational_excitation.simulation.convergence._execute_one")
def test_generated_convergence_reports_caller_selected_max_absolute_difference(
    execute_one,
):
    execute_one.side_effect = [
        np.array([[0.875, 0.125]]),
        np.array([[0.75, 0.25]]),
    ]
    coarse = _generated_case(dt=0.1)
    fine = _generated_case(dt=0.05)
    original_coarse = deepcopy(coarse)
    original_fine = deepcopy(fine)

    report = assess_simulation_convergence(
        coarse,
        fine,
        observable_name="final_population",
        observable=lambda population: population[-1],
        tolerance=0.125,
    )

    assert report.observable_name == "final_population"
    assert report.field_kind == "generated"
    assert report.coarse_field_dt_fs == pytest.approx(0.1)
    assert report.fine_field_dt_fs == pytest.approx(0.05)
    assert report.refinement_ratio == pytest.approx(2.0)
    assert report.tolerance == 0.125
    assert report.max_absolute_difference == 0.125
    assert report.converged is True
    np.testing.assert_array_equal(report.coarse_observable, [0.875, 0.125])
    np.testing.assert_array_equal(report.fine_observable, [0.75, 0.25])
    assert not report.coarse_observable.flags.writeable
    assert not report.fine_observable.flags.writeable
    assert coarse == original_coarse
    assert fine == original_fine
    assert all(call.kwargs["field"] is None for call in execute_one.call_args_list)
    assert all(call.args[0]["save"] is False for call in execute_one.call_args_list)


@patch("rovibrational_excitation.simulation.convergence._execute_one")
def test_external_fields_use_their_explicit_grids(execute_one):
    execute_one.side_effect = [np.array([[1.0, 0.0]]), np.array([[1.0, 0.0]])]
    coarse_grid = TimeGrid.from_bounds(0.0, 0.4, 0.1)
    fine_grid = TimeGrid.from_bounds(0.0, 0.4, 0.05)
    coarse_field = ScalarField(coarse_grid, np.zeros(5))
    fine_field = ScalarField(fine_grid, np.zeros(9))

    report = assess_simulation_convergence(
        _external_case(),
        _external_case(),
        coarse_field=coarse_field,
        fine_field=fine_field,
        observable_name="final_population",
        observable=lambda population: population[-1],
        tolerance=0.0,
    )

    assert report.field_kind == "scalar"
    assert report.max_absolute_difference == 0.0
    assert report.converged is True
    assert execute_one.call_args_list[0].kwargs["field"] is coarse_field
    assert execute_one.call_args_list[1].kwargs["field"] is fine_field


@pytest.mark.parametrize("tolerance", [-1.0, np.inf, np.nan, True])
def test_tolerance_must_be_finite_nonnegative_number(tolerance):
    with pytest.raises(ConvergenceConfigurationError, match="tolerance"):
        assess_simulation_convergence(
            _generated_case(dt=0.1),
            _generated_case(dt=0.05),
            observable_name="final_population",
            observable=lambda population: population[-1],
            tolerance=tolerance,
        )


@pytest.mark.parametrize("name", ["", "   ", 1])
def test_observable_name_must_be_a_nonempty_string(name):
    with pytest.raises(ConvergenceConfigurationError, match="observable_name"):
        assess_simulation_convergence(
            _generated_case(dt=0.1),
            _generated_case(dt=0.05),
            observable_name=name,
            observable=lambda population: population[-1],
            tolerance=1.0,
        )


def test_observable_must_be_callable():
    with pytest.raises(ConvergenceConfigurationError, match="observable"):
        assess_simulation_convergence(
            _generated_case(dt=0.1),
            _generated_case(dt=0.05),
            observable_name="final_population",
            observable=np.array([1.0]),
            tolerance=1.0,
        )


def test_fine_grid_must_have_a_smaller_explicit_step():
    with pytest.raises(ConvergenceConfigurationError, match="fine.*smaller"):
        assess_simulation_convergence(
            _generated_case(dt=0.05),
            _generated_case(dt=0.1),
            observable_name="final_population",
            observable=lambda population: population[-1],
            tolerance=1.0,
        )


def test_grids_must_have_identical_endpoints():
    fine = _generated_case(dt=0.05)
    fine["t_end"] = 0.5

    with pytest.raises(ConvergenceConfigurationError, match="endpoints"):
        assess_simulation_convergence(
            _generated_case(dt=0.1),
            fine,
            observable_name="final_population",
            observable=lambda population: population[-1],
            tolerance=1.0,
        )


def test_non_grid_calculation_parameters_must_match():
    fine = _generated_case(dt=0.05)
    fine["amplitude"] = 2.0e8

    with pytest.raises(ConvergenceConfigurationError, match="amplitude"):
        assess_simulation_convergence(
            _generated_case(dt=0.1),
            fine,
            observable_name="final_population",
            observable=lambda population: population[-1],
            tolerance=1.0,
        )


def test_generated_and_external_routes_cannot_be_mixed():
    fine_grid = TimeGrid.from_bounds(0.0, 0.4, 0.05)
    fine_field = ScalarField(fine_grid, np.zeros(9))

    with pytest.raises(ConvergenceConfigurationError, match="both.*fields"):
        assess_simulation_convergence(
            _generated_case(dt=0.1),
            _external_case(),
            fine_field=fine_field,
            observable_name="final_population",
            observable=lambda population: population[-1],
            tolerance=1.0,
        )


def test_convergence_service_rejects_result_writes():
    coarse = _generated_case(dt=0.1)
    coarse["save"] = True

    with pytest.raises(ConvergenceConfigurationError, match="save=False"):
        assess_simulation_convergence(
            coarse,
            _generated_case(dt=0.05),
            observable_name="final_population",
            observable=lambda population: population[-1],
            tolerance=1.0,
        )


@patch("rovibrational_excitation.simulation.convergence._execute_one")
@pytest.mark.parametrize(
    ("coarse_observable", "fine_observable", "message"),
    [
        (np.array([]), np.array([]), "nonempty"),
        (np.array([1.0]), np.array([[1.0]]), "same shape"),
        (np.array([np.nan]), np.array([1.0]), "finite"),
        (np.array([True]), np.array([False]), "numeric"),
    ],
)
def test_observable_outputs_are_strictly_validated(
    execute_one,
    coarse_observable,
    fine_observable,
    message,
):
    execute_one.side_effect = [np.array([[1.0, 0.0]]), np.array([[1.0, 0.0]])]
    outputs = iter((coarse_observable, fine_observable))

    with pytest.raises(ConvergenceConfigurationError, match=message):
        assess_simulation_convergence(
            _generated_case(dt=0.1),
            _generated_case(dt=0.05),
            observable_name="test_observable",
            observable=lambda _population: next(outputs),
            tolerance=1.0,
        )


def test_real_generated_convergence_executes_without_changing_either_case():
    coarse = _generated_case(dt=0.1)
    fine = _generated_case(dt=0.05)
    coarse["amplitude"] = 0.0
    fine["amplitude"] = 0.0
    original_coarse = deepcopy(coarse)
    original_fine = deepcopy(fine)

    report = assess_simulation_convergence(
        coarse,
        fine,
        observable_name="final_population",
        observable=lambda population: population[-1],
        tolerance=0.0,
    )

    assert report.converged is True
    assert report.max_absolute_difference == 0.0
    np.testing.assert_array_equal(report.coarse_observable, [1.0, 0.0])
    np.testing.assert_array_equal(report.fine_observable, [1.0, 0.0])
    assert coarse == original_coarse
    assert fine == original_fine
