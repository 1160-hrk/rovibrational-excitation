"""Contracts for the immutable normal-simulation case boundary."""

from dataclasses import FrozenInstanceError
from unittest.mock import patch

import numpy as np
import pytest

from rovibrational_excitation.core.time import TimeGrid
from rovibrational_excitation.dynamics.result import PropagationResult
from rovibrational_excitation.dynamics.scaling import converter as scaling_converter
from rovibrational_excitation.dynamics.scaling import reporting as scaling_reporting
from rovibrational_excitation.dynamics.schrodinger import SchrodingerPropagator
from rovibrational_excitation.fields import ScalarField
from rovibrational_excitation.models.factory import build_model_from_parameters
from rovibrational_excitation.models.two_level import TwoLevelParameters
from rovibrational_excitation.simulation import validation as simulation_validation
from rovibrational_excitation.simulation.case import SimulationCase
from rovibrational_excitation.simulation.runner import _run_one
from rovibrational_excitation.simulation.validation import validate_simulation_case


def _twolevel_case():
    return {
        "basis_type": "twolevel",
        "energy_gap": 0.2,
        "energy_gap_units": "rad/fs",
        "dipole_scale": 3.0e-30,
        "dipole_scale_units": "C*m",
        "t_start": -0.5,
        "t_start_units": "fs",
        "t_end": 0.5,
        "t_end_units": "fs",
        "dt": 0.05,
        "dt_units": "fs",
        "duration": 0.3,
        "duration_units": "fs",
        "t_center": 0.0,
        "t_center_units": "fs",
        "envelope_kind": "gaussian_fwhm",
        "modulation_kind": "none",
        "carrier_frequency": 0.1,
        "carrier_frequency_units": "PHz",
        "amplitude": 0.0,
        "amplitude_units": "V/m",
        "initial_states": [0],
        "save": False,
        "backend": "numpy",
        "storage": "dense",
        "algorithm": "rk4",
        "return_traj": True,
        "sample_stride": 1,
        "nondimensional": False,
        "renorm": False,
    }


def test_simulation_case_freezes_model_state_field_and_execution_values():
    params = _twolevel_case()
    options = validate_simulation_case(params)
    grid = TimeGrid.from_bounds(
        params["t_start"],
        params["t_end"],
        params["dt"],
    )
    field = ScalarField(grid, np.zeros(grid.field_times_fs.size))

    case = SimulationCase.from_validated_mapping(
        params,
        field=field,
        options=options,
    )
    params["initial_states"][0] = 1

    assert isinstance(case.model_parameters, TwoLevelParameters)
    assert case.initial_states == (0,)
    assert case.field is field
    assert case.time_grid is grid
    assert case.options is options
    assert case.representation is None
    assert case.axes is None
    with pytest.raises(FrozenInstanceError):
        case.verbose = True  # type: ignore[misc]


def test_runner_builds_from_the_frozen_model_schema():
    params = _twolevel_case()

    with patch(
        "rovibrational_excitation.simulation.execution.build_model_from_parameters",
        wraps=build_model_from_parameters,
    ) as typed_builder:
        population = _run_one(params)

    assert population.shape == (11, 2)
    model_parameters = typed_builder.call_args.args[0]
    assert isinstance(model_parameters, TwoLevelParameters)
    assert typed_builder.call_args.kwargs["initial_states"] == (0,)


def test_one_case_preserves_preparation_scaling_and_host_conversion_order(
    monkeypatch,
):
    events = []

    original_resolve = simulation_validation._resolve_simulation_case

    def tracked_resolve(*args, **kwargs):
        events.append("validate")
        return original_resolve(*args, **kwargs)

    original_freeze = SimulationCase.from_validated_mapping.__func__

    def tracked_freeze(cls, *args, **kwargs):
        events.append("freeze")
        return original_freeze(cls, *args, **kwargs)

    original_propagate = SchrodingerPropagator.propagate

    def tracked_propagate(self, *args, **kwargs):
        events.append("propagate_start")
        result = original_propagate(self, *args, **kwargs)
        events.append("propagate_end")
        return result

    original_nondimensionalize = scaling_converter.nondimensionalize_from_objects

    def tracked_nondimensionalize(*args, **kwargs):
        events.append("nondimensionalize")
        return original_nondimensionalize(*args, **kwargs)

    original_analyze = scaling_reporting.analyze_regime

    def tracked_analyze(*args, **kwargs):
        events.append("analyze_regime")
        return original_analyze(*args, **kwargs)

    original_to_numpy = PropagationResult.to_numpy

    def tracked_to_numpy(self):
        events.append("to_numpy")
        return original_to_numpy(self)

    monkeypatch.setattr(
        simulation_validation, "_resolve_simulation_case", tracked_resolve
    )
    monkeypatch.setattr(
        SimulationCase,
        "from_validated_mapping",
        classmethod(tracked_freeze),
    )
    monkeypatch.setattr(SchrodingerPropagator, "propagate", tracked_propagate)
    monkeypatch.setattr(
        scaling_converter,
        "nondimensionalize_from_objects",
        tracked_nondimensionalize,
    )
    monkeypatch.setattr(scaling_reporting, "analyze_regime", tracked_analyze)
    monkeypatch.setattr(PropagationResult, "to_numpy", tracked_to_numpy)

    population = _run_one(
        {**_twolevel_case(), "amplitude": 1.0e8, "nondimensional": True}
    )

    assert population.shape == (11, 2)
    assert events == [
        "validate",
        "freeze",
        "propagate_start",
        "nondimensionalize",
        "propagate_end",
        "nondimensionalize",
        "analyze_regime",
        "to_numpy",
    ]
