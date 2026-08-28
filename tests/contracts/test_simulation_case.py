"""Contracts for the immutable normal-simulation case boundary."""

from dataclasses import FrozenInstanceError
from unittest.mock import patch

import numpy as np
import pytest

from rovibrational_excitation.core.time import TimeGrid
from rovibrational_excitation.fields import ScalarField
from rovibrational_excitation.models.factory import build_model_from_parameters
from rovibrational_excitation.models.parameters import TwoLevelParameters
from rovibrational_excitation.simulation.case import SimulationCase
from rovibrational_excitation.simulation.runner import _run_one
from rovibrational_excitation.simulation.validation import validate_simulation_case


def _twolevel_case():
    return {
        "basis_type": "twolevel",
        "energy_gap": 0.2,
        "energy_gap_units": "rad/fs",
        "mu0_Cm": 3.0e-30,
        "t_start": -0.5,
        "t_end": 0.5,
        "dt": 0.05,
        "duration": 0.3,
        "t_center": 0.0,
        "envelope_kind": "gaussian_fwhm",
        "modulation_kind": "none",
        "carrier_frequency": 0.1,
        "carrier_frequency_units": "PHz",
        "amplitude": 0.0,
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
        "rovibrational_excitation.simulation.runner.build_model_from_parameters",
        wraps=build_model_from_parameters,
    ) as typed_builder:
        population = _run_one(params)

    assert population.shape == (11, 2)
    model_parameters = typed_builder.call_args.args[0]
    assert isinstance(model_parameters, TwoLevelParameters)
    assert typed_builder.call_args.kwargs["initial_states"] == (0,)
