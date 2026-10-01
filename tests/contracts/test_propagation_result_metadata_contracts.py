"""Reproducible metadata contracts for typed propagation results."""

import numpy as np
import pytest

from rovibrational_excitation.core.states import PureState
from rovibrational_excitation.dynamics import (
    PropagationResult,
    SchrodingerPropagator,
)
from rovibrational_excitation.dynamics.options import ScalingMode
from rovibrational_excitation.dynamics.result import RESULT_SCHEMA_VERSION
from tests.propagation_options import propagation_options
from tests.propagation_problem import propagation_problem


def _problem():
    return propagation_problem(PureState(np.array([1.0, 0.0], dtype=np.complex128)))


def test_result_metadata_records_every_execution_choice_and_hash_scope():
    problem = _problem()
    options = propagation_options(return_trajectory=False, sample_stride=3)
    result = SchrodingerPropagator(validate_units=False).propagate(
        problem,
        options=options,
    )

    assert RESULT_SCHEMA_VERSION == 1
    assert result.metadata["result_schema_version"] == 1
    assert result.metadata["model"]["name"] == problem.model.name
    assert result.metadata["model"]["dimension"] == 2
    assert result.metadata["algorithm"] == "rk4"
    assert result.metadata["execution_backend"] == "numpy"
    assert result.metadata["matrix_storage"] == "dense"
    assert result.metadata["scaling"] == "dimensional"
    assert result.metadata["renormalization"] == "disabled"
    assert result.metadata["sample_stride"] == 3
    assert result.metadata["configuration_hash_scope"] == (
        "declared_model_metadata_and_propagation_contract"
    )
    assert len(result.metadata["configuration_hash"]) == 64


def test_configuration_hash_is_deterministic_and_changes_with_declared_options():
    problem = _problem()
    solver = SchrodingerPropagator(validate_units=False)
    first = solver.propagate(
        problem,
        options=propagation_options(return_trajectory=False, sample_stride=1),
    )
    second = solver.propagate(
        problem,
        options=propagation_options(return_trajectory=False, sample_stride=1),
    )
    changed = solver.propagate(
        problem,
        options=propagation_options(return_trajectory=False, sample_stride=2),
    )

    assert first.metadata["configuration_hash"] == second.metadata["configuration_hash"]
    assert (
        first.metadata["configuration_hash"] != changed.metadata["configuration_hash"]
    )


def test_nondimensional_result_records_the_scales_used_by_the_same_call():
    problem = _problem()
    problem.field.Efield[:, 0] = 1.0
    result = SchrodingerPropagator(validate_units=False).propagate(
        problem,
        options=propagation_options(
            return_trajectory=False,
            scaling=ScalingMode.NONDIMENSIONAL,
        ),
    )

    scales = result.metadata["nondimensionalization_scales"]
    assert scales["energy_J"] > 0.0
    assert scales["time_s"] > 0.0
    assert scales["lambda_coupling"] >= 0.0
    assert scales["energy_source"] in {"derived", "explicit"}


def test_result_rejects_non_json_metadata_instead_of_stringifying_it():
    with pytest.raises(TypeError, match="JSON-compatible"):
        PropagationResult(
            times_fs=np.array([0.2]),
            state=np.array([1.0, 0.0], dtype=np.complex128),
            state_kind="wavefunction",
            trajectory=False,
            backend="numpy",
            metadata={"ambiguous": object()},
        )
