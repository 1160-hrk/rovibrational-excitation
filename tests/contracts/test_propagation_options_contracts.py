"""Contracts for required typed propagation options."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from rovibrational_excitation.core.execution import (
    ArrayBackend,
    ExecutionPolicy,
    MatrixStorage,
)
from rovibrational_excitation.dynamics.capabilities import (
    PropagationAlgorithm,
)
from rovibrational_excitation.dynamics.options import (
    PropagationOptions,
    RenormalizationPolicy,
    ScalingMode,
)


def _options(**overrides):
    values = {
        "algorithm": PropagationAlgorithm.RK4,
        "execution": ExecutionPolicy(
            backend=ArrayBackend.NUMPY,
            storage=MatrixStorage.DENSE,
        ),
        "return_trajectory": True,
        "sample_stride": 1,
        "scaling": ScalingMode.DIMENSIONAL,
        "renormalization": RenormalizationPolicy.DISABLED,
    }
    values.update(overrides)
    return PropagationOptions(**values)


def test_core_states_import_is_independent_of_propagation_facade_order():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from rovibrational_excitation.core.states import PureState; "
                "from rovibrational_excitation.dynamics import "
                "PropagationOptions"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).parents[2] / "src"),
        },
    )

    assert completed.returncode == 0, completed.stderr


def test_propagation_options_have_no_defaults():
    with pytest.raises(TypeError):
        PropagationOptions()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("algorithm", "rk4", "PropagationAlgorithm"),
        ("execution", "numpy-dense", "ExecutionPolicy"),
        ("return_trajectory", 1, "return_trajectory"),
        ("scaling", "dimensional", "ScalingMode"),
        ("renormalization", False, "RenormalizationPolicy"),
    ],
)
def test_propagation_options_require_typed_explicit_values(field, value, message):
    with pytest.raises(TypeError, match=message):
        _options(**{field: value})


@pytest.mark.parametrize("sample_stride", [True, 0, -1, 1.5])
def test_propagation_options_require_positive_integer_stride(sample_stride):
    with pytest.raises((TypeError, ValueError), match="sample_stride"):
        _options(sample_stride=sample_stride)


def test_propagation_options_project_only_to_legacy_solver_values():
    options = _options(
        algorithm=PropagationAlgorithm.SPLIT_OPERATOR,
        execution=ExecutionPolicy(
            backend=ArrayBackend.NUMPY,
            storage=MatrixStorage.CSR,
        ),
        return_trajectory=False,
        sample_stride=3,
        scaling=ScalingMode.NONDIMENSIONAL,
        renormalization=RenormalizationPolicy.PER_STEP,
    )

    assert options.algorithm_name == "split_operator"
    assert options.backend_name == "numpy"
    assert options.sparse is True
    assert options.nondimensional is True
    assert options.renorm is True
