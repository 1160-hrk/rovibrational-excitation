"""Exact public-root contract accepted for v0.3 under D-073."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import NoneType
from typing import get_args, get_type_hints

import rovibrational_excitation as rve
from rovibrational_excitation.core.execution import ExecutionPolicy
from rovibrational_excitation.core.time import TimeGrid
from rovibrational_excitation.dynamics.options import PropagationOptions
from rovibrational_excitation.dynamics.problem import PropagationProblem
from rovibrational_excitation.dynamics.result import PropagationResult
from rovibrational_excitation.fields.field import ElectricField
from rovibrational_excitation.fields.sampled import CartesianField, ScalarField
from rovibrational_excitation.simulation.runner import run_simulation_case

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src" / "rovibrational_excitation"

EXPECTED_ROOT_EXPORTS = [
    "__version__",
    "ElectricField",
    "TimeGrid",
    "ExecutionPolicy",
    "PropagationProblem",
    "PropagationOptions",
    "PropagationResult",
    "run_simulation_case",
]


def test_root_exports_exact_typed_v03_surface():
    assert rve.__all__ == EXPECTED_ROOT_EXPORTS
    assert rve.ElectricField is ElectricField
    assert rve.TimeGrid is TimeGrid
    assert rve.ExecutionPolicy is ExecutionPolicy
    assert rve.PropagationProblem is PropagationProblem
    assert rve.PropagationOptions is PropagationOptions
    assert rve.PropagationResult is PropagationResult
    assert rve.run_simulation_case is run_simulation_case


def test_root_runner_types_both_explicit_field_routes():
    field_types = set(get_args(get_type_hints(run_simulation_case)["field"]))

    assert field_types == {ScalarField, CartesianField, NoneType}


def test_removed_root_convenience_names_have_no_compatibility_shims():
    removed = {
        "AbsorbanceCalculator",
        "DensityMatrix",
        "ExperimentalConditions",
        "Hamiltonian",
        "LinMolBasis",
        "LinMolDipoleMatrix",
        "StateVector",
        "create_calculator_from_params",
    }

    assert removed.isdisjoint(vars(rve))
    for name in removed:
        assert not hasattr(rve, name)


def test_fresh_root_import_does_not_load_heavy_or_workflow_subpackages():
    environment = {**os.environ, "PYTHONPATH": str(PACKAGE.parent)}
    forbidden = [
        "rovibrational_excitation.io",
        "rovibrational_excitation.optimization",
        "rovibrational_excitation.simulation",
        "rovibrational_excitation.spectroscopy",
        "rovibrational_excitation.visualization",
        "matplotlib",
        "pandas",
    ]
    script = (
        "import sys; import rovibrational_excitation as rve; "
        f"assert rve.__all__ == {EXPECTED_ROOT_EXPORTS!r}; "
        f"forbidden = {forbidden!r}; "
        "loaded = [name for name in forbidden if name in sys.modules]; "
        "assert loaded == [], loaded"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
