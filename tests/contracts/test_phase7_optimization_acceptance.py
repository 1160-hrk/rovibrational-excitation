"""Acceptance wiring and dependency guards for Phase 7.3 optimization."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np

import rovibrational_excitation.optimization.config as config_module
import rovibrational_excitation.optimization.grape as grape_module
import rovibrational_excitation.optimization.krotov as krotov_module
import rovibrational_excitation.optimization.legacy_batch_overlap as legacy_module
import rovibrational_excitation.optimization.local as local_module
import rovibrational_excitation.optimization.objective as objective_module
import rovibrational_excitation.optimization.options as options_module
import rovibrational_excitation.optimization.result as result_module
import rovibrational_excitation.optimization.spectral_constraints as spectral_module
from rovibrational_excitation.optimization import ALGO_REGISTRY
from rovibrational_excitation.simulation import optimize_runner

ROOT = Path(__file__).resolve().parents[2]
OPTIMIZATION_DIR = ROOT / "src" / "rovibrational_excitation" / "optimization"


def test_optimizer_registry_is_exact_and_has_no_fallback() -> None:
    assert ALGO_REGISTRY == {
        "local": local_module.run_local_optimization,
        "krotov": krotov_module.run_krotov_optimization,
        "legacy_batch_overlap": legacy_module.run_legacy_batch_overlap_optimization,
        "grape": grape_module.run_grape_optimization,
    }
    assert optimize_runner.ALGO_REGISTRY is ALGO_REGISTRY


def test_optimizer_runners_use_single_typed_contract_owners() -> None:
    for runner_module in (
        grape_module,
        krotov_module,
        legacy_module,
        local_module,
    ):
        assert runner_module.OptimizationResult is result_module.OptimizationResult
        assert (
            runner_module.IndexedTargetPopulation
            is objective_module.IndexedTargetPopulation
        )

    assert (
        grape_module.DiscreteL2TargetObjective
        is objective_module.DiscreteL2TargetObjective
    )
    assert (
        local_module.DiagonalObservableLocalEvaluator
        is objective_module.DiagonalObservableLocalEvaluator
    )
    assert (
        local_module.TargetOverlapLocalEvaluator
        is objective_module.TargetOverlapLocalEvaluator
    )
    assert (
        legacy_module.parse_legacy_spectral_constraint
        is spectral_module.parse_legacy_spectral_constraint
    )
    assert (
        config_module.validate_algorithm_options
        is options_module.validate_algorithm_options
    )


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    return imported


def test_optimization_does_not_depend_on_upper_workflow_layers() -> None:
    forbidden = (
        "rovibrational_excitation.cli",
        "rovibrational_excitation.io",
        "rovibrational_excitation.simulation",
        "rovibrational_excitation.visualization",
    )
    for path in OPTIMIZATION_DIR.glob("*.py"):
        imports = _imported_modules(path)
        assert not {
            imported for imported in imports if imported.startswith(forbidden)
        }, path


def test_numerical_optimization_owners_do_not_import_workflow_services() -> None:
    forbidden = (
        "yaml",
        "rovibrational_excitation.cli",
        "rovibrational_excitation.fields",
        "rovibrational_excitation.io",
        "rovibrational_excitation.models",
        "rovibrational_excitation.simulation",
        "rovibrational_excitation.visualization",
    )
    for filename in (
        "grape_rk4.py",
        "krotov_rk4.py",
        "objective.py",
        "spectral_constraints.py",
    ):
        path = OPTIMIZATION_DIR / filename
        imports = _imported_modules(path)
        assert not {
            imported for imported in imports if imported.startswith(forbidden)
        }, path


def test_optimization_has_no_broad_exception_suppression() -> None:
    paths = list(OPTIMIZATION_DIR.glob("*.py")) + [
        ROOT / "src" / "rovibrational_excitation" / "simulation" / "optimize_runner.py"
    ]
    for path in paths:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            assert node.type is not None, f"bare except in {path}:{node.lineno}"
            names = {
                child.id for child in ast.walk(node.type) if isinstance(child, ast.Name)
            }
            assert not names & {"Exception", "BaseException"}, (
                f"broad exception suppression in {path}:{node.lineno}"
            )


def test_legacy_reference_artifact_is_internally_consistent() -> None:
    json_path = ROOT / "benchmarks" / "krotov-v0-v3-v0.3.json"
    npz_path = ROOT / "benchmarks" / "krotov-v0-v3-v0.3.npz"
    summary = json.loads(json_path.read_text())

    with np.load(npz_path, allow_pickle=False) as artifact:
        trajectory = artifact["psi_traj"]
        field = artifact["field_V_per_m"]
        final_populations = artifact["final_populations"]
        np.testing.assert_array_equal(
            final_populations,
            np.abs(trajectory[-1]) ** 2,
        )
        assert list(trajectory.shape) == summary["trajectory_shape"]
        assert list(field.shape) == summary["field_shape"]
        assert artifact["time_fs"].shape[0] == trajectory.shape[0]
        assert artifact["field_time_fs"].shape[0] == field.shape[0]

    target_index = summary["target_index"]
    assert summary["algorithm"] == "legacy_batch_overlap"
    assert summary["fidelity"] == final_populations[target_index]
    assert summary["reported_fidelity"] == summary["fidelity"]
    assert summary["independent_trajectory_max_abs_difference"] == 0.0
