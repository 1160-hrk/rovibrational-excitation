"""Acceptance contracts for residual Phase 3 package structure."""

from __future__ import annotations

import ast
from pathlib import Path

from rovibrational_excitation.simulation import run_simulation_case
from rovibrational_excitation.simulation.runner import (
    run_simulation_case as runner_entry,
)

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src" / "rovibrational_excitation"


def test_obsolete_simulation_placeholders_and_time_shim_are_removed():
    assert not (PACKAGE / "simulation" / "manager.py").exists()
    assert not (PACKAGE / "simulation" / "timegrid.py").exists()


def test_simulation_package_exposes_one_normal_case_entry_point():
    assert run_simulation_case is runner_entry


def test_generic_parameter_processor_is_removed():
    assert not (PACKAGE / "core" / "units" / "parameter_processor.py").exists()


def test_internal_modules_do_not_import_root_convenience_names():
    violations: list[str] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if path == PACKAGE / "__init__.py":
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0:
                if node.module == "rovibrational_excitation":
                    violations.append(str(path.relative_to(ROOT)))

    assert violations == []
