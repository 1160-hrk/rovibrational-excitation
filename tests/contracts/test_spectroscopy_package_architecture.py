"""Ownership contracts for the decomposed spectroscopy package."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from rovibrational_excitation.spectroscopy import (
    ExperimentalConditions,
    absorbance_calculator,
    conditions,
)

ROOT = Path(__file__).resolve().parents[2]
SPECTROSCOPY = ROOT / "src" / "rovibrational_excitation" / "spectroscopy"


def test_experimental_conditions_have_one_package_owner() -> None:
    assert ExperimentalConditions is conditions.ExperimentalConditions
    assert absorbance_calculator.ExperimentalConditions is ExperimentalConditions
    assert inspect.getmodule(ExperimentalConditions) is conditions

    monolith = (SPECTROSCOPY / "absorbance_calculator.py").read_text()
    assert "class ExperimentalConditions" not in monolith


def test_conditions_owner_depends_only_on_core_and_array_support() -> None:
    path = SPECTROSCOPY / "conditions.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)

    forbidden = (
        "rovibrational_excitation.cli",
        "rovibrational_excitation.dynamics",
        "rovibrational_excitation.fields",
        "rovibrational_excitation.io",
        "rovibrational_excitation.models",
        "rovibrational_excitation.optimization",
        "rovibrational_excitation.simulation",
        "rovibrational_excitation.visualization",
    )
    assert not {name for name in imported if name.startswith(forbidden)}, imported
