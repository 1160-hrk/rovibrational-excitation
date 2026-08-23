"""Ownership and dependency contracts for model construction."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src" / "rovibrational_excitation"
MODELS = PACKAGE / "models"
LEGACY_MODELS = PACKAGE / "simulation" / "models"
M_AVERAGE = PACKAGE / "simulation" / "m_average.py"


def test_model_construction_and_m_average_workflow_have_distinct_owners():
    from rovibrational_excitation.models import ModelComponents, build_model
    from rovibrational_excitation.simulation.m_average import (
        MAveragePropagationResult,
        build_m_average_blocks,
        propagate_m_average,
    )

    assert (MODELS / "__init__.py").is_file()
    assert (MODELS / "factory.py").is_file()
    assert (MODELS / "validation.py").is_file()
    assert (MODELS / "linmol.py").is_file()
    assert (MODELS / "twolevel.py").is_file()
    assert (MODELS / "vibladder.py").is_file()
    assert M_AVERAGE.is_file()
    assert not LEGACY_MODELS.exists()
    assert ModelComponents.__module__ == "rovibrational_excitation.models.factory"
    assert build_model.__module__ == "rovibrational_excitation.models.factory"
    assert MAveragePropagationResult.__module__ == (
        "rovibrational_excitation.simulation.m_average"
    )
    assert build_m_average_blocks.__module__ == (
        "rovibrational_excitation.simulation.m_average"
    )
    assert propagate_m_average.__module__ == (
        "rovibrational_excitation.simulation.m_average"
    )


def test_models_has_only_exact_transitional_higher_layer_dependencies():
    forbidden = {
        "cli",
        "dynamics",
        "io",
        "optimization",
        "simulation",
        "spectroscopy",
        "visualization",
    }
    violations: list[str] = []

    for path in sorted(MODELS.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
                level = 0
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                modules = [node.module]
                level = node.level
            else:
                continue
            for module in modules:
                parts = module.split(".")
                if module == "rovibrational_excitation":
                    violations.append(f"{path.relative_to(ROOT)} imports package root")
                elif parts[0] == "rovibrational_excitation" and len(parts) > 1:
                    if parts[1] in forbidden:
                        violations.append(f"{path.relative_to(ROOT)} imports {module}")
                elif level > 1 and parts[0] in forbidden:
                    violations.append(f"{path.relative_to(ROOT)} imports {module}")

    expected_transition_debt = {
        "src/rovibrational_excitation/models/__init__.py imports rovibrational_excitation.dynamics.problem",
        "src/rovibrational_excitation/models/factory.py imports rovibrational_excitation.dynamics.problem",
    }
    assert len(violations) == len(expected_transition_debt)
    assert set(violations) == expected_transition_debt
