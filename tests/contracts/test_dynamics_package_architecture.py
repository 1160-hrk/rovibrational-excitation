"""Import and dependency contracts for the target dynamics package."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src" / "rovibrational_excitation"
DYNAMICS = PACKAGE / "dynamics"
LEGACY_DYNAMICS = PACKAGE / "core" / "propagation"


def test_propagation_is_owned_by_the_target_dynamics_package():
    from rovibrational_excitation.dynamics import (
        LiouvillePropagator,
        PropagationOptions,
        PropagationProblem,
        PropagationResult,
        SchrodingerPropagator,
    )

    assert (DYNAMICS / "__init__.py").is_file()
    assert (DYNAMICS / "problem.py").is_file()
    assert (DYNAMICS / "options.py").is_file()
    assert (DYNAMICS / "result.py").is_file()
    assert (DYNAMICS / "algorithms" / "rk4" / "schrodinger.py").is_file()
    assert not (LEGACY_DYNAMICS / "__init__.py").exists()
    assert not (LEGACY_DYNAMICS / "problem.py").exists()
    assert not (LEGACY_DYNAMICS / "algorithms" / "__init__.py").exists()
    assert PropagationProblem.__module__ == "rovibrational_excitation.dynamics.problem"
    assert PropagationOptions.__module__ == "rovibrational_excitation.dynamics.options"
    assert PropagationResult.__module__ == "rovibrational_excitation.dynamics.result"
    assert SchrodingerPropagator.__module__ == (
        "rovibrational_excitation.dynamics.schrodinger"
    )
    assert (
        LiouvillePropagator.__module__ == "rovibrational_excitation.dynamics.liouville"
    )


def test_dynamics_has_only_explicit_lower_layer_dependencies():
    forbidden = {
        "cli",
        "dipole",
        "io",
        "models",
        "optimization",
        "simulation",
        "spectroscopy",
        "visualization",
    }
    violations: list[str] = []

    for path in sorted(DYNAMICS.rglob("*.py")):
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
        "src/rovibrational_excitation/dynamics/utils.py imports dipole.base",
        "src/rovibrational_excitation/dynamics/scaling/converter.py imports rovibrational_excitation.dipole.base",
    }
    assert len(violations) == len(expected_transition_debt)
    assert set(violations) == expected_transition_debt
