"""Import and dependency contracts for the mechanical package migration."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src" / "rovibrational_excitation"
CORE = PACKAGE / "core"


def test_hamiltonian_is_owned_by_the_target_core_operator_module():
    from rovibrational_excitation.core.operators import Hamiltonian

    assert (CORE / "__init__.py").is_file()
    assert (CORE / "operators.py").is_file()
    assert not (CORE / "basis" / "hamiltonian.py").exists()
    assert Hamiltonian.__module__ == "rovibrational_excitation.core.operators"


def test_core_has_no_imports_from_higher_application_layers():
    forbidden = {
        "cli",
        "dynamics",
        "fields",
        "io",
        "models",
        "optimization",
        "simulation",
        "spectroscopy",
        "visualization",
    }
    violations: list[str] = []

    for path in sorted(CORE.rglob("*.py")):
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

    assert violations == []
