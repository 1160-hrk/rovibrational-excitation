"""Ownership and dependency contracts for persistence helpers."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src" / "rovibrational_excitation"
IO = PACKAGE / "io"
LEGACY_FILES = {
    PACKAGE / "simulation" / "checkpoint.py",
    PACKAGE / "simulation" / "serialization.py",
    PACKAGE / "simulation" / "storage.py",
}


def test_persistence_helpers_are_owned_by_io():
    from rovibrational_excitation.io import (
        CheckpointManager,
        deserialize_polarization,
        json_safe,
        make_results_root,
        update_summary,
    )

    assert (IO / "__init__.py").is_file()
    assert (IO / "checkpoint.py").is_file()
    assert (IO / "serialization.py").is_file()
    assert (IO / "storage.py").is_file()
    assert not any(path.exists() for path in LEGACY_FILES)
    assert CheckpointManager.__module__ == "rovibrational_excitation.io.checkpoint"
    assert json_safe.__module__ == "rovibrational_excitation.io.serialization"
    assert deserialize_polarization.__module__ == (
        "rovibrational_excitation.io.serialization"
    )
    assert make_results_root.__module__ == "rovibrational_excitation.io.storage"
    assert update_summary.__module__ == "rovibrational_excitation.io.storage"


def test_io_does_not_depend_on_higher_level_packages():
    forbidden = {
        "cli",
        "dynamics",
        "fields",
        "models",
        "optimization",
        "simulation",
        "spectroscopy",
        "visualization",
    }
    violations: list[str] = []

    for path in sorted(IO.rglob("*.py")):
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
