"""Import and dependency contracts for the target fields package."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src" / "rovibrational_excitation"
FIELDS = PACKAGE / "fields"
LEGACY_FIELDS = PACKAGE / "core" / "electric_field"


def test_electric_field_is_owned_by_the_target_fields_package():
    from rovibrational_excitation.fields import (
        CartesianField,
        ElectricField,
        ScalarField,
        apply_dispersion,
        gaussian,
    )

    assert (FIELDS / "__init__.py").is_file()
    assert (FIELDS / "field.py").is_file()
    assert (FIELDS / "envelopes.py").is_file()
    assert (FIELDS / "modulation.py").is_file()
    assert (FIELDS / "sampled.py").is_file()
    assert not (LEGACY_FIELDS / "__init__.py").exists()
    assert not (LEGACY_FIELDS / "core.py").exists()
    assert not (LEGACY_FIELDS / "envelopes.py").exists()
    assert not (LEGACY_FIELDS / "modulation.py").exists()
    assert ElectricField.__module__ == "rovibrational_excitation.fields.field"
    assert ScalarField.__module__ == "rovibrational_excitation.fields.sampled"
    assert CartesianField.__module__ == "rovibrational_excitation.fields.sampled"
    assert gaussian.__module__ == "rovibrational_excitation.fields.envelopes"
    assert apply_dispersion.__module__ == "rovibrational_excitation.fields.modulation"


def test_fields_import_only_core_and_third_party_dependencies():
    forbidden = {
        "cli",
        "dynamics",
        "io",
        "models",
        "optimization",
        "simulation",
        "spectroscopy",
        "visualization",
    }
    violations: list[str] = []

    for path in sorted(FIELDS.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                modules = [node.module]
            else:
                continue
            for module in modules:
                parts = module.split(".")
                if module == "rovibrational_excitation":
                    violations.append(f"{path.relative_to(ROOT)} imports package root")
                elif parts[0] == "rovibrational_excitation" and len(parts) > 1:
                    if parts[1] in forbidden:
                        violations.append(f"{path.relative_to(ROOT)} imports {module}")

    assert violations == []
