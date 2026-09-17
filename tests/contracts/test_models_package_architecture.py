"""Ownership and dependency contracts for model construction."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src" / "rovibrational_excitation"
MODELS = PACKAGE / "models"
TWO_LEVEL = MODELS / "two_level"
VIB_LADDER = MODELS / "vib_ladder"
LINEAR_MOLECULE = MODELS / "linear_molecule"
LEGACY_MODELS = PACKAGE / "simulation" / "models"
M_AVERAGE = PACKAGE / "simulation" / "m_average.py"


def test_model_construction_and_m_average_workflow_have_distinct_owners():
    import rovibrational_excitation.models.linear_molecule as linear_molecule
    from rovibrational_excitation.models import ModelComponents, build_model
    from rovibrational_excitation.simulation.m_average import (
        MAveragePropagationResult,
        build_m_average_blocks,
        propagate_m_average,
    )

    assert (MODELS / "__init__.py").is_file()
    assert (MODELS / "factory.py").is_file()
    assert (MODELS / "validation.py").is_file()
    assert (LINEAR_MOLECULE / "__init__.py").is_file()
    assert (LINEAR_MOLECULE / "basis.py").is_file()
    assert (LINEAR_MOLECULE / "dipole.py").is_file()
    assert (LINEAR_MOLECULE / "dipole_builder.py").is_file()
    assert (LINEAR_MOLECULE / "model.py").is_file()
    assert (LINEAR_MOLECULE / "parameters.py").is_file()
    assert linear_molecule.__all__ == [
        "LinMolBasis",
        "LinMolDipoleMatrix",
        "LinMolParameters",
        "build_linmol_from_parameters",
        "build_linmol_operators_from_parameters",
    ]
    assert not hasattr(linear_molecule, "build_linmol")
    assert not hasattr(linear_molecule, "build_mu")
    assert not (MODELS / "linmol.py").exists()
    assert not (PACKAGE / "core" / "basis" / "linmol.py").exists()
    assert not (PACKAGE / "dipole" / "linmol").exists()
    assert (TWO_LEVEL / "__init__.py").is_file()
    assert (TWO_LEVEL / "basis.py").is_file()
    assert (TWO_LEVEL / "dipole.py").is_file()
    assert (TWO_LEVEL / "model.py").is_file()
    assert (TWO_LEVEL / "parameters.py").is_file()
    assert not (TWO_LEVEL / "dipole_builder.py").exists()
    assert (VIB_LADDER / "__init__.py").is_file()
    assert (VIB_LADDER / "basis.py").is_file()
    assert (VIB_LADDER / "dipole.py").is_file()
    assert (VIB_LADDER / "model.py").is_file()
    assert (VIB_LADDER / "parameters.py").is_file()
    assert not (VIB_LADDER / "dipole_builder.py").exists()
    assert not (MODELS / "vibladder.py").exists()
    assert not (PACKAGE / "core" / "basis" / "viblad.py").exists()
    assert not (PACKAGE / "dipole" / "viblad").exists()
    assert not (MODELS / "twolevel.py").exists()
    assert not (PACKAGE / "core" / "basis" / "twolevel.py").exists()
    assert not (PACKAGE / "dipole" / "twolevel").exists()
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


def test_models_has_no_higher_layer_dependencies():
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

    assert violations == []
