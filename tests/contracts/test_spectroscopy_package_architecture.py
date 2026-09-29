"""Ownership contracts for the decomposed spectroscopy package."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from rovibrational_excitation.spectroscopy import (
    ExperimentalConditions,
    SpectroscopyCalculationReport,
    absorbance_calculator,
    broadening,
    conditions,
    observables,
    report,
    transform,
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


def test_radiation_transform_has_one_package_owner() -> None:
    assert inspect.getmodule(transform.radiation_response) is transform

    monolith = (SPECTROSCOPY / "absorbance_calculator.py").read_text()
    assert "resp_lin_per_mole += -(" not in monolith


def test_transform_owner_has_no_upper_layer_dependencies() -> None:
    path = SPECTROSCOPY / "transform.py"
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


def test_absorbance_conversion_has_one_package_owner() -> None:
    assert inspect.getmodule(observables.response_to_absorbance) is observables

    monolith = (SPECTROSCOPY / "absorbance_calculator.py").read_text()
    assert "CONSTANTS.EPSILON0" not in monolith
    assert "np.log10(np.exp(1))" not in monolith


def test_observables_owner_has_no_upper_layer_dependencies() -> None:
    path = SPECTROSCOPY / "observables.py"
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


def test_calculation_report_has_one_package_owner() -> None:
    assert SpectroscopyCalculationReport is report.SpectroscopyCalculationReport
    assert (
        absorbance_calculator.SpectroscopyCalculationReport
        is SpectroscopyCalculationReport
    )
    assert inspect.getmodule(SpectroscopyCalculationReport) is report

    monolith = (SPECTROSCOPY / "absorbance_calculator.py").read_text()
    assert "class SpectroscopyCalculationReport" not in monolith


def test_broadening_and_device_kernels_have_one_package_owner() -> None:
    assert inspect.getmodule(broadening.uniform_grid_spacing) is broadening
    assert inspect.getmodule(broadening.filter_complex_gaussian) is broadening
    assert inspect.getmodule(broadening.apply_doppler_broadening) is broadening
    assert inspect.getmodule(broadening.apply_device_function) is broadening

    monolith = (SPECTROSCOPY / "absorbance_calculator.py").read_text()
    assert "from scipy import ndimage" not in monolith
    assert "CONSTANTS.BOLTZMANN" not in monolith


def test_broadening_owner_has_no_upper_layer_dependencies() -> None:
    path = SPECTROSCOPY / "broadening.py"
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
