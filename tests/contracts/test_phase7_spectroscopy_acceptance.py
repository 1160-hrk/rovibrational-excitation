"""Acceptance audit for the decomposed spectroscopy boundary."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from rovibrational_excitation import spectroscopy
from rovibrational_excitation.spectroscopy import (
    AbsorbanceCalculator,
    ExperimentalConditions,
    SpectroscopyCalculationReport,
    broadening,
    conditions,
    observables,
    report,
    response,
    transform,
)

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src" / "rovibrational_excitation" / "spectroscopy"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    return imported


def test_facade_and_typed_owners_are_single_identity() -> None:
    assert spectroscopy.__all__ == [
        "AbsorbanceCalculator",
        "ExperimentalConditions",
        "SpectroscopyCalculationReport",
        "create_calculator_from_params",
    ]
    assert inspect.getmodule(ExperimentalConditions) is conditions
    assert inspect.getmodule(SpectroscopyCalculationReport) is report
    assert inspect.getmodule(AbsorbanceCalculator).__name__.endswith(
        "absorbance_calculator"
    )


def test_scientific_kernel_owners_are_distinct_and_complete() -> None:
    owners = {
        broadening.apply_doppler_broadening: broadening,
        broadening.apply_device_function: broadening,
        response.calculate_2d_response: response,
        response.calculate_matrix_response: response,
        response.calculate_loop_response: response,
        response.calculate_chunked_response: response,
        transform.radiation_response: transform,
        observables.response_to_absorbance: observables,
    }
    for function, owner in owners.items():
        assert inspect.getmodule(function) is owner


def test_kernel_owners_have_no_upper_layer_dependencies() -> None:
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
    for name in (
        "broadening.py",
        "conditions.py",
        "observables.py",
        "report.py",
        "response.py",
        "transform.py",
    ):
        imported = _imports(PACKAGE / name)
        assert not {item for item in imported if item.startswith(forbidden)}, (
            name,
            imported,
        )


def test_spectroscopy_has_no_broad_or_print_only_failure_path() -> None:
    for path in PACKAGE.glob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                assert node.type is not None, path
                if isinstance(node.type, ast.Name):
                    assert node.type.id not in {"Exception", "BaseException"}, path
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id != "print", path


def test_method_and_pathway_policy_are_required_but_constructor_defaults_remain_debt() -> (
    None
):
    constructor = inspect.signature(AbsorbanceCalculator)
    assert constructor.parameters["phase_matching"].default is inspect.Parameter.empty
    assert constructor.parameters["axes"].default == "xy"
    assert constructor.parameters["pol_int"].default is None
    assert constructor.parameters["pol_det"].default is None

    calculate = inspect.signature(AbsorbanceCalculator.calculate)
    assert calculate.parameters["method"].default is inspect.Parameter.empty
    assert calculate.parameters["wavenumber_units"].default is inspect.Parameter.empty

    source = (PACKAGE / "absorbance_calculator.py").read_text()
    assert "self.axes = axes.lower()" in source
    assert "if pol_int is None:" in source
