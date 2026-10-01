"""Ownership contracts for the decomposed spectroscopy package."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import rovibrational_excitation.spectroscopy as spectroscopy
from rovibrational_excitation.spectroscopy import (
    CartesianAnalyzerProjection,
    ComplexResponseSpectrum,
    ExperimentalConditions,
    SpectroscopyCalculationReport,
    absorbance_calculator,
    broadening,
    conditions,
    observables,
    report,
    response,
    result,
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


def test_dense_response_kernels_have_one_package_owner() -> None:
    assert inspect.getmodule(response.prepare_2d_denominators) is response
    assert inspect.getmodule(response.calculate_2d_response) is response
    assert inspect.getmodule(response.calculate_matrix_response) is response
    assert inspect.getmodule(response.calculate_loop_response) is response
    assert inspect.getmodule(response.sparse_commutator) is response
    assert inspect.getmodule(response.select_response_entries) is response
    assert inspect.getmodule(response.calculate_chunked_response) is response

    monolith = (SPECTROSCOPY / "absorbance_calculator.py").read_text()
    assert "intensity_factors" not in monolith
    assert "responses.append" not in monolith
    assert "resp_lin_per_mole += response" not in monolith
    assert "from scipy.sparse import csr_matrix" not in monolith
    assert "relative_threshold * scale" not in monolith


def test_response_owner_has_no_upper_layer_dependencies() -> None:
    path = SPECTROSCOPY / "response.py"
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


def test_response_routes_do_not_own_observable_conversion() -> None:
    path = SPECTROSCOPY / "absorbance_calculator.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    calculator = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AbsorbanceCalculator"
    )
    methods = {
        node.name: node
        for node in calculator.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    route_names = (
        "_calculate_2d",
        "_calculate_matrix",
        "_calculate_loop",
        "_calculate_chunked",
    )
    for name in route_names:
        calls = {
            node.func.attr
            for node in ast.walk(methods[name])
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert "_response_to_absorbance" not in calls, name

    calculate_calls = [
        node
        for node in ast.walk(methods["calculate"])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_response_to_absorbance"
    ]
    assert len(calculate_calls) == 1


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


def test_cartesian_analyzer_projection_has_one_package_owner() -> None:
    assert inspect.getmodule(CartesianAnalyzerProjection).__name__.endswith(
        "projection"
    )


def test_complex_response_result_has_one_package_owner() -> None:
    assert ComplexResponseSpectrum is result.ComplexResponseSpectrum
    assert inspect.getmodule(ComplexResponseSpectrum) is result

    monolith = (SPECTROSCOPY / "absorbance_calculator.py").read_text()
    assert "class ComplexResponseSpectrum" not in monolith


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


def test_spectroscopy_package_has_no_competing_distribution_metadata() -> None:
    source = (SPECTROSCOPY / "__init__.py").read_text()

    for name in ("__version__", "__author__", "__email__"):
        assert name not in spectroscopy.__dict__
        assert name not in source
    assert "contact@example.com" not in source
