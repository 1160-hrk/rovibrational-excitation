"""Contracts for mandatory GitHub Actions quality gates."""

from __future__ import annotations

from pathlib import Path

import yaml

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[2]
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def _workflow() -> dict:
    return yaml.load(CI_WORKFLOW.read_text(), Loader=yaml.BaseLoader)


def _commands(job: dict) -> str:
    return "\n".join(step.get("run", "") for step in job["steps"])


def test_one_workflow_enforces_declared_python_and_physics_matrix():
    workflow = _workflow()
    jobs = workflow["jobs"]

    assert not (ROOT / ".github" / "workflows" / "tests.yml").exists()
    assert workflow["on"]["push"]["branches"] == [
        "main",
        "develop",
        "refactor/**",
    ]
    assert workflow["on"]["pull_request"]["branches"] == ["main", "develop"]
    assert workflow["on"]["workflow_dispatch"] == {}
    assert jobs["test"]["strategy"]["matrix"]["python-version"] == [
        "3.10",
        "3.11",
        "3.12",
        "3.13",
    ]
    assert "pytest -q" in _commands(jobs["test"])
    assert "tests/physics tests/contracts" in _commands(jobs["physics"])
    assert "continue-on-error" not in CI_WORKFLOW.read_text()


def test_ci_enforces_quality_coverage_and_wheel_import():
    jobs = _workflow()["jobs"]

    quality = _commands(jobs["quality"])
    active_scope = "src tests examples benchmarks scripts"
    assert f"ruff check --no-fix {active_scope}" in quality
    assert f"ruff format --check {active_scope}" in quality
    assert "python scripts/smoke_examples.py" in quality
    assert "mypy" in quality

    coverage = _commands(jobs["coverage"])
    assert "--data-file=/tmp/rve-coverage" in coverage
    assert "--fail-under=47" in coverage

    build = _commands(jobs["build"])
    assert "python -m build" in build
    assert "twine check" in build
    assert "pip install dist/*.whl" in build
    assert "import rovibrational_excitation" in build


def test_mypy_is_mandatory_only_for_named_typed_modules():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    mypy = pyproject["tool"]["mypy"]

    assert mypy["strict"] is True
    assert mypy["follow_imports"] == "silent"
    assert mypy["files"] == [
        "src/rovibrational_excitation/core/dipole.py",
        "src/rovibrational_excitation/core/model.py",
        "src/rovibrational_excitation/core/execution.py",
        "src/rovibrational_excitation/dynamics/scaling/scales.py",
        "src/rovibrational_excitation/dynamics/base.py",
        "src/rovibrational_excitation/dynamics/capabilities.py",
        "src/rovibrational_excitation/dynamics/factory.py",
        "src/rovibrational_excitation/dynamics/algorithms/rk4/liouville_numpy.py",
        "src/rovibrational_excitation/dynamics/liouville.py",
        "src/rovibrational_excitation/dynamics/mixed_state.py",
        "src/rovibrational_excitation/dynamics/options.py",
        "src/rovibrational_excitation/dynamics/problem.py",
        "src/rovibrational_excitation/dynamics/result.py",
        "src/rovibrational_excitation/dynamics/schrodinger.py",
        "src/rovibrational_excitation/core/time.py",
        "src/rovibrational_excitation/fields/sampled.py",
        "src/rovibrational_excitation/core/states.py",
        "src/rovibrational_excitation/core/units/constants.py",
        "src/rovibrational_excitation/core/units/scalar_quantities.py",
        "src/rovibrational_excitation/io/checkpoint.py",
        "src/rovibrational_excitation/io/serialization.py",
        "src/rovibrational_excitation/simulation/case.py",
        "src/rovibrational_excitation/simulation/convergence.py",
        "src/rovibrational_excitation/simulation/generated.py",
        "src/rovibrational_excitation/simulation/field_preparation.py",
        "src/rovibrational_excitation/simulation/execution.py",
        "src/rovibrational_excitation/simulation/result_persistence.py",
        "src/rovibrational_excitation/simulation/safe_execution.py",
        "src/rovibrational_excitation/simulation/batch.py",
        "src/rovibrational_excitation/models/_parameter_validation.py",
        "src/rovibrational_excitation/models/parameters.py",
        "src/rovibrational_excitation/models/linear_molecule/parameters.py",
        "src/rovibrational_excitation/models/two_level/parameters.py",
        "src/rovibrational_excitation/models/vib_ladder/parameters.py",
        "src/rovibrational_excitation/models/validation.py",
        "src/rovibrational_excitation/models/symmetry/groups.py",
        "src/rovibrational_excitation/models/symmetry/policy.py",
        "src/rovibrational_excitation/models/symmetry/presets.py",
        "src/rovibrational_excitation/models/symmetric_top/rotational.py",
        "src/rovibrational_excitation/models/symmetric_top/basis.py",
        "src/rovibrational_excitation/models/symmetric_top/dipole.py",
        "src/rovibrational_excitation/models/symmetric_top/model.py",
        "src/rovibrational_excitation/optimization/config.py",
        "src/rovibrational_excitation/optimization/krotov_initial_field.py",
        "src/rovibrational_excitation/optimization/local_initialization.py",
        "src/rovibrational_excitation/optimization/model.py",
        "src/rovibrational_excitation/optimization/options.py",
        "src/rovibrational_excitation/optimization/spectral_constraints.py",
    ]


def test_build_metadata_uses_supported_spdx_license_and_runtime_dependencies():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())

    assert pyproject["build-system"]["requires"][0] == "setuptools>=77"
    assert pyproject["project"]["license"] == "MIT"
    assert "sympy" in pyproject["project"]["dependencies"]
    assert "ruff==0.16.2" in pyproject["project"]["optional-dependencies"]["dev"]


def test_supported_examples_are_explicit_and_archives_are_excluded():
    examples = ROOT / "examples"
    supported = {path.name for path in examples.glob("example_*.py")}

    assert supported == {
        "example_external_scalar_field.py",
        "example_typed_spectral_modulation.py",
        "example_typed_twolevel.py",
    }
    assert (examples / "archives" / "v0_2_scripts").is_dir()

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert pyproject["tool"]["ruff"]["extend-exclude"] == ["examples/archives"]
