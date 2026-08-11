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
    assert "ruff check --no-fix src tests" in quality
    assert "ruff format --check src tests" in quality
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
        "src/rovibrational_excitation/core/nondimensional/scales.py",
        "src/rovibrational_excitation/core/units/constants.py",
        "src/rovibrational_excitation/simulation/timegrid.py",
    ]


def test_build_metadata_uses_supported_spdx_license_and_runtime_dependencies():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())

    assert pyproject["build-system"]["requires"][0] == "setuptools>=77"
    assert pyproject["project"]["license"] == "MIT"
    assert "sympy" in pyproject["project"]["dependencies"]
    assert "ruff==0.16.2" in pyproject["project"]["optional-dependencies"]["dev"]
