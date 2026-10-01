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
ACTIONLINT_CONFIG = ROOT / ".github" / "actionlint.yaml"


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
    test_commands = _commands(jobs["test"])
    physics_commands = _commands(jobs["physics"])
    assert "pytest -q" in test_commands
    assert "python scripts/report_ci_failures.py junit /tmp/test-results.xml" in (
        test_commands
    )
    assert "tests/physics tests/contracts" in physics_commands
    assert "python scripts/report_ci_failures.py junit /tmp/physics-results.xml" in (
        physics_commands
    )
    assert "continue-on-error" not in CI_WORKFLOW.read_text()


def test_ci_enforces_quality_coverage_and_wheel_import():
    jobs = _workflow()["jobs"]

    quality = _commands(jobs["quality"])
    active_scope = "src tests examples benchmarks scripts"
    assert f"ruff check --no-fix {active_scope}" in quality
    assert f"ruff format --check {active_scope}" in quality
    assert "python scripts/smoke_examples.py" in quality
    assert "python examples/tools/build_index.py --check" in quality
    assert "python -m mypy --no-incremental" in quality
    assert "tee /tmp/mypy-output.txt" in quality
    assert (
        'python scripts/report_ci_failures.py text "mypy failed" /tmp/mypy-output.txt'
    ) in quality

    coverage = _commands(jobs["coverage"])
    assert "--data-file=/tmp/rve-coverage" in coverage
    assert "--fail-under=47" in coverage

    build = _commands(jobs["build"])
    assert "python -m build" in build
    assert "twine check" in build
    assert "pip install dist/*.whl" in build
    assert "import rovibrational_excitation" in build

    assert _commands(jobs["container-smoke"]).strip() == "scripts/smoke_container.sh"
    assert set(jobs["required"]["needs"]) == {
        "quality",
        "test",
        "physics",
        "coverage",
        "build",
        "container-smoke",
    }
    required = _commands(jobs["required"])
    assert "CONTAINER_RESULT" in required
    assert "= success" in required


def test_ci_uses_checksum_verified_actionlint_for_declared_runner_labels():
    workflow = _workflow()
    quality_job = workflow["jobs"]["quality"]
    setup_step = next(
        step
        for step in quality_job["steps"]
        if step.get("uses", "").startswith("actions/setup-python@")
    )
    install_step = next(
        step
        for step in quality_job["steps"]
        if step.get("name") == "Install actionlint and ShellCheck"
    )
    lint_step = next(
        step
        for step in quality_job["steps"]
        if step.get("name") == "Lint GitHub Actions workflows"
    )

    assert setup_step["with"]["python-version"] == "3.12"
    assert install_step["env"] == {
        "ACTIONLINT_VERSION": "1.7.12",
        "ACTIONLINT_SHA256": (
            "8aca8db96f1b94770f1b0d72b6dddcb1ebb8123cb3712530b08cc387b349a3d8"
        ),
        "SHELLCHECK_VERSION": "0.11.0",
        "SHELLCHECK_SHA256": (
            "8c3be12b05d5c177a04c29e3c78ce89ac86f1595681cab149b65b97c4e227198"
        ),
    }
    install_command = install_step["run"]
    assert "actionlint_${ACTIONLINT_VERSION}_linux_amd64.tar.gz" in install_command
    assert install_command.count("sha256sum -c -") == 2
    assert "shellcheck-v${SHELLCHECK_VERSION}.linux.x86_64.tar.xz" in install_command
    assert lint_step["run"] == (
        "/tmp/actionlint -no-color -shellcheck /tmp/shellcheck-v0.11.0/shellcheck"
    )

    actionlint_config = yaml.load(ACTIONLINT_CONFIG.read_text(), Loader=yaml.BaseLoader)
    assert actionlint_config == {"self-hosted-runner": {"labels": ["gpu"]}}


def test_v03_checkpoint_version_is_development_or_final():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert pyproject["project"]["version"] in {"0.3.0.dev1", "0.3.0"}


def test_mypy_is_mandatory_only_for_named_typed_modules():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    mypy = pyproject["tool"]["mypy"]

    assert mypy["strict"] is True
    assert mypy["follow_imports"] == "silent"
    assert mypy["files"] == [
        "src/rovibrational_excitation/__init__.py",
        "src/rovibrational_excitation/cli/simulate.py",
        "src/rovibrational_excitation/core/dipole.py",
        "src/rovibrational_excitation/core/model.py",
        "src/rovibrational_excitation/core/execution.py",
        "src/rovibrational_excitation/dynamics/scaling/scales.py",
        "src/rovibrational_excitation/dynamics/base.py",
        "src/rovibrational_excitation/dynamics/capabilities.py",
        "src/rovibrational_excitation/dynamics/factory.py",
        "src/rovibrational_excitation/dynamics/algorithms/rk4/liouville_numpy.py",
        "src/rovibrational_excitation/dynamics/algorithms/rk4/schrodinger_cupy.py",
        "src/rovibrational_excitation/dynamics/algorithms/split_operator/schrodinger_cupy.py",
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
        "src/rovibrational_excitation/io/atomic.py",
        "src/rovibrational_excitation/io/result_schema.py",
        "src/rovibrational_excitation/io/serialization.py",
        "src/rovibrational_excitation/io/storage.py",
        "src/rovibrational_excitation/visualization/result_data.py",
        "src/rovibrational_excitation/visualization/plot_electric_field.py",
        "src/rovibrational_excitation/visualization/plot_electric_field_vector.py",
        "src/rovibrational_excitation/visualization/plot_population.py",
        "src/rovibrational_excitation/visualization/spectrogram.py",
        "src/rovibrational_excitation/spectroscopy/broadening.py",
        "src/rovibrational_excitation/spectroscopy/conditions.py",
        "src/rovibrational_excitation/spectroscopy/observables.py",
        "src/rovibrational_excitation/spectroscopy/projection.py",
        "src/rovibrational_excitation/spectroscopy/report.py",
        "src/rovibrational_excitation/spectroscopy/response.py",
        "src/rovibrational_excitation/spectroscopy/transform.py",
        "src/rovibrational_excitation/simulation/case.py",
        "src/rovibrational_excitation/simulation/convergence.py",
        "src/rovibrational_excitation/simulation/generated.py",
        "src/rovibrational_excitation/simulation/field_preparation.py",
        "src/rovibrational_excitation/simulation/execution.py",
        "src/rovibrational_excitation/simulation/result_persistence.py",
        "src/rovibrational_excitation/simulation/safe_execution.py",
        "src/rovibrational_excitation/simulation/batch.py",
        "src/rovibrational_excitation/simulation/case_paths.py",
        "src/rovibrational_excitation/simulation/reporting.py",
        "src/rovibrational_excitation/simulation/resume.py",
        "src/rovibrational_excitation/simulation/runner.py",
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
        "src/rovibrational_excitation/optimization/__init__.py",
        "src/rovibrational_excitation/optimization/config.py",
        "src/rovibrational_excitation/optimization/grape.py",
        "src/rovibrational_excitation/optimization/grape_rk4.py",
        "src/rovibrational_excitation/optimization/krotov.py",
        "src/rovibrational_excitation/optimization/krotov_controls.py",
        "src/rovibrational_excitation/optimization/krotov_initial_field.py",
        "src/rovibrational_excitation/optimization/krotov_rk4.py",
        "src/rovibrational_excitation/optimization/krotov_timegrid.py",
        "src/rovibrational_excitation/optimization/legacy_batch_overlap.py",
        "src/rovibrational_excitation/optimization/local.py",
        "src/rovibrational_excitation/optimization/local_initialization.py",
        "src/rovibrational_excitation/optimization/model.py",
        "src/rovibrational_excitation/optimization/objective.py",
        "src/rovibrational_excitation/optimization/options.py",
        "src/rovibrational_excitation/optimization/result.py",
        "src/rovibrational_excitation/optimization/spectral_constraints.py",
        "src/rovibrational_excitation/optimization/timegrid.py",
        "src/rovibrational_excitation/simulation/optimize_runner.py",
    ]


def test_build_metadata_uses_supported_spdx_license_and_runtime_dependencies():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())

    assert pyproject["build-system"]["requires"][0] == "setuptools>=77"
    assert pyproject["project"]["license"] == "MIT"
    assert "sympy" in pyproject["project"]["dependencies"]
    dev_dependencies = pyproject["project"]["optional-dependencies"]["dev"]
    assert "ruff==0.16.2" in dev_dependencies
    assert "mypy==1.19.1" in dev_dependencies


def test_supported_examples_are_explicit_and_archives_are_excluded():
    examples = ROOT / "examples"
    supported = {path.name for path in examples.glob("example_*.py")}

    assert supported == {
        "example_external_scalar_field.py",
        "example_typed_spectral_modulation.py",
        "example_typed_twolevel.py",
    }
    archive = examples / "archives" / "v0_2"
    assert (archive / "scripts").is_dir()
    assert (archive / "optimization_configs").is_dir()
    assert not list((examples / "archives").glob("*.py"))
    assert not (examples / "archives" / "v0_2_scripts").exists()
    assert not (examples / "archives" / "v0_2_optimization_configs").exists()

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert pyproject["tool"]["ruff"]["extend-exclude"] == ["examples/archives"]
