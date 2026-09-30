"""Contracts for repository examples and release/development tooling."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
RELEASE_SCRIPT = ROOT / "scripts" / "release.py"
RELEASE_WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"
CUDA_WORKFLOW = ROOT / ".github" / "workflows" / "cuda-validation.yml"
JUPYTER_SCRIPT = ROOT / "scripts" / "start_jupyter.sh"
INDEX_SCRIPT = ROOT / "examples" / "tools" / "build_index.py"
TEST_GUIDE = ROOT / "tests" / "README.md"


def _load_release_module():
    spec = importlib.util.spec_from_file_location("repository_release", RELEASE_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _workflow() -> dict:
    return yaml.load(RELEASE_WORKFLOW.read_text(), Loader=yaml.BaseLoader)


def _cuda_workflow() -> dict:
    return yaml.load(CUDA_WORKFLOW.read_text(), Loader=yaml.BaseLoader)


def _commands(job: dict) -> str:
    return "\n".join(step.get("run", "") for step in job["steps"])


def test_release_version_transition_accepts_dev_to_matching_final() -> None:
    release = _load_release_module()

    assert release.validate_release_transition("0.3.0.dev1", "0.3.0") == (
        (0, 3, 0),
        (0, 3, 0),
    )
    assert release.validate_release_transition("0.3.0", "0.3.1") == (
        (0, 3, 0),
        (0, 3, 1),
    )
    with pytest.raises(ValueError, match="final X.Y.Z"):
        release.validate_release_transition("0.3.0.dev1", "0.3.0rc1")
    with pytest.raises(ValueError, match="newer"):
        release.validate_release_transition("0.3.0", "0.3.0")


def test_release_dry_run_is_read_only_and_supports_current_version() -> None:
    release = _load_release_module()
    current = release.read_current_version()
    target = {"0.3.0.dev1": "0.3.0", "0.3.0": "0.3.1"}[current]
    before = (ROOT / "pyproject.toml").read_bytes()
    completed = subprocess.run(
        [sys.executable, str(RELEASE_SCRIPT), target, "--dry-run"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert f"{current} -> {target}" in completed.stdout
    assert (ROOT / "pyproject.toml").read_bytes() == before


def test_local_release_tool_never_commits_tags_or_pushes() -> None:
    source = RELEASE_SCRIPT.read_text()

    for forbidden in ("git commit", "git tag", "git push", "input("):
        assert forbidden not in source
    assert "--apply" in source
    assert "pytest" in source
    assert "twine" in source


def test_release_workflow_requires_final_version_cpu_and_real_gpu_gates() -> None:
    workflow = _workflow()
    jobs = workflow["jobs"]
    verify = _commands(jobs["verify-version"])
    cpu = _commands(jobs["cpu-release-gates"])
    gpu = _commands(jobs["gpu-validation"])

    assert "final X.Y.Z" in verify
    assert "pytest -q" in cpu
    assert "ruff check --no-fix" in cpu
    assert "python scripts/smoke_examples.py" in cpu
    assert "mypy" in cpu
    assert jobs["gpu-validation"]["runs-on"] == ["self-hosted", "linux", "x64", "gpu"]
    assert "getDeviceCount" in gpu
    assert "test_numpy_and_cupy_final_state_agree" in gpu
    assert "benchmarks/run_cuda_evidence.py" in gpu
    gpu_artifacts = [
        step
        for step in jobs["gpu-validation"]["steps"]
        if step.get("uses") == "actions/upload-artifact@v4"
    ]
    assert len(gpu_artifacts) == 1
    assert gpu_artifacts[0]["if"] == "always()"
    assert gpu_artifacts[0]["with"]["name"] == "real-cuda-evidence"
    assert gpu_artifacts[0]["with"]["if-no-files-found"] == "error"
    assert (
        _commands(jobs["container-validation"]).strip() == "scripts/smoke_container.sh"
    )
    assert set(jobs["build-and-test"]["needs"]) == {
        "verify-version",
        "cpu-release-gates",
        "gpu-validation",
        "container-validation",
    }
    assert jobs["publish-pypi"]["needs"] == ["verify-version", "build-and-test"]
    assert jobs["create-release"]["needs"] == [
        "verify-version",
        "build-and-test",
        "publish-pypi",
    ]
    release_steps = jobs["create-release"]["steps"]
    assert any(
        step.get("uses") == "actions/download-artifact@v4"
        and step.get("with", {}).get("name") == "real-cuda-evidence"
        for step in release_steps
    )
    assert "cuda-evidence/*.json" in _commands(jobs["create-release"])


def test_manual_cuda_workflow_records_pre_tag_evidence() -> None:
    workflow = _cuda_workflow()
    assert set(workflow["on"]) == {"workflow_dispatch"}
    assert workflow["permissions"] == {"contents": "read"}

    job = workflow["jobs"]["cuda-validation"]
    commands = _commands(job)
    assert job["runs-on"] == ["self-hosted", "linux", "x64", "gpu"]
    assert "getDeviceCount" in commands
    assert "pytest -q -m gpu" in commands
    assert "benchmarks/run_cuda_evidence.py" in commands

    artifacts = [
        step
        for step in job["steps"]
        if step.get("uses") == "actions/upload-artifact@v4"
    ]
    assert len(artifacts) == 1
    assert artifacts[0]["if"] == "always()"
    assert artifacts[0]["with"]["if-no-files-found"] == "error"
    assert artifacts[0]["with"]["retention-days"] == "90"


def test_jupyter_launcher_has_safe_local_authenticated_defaults() -> None:
    source = JUPYTER_SCRIPT.read_text()

    assert "127.0.0.1" in source
    for forbidden in (
        "$HOME/.jupyter",
        "disable_check_xsrf=True",
        "allow_origin='*'",
        "token=''",
        "password=''",
        "root_dir = '/workspace'",
    ):
        assert forbidden not in source


def test_example_index_and_template_are_checked_by_ci_smoke() -> None:
    completed = subprocess.run(
        [sys.executable, str(INDEX_SCRIPT), "--check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    index_source = INDEX_SCRIPT.read_text()
    smoke_source = (ROOT / "scripts" / "smoke_examples.py").read_text()
    assert "rglob" not in index_source
    assert 'glob("example_*.py")' in index_source
    assert "params_template.py" in smoke_source


def test_pyproject_is_the_only_dependency_manifest() -> None:
    assert not (ROOT / "requirements.txt").exists()
    assert not (ROOT / "requirements-dev.txt").exists()

    pyproject = (ROOT / "pyproject.toml").read_text()
    assert "[project]" in pyproject
    assert "[project.optional-dependencies]" in pyproject


def test_test_guide_uses_current_installation_and_ci_commands() -> None:
    text = TEST_GUIDE.read_text()

    assert 'pip install -e ".[dev,io,plot]"' in text
    assert "coverage run --data-file=/tmp/rve-coverage" in text
    assert "pytest -m gpu" in text
    assert ".github/workflows/ci.yml" in text
    for stale in (
        "63%",
        "requirements.txt",
        "requirements-dev.txt",
        "python run_tests.py",
        "actions/checkout@v2",
        "python-version: 3.9",
        "NUMBA_DISABLE_JIT",
        'pytest -k "not cupy"',
    ):
        assert stale not in text
