"""Contracts for repository-owned coverage enforcement without stale services."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def _workflow() -> dict:
    return yaml.load(CI_WORKFLOW.read_text(), Loader=yaml.BaseLoader)


def _commands(job: dict) -> str:
    return "\n".join(step.get("run", "") for step in job["steps"])


def test_coverage_has_one_local_ci_authority_and_no_unwired_codecov_files() -> None:
    assert not (ROOT / "codecov.yml").exists()
    assert not (ROOT / "docs" / "CODECOV_SETUP.md").exists()

    workflow_text = CI_WORKFLOW.read_text().lower()
    assert "codecov" not in workflow_text

    coverage = _workflow()["jobs"]["coverage"]
    commands = _commands(coverage)
    assert "coverage run --data-file=/tmp/rve-coverage" in commands
    assert "coverage report --data-file=/tmp/rve-coverage" in commands
    assert "--fail-under=47" in commands
    assert "coverage xml --data-file=/tmp/rve-coverage -o /tmp/coverage.xml" in commands

    artifact = next(
        step
        for step in coverage["steps"]
        if step.get("name") == "Upload coverage report"
    )
    assert artifact["uses"] == "actions/upload-artifact@v4"
    assert "/tmp/coverage-report.txt" in artifact["with"]["path"]
    assert "/tmp/coverage.xml" in artifact["with"]["path"]


def test_public_readmes_do_not_claim_codecov_evidence() -> None:
    for name in ("README.md", "README_JP.md"):
        text = (ROOT / name).read_text().lower()
        assert "codecov" not in text
        assert "graph/badge.svg" not in text
