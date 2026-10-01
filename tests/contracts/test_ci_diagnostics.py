"""Contracts for publicly observable GitHub Actions failure diagnostics."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTER = ROOT / "scripts" / "report_junit_failures.py"


def _load_reporter():
    spec = importlib.util.spec_from_file_location("report_junit_failures", REPORTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_junit_reporter_escapes_failure_and_error_annotations(tmp_path: Path) -> None:
    report = tmp_path / "results.xml"
    report.write_text(
        """<testsuites><testsuite>
        <testcase classname="tests.test_demo" name="test_percent">
          <failure>line one\nline two 50%</failure>
        </testcase>
        <testcase classname="tests.test_demo" name="test_error,case">
          <error message="boom" />
        </testcase>
        <testcase classname="tests.test_demo" name="test_pass" />
        </testsuite></testsuites>"""
    )

    reporter = _load_reporter()

    assert list(reporter.failure_annotations(report)) == [
        "::error title=tests.test_demo%3A%3Atest_percent::line one%0Aline two 50%25",
        "::error title=tests.test_demo%3A%3Atest_error%2Ccase::boom",
    ]


def test_junit_reporter_exposes_missing_xml_without_masking_failure(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.xml"

    completed = subprocess.run(
        [sys.executable, str(REPORTER), str(missing)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "JUnit diagnostics unavailable" in completed.stdout
    assert str(missing) in completed.stdout
