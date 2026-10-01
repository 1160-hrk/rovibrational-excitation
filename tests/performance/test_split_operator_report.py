"""Contracts for split-operator setup and propagation benchmark reporting."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.run_split_operator import _workload  # noqa: E402

REPORT_PATH = ROOT / "benchmarks" / "split-polarization-v0.3.json"

pytestmark = pytest.mark.performance


def test_split_benchmark_executes_public_setup_and_inner_paths() -> None:
    result = _workload(j_max=1, steps=5, dt=0.01, repeats=1)

    assert result["inner_vs_public_l2"] == {
        "split_cartesian": 0.0,
        "split_helicity_projected": 0.0,
    }
    component_times = result["component_median_ms"]
    assert set(component_times) == {
        "cartesian_spectral_setup",
        "cartesian_inner_propagation",
        "helicity_projected_spectral_setup",
        "helicity_projected_inner_propagation",
    }
    assert all(value > 0.0 for value in component_times.values())


def test_committed_split_report_separates_setup_and_propagation() -> None:
    report = json.loads(REPORT_PATH.read_text())

    assert report["methodology"]["timed_scope"] == {
        "public": "validation, preparation, eigendecomposition, propagation",
        "spectral_setup": (
            "interaction construction where applicable, eigendecomposition, "
            "and contiguous spectral arrays"
        ),
        "inner_propagation": "prepared NumPy/Numba time loop only",
    }
    assert report["environment"]["gpu"].startswith("not run")
    for workload in report["workloads"]:
        assert workload["inner_vs_public_l2"] == {
            "split_cartesian": 0.0,
            "split_helicity_projected": 0.0,
        }
