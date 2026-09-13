"""Contracts for the Liouville endpoint-Hamiltonian reuse benchmark."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.run_liouville_endpoint_reuse import (  # noqa: E402
    DEFAULT_OUTPUT,
    DEFAULT_WORKLOADS,
    benchmark_workload,
    build_problem,
)

pytestmark = pytest.mark.performance


def test_endpoint_reuse_benchmark_problem_is_physical_and_deterministic() -> None:
    first = build_problem(3, 4)
    second = build_problem(3, 4)

    for first_array, second_array in zip(first, second, strict=True):
        np.testing.assert_array_equal(first_array, second_array)
    for operator in first[:3]:
        np.testing.assert_array_equal(operator, operator.conj().T)
    np.testing.assert_allclose(np.trace(first[-1]), 1.0, rtol=0.0, atol=1.0e-15)
    assert first[3].shape == (9,)
    assert first[4].shape == (9,)


def test_endpoint_reuse_benchmark_executes_with_exact_legacy_parity() -> None:
    result = benchmark_workload(dimension=3, steps=4, repeats=1)

    assert result["exact_equal"] is True
    assert result["max_abs_difference"] == 0.0
    assert result["legacy_hamiltonian_builds"] == 12
    assert result["endpoint_reuse_hamiltonian_builds"] == 9
    assert result["avoided_hamiltonian_builds"] == 3
    assert result["avoided_hamiltonian_allocation_traffic_bytes"] == 3 * 3 * 3 * 16


def test_committed_endpoint_reuse_report_has_complete_exact_results() -> None:
    report = json.loads(DEFAULT_OUTPUT.read_text())

    assert report["schema_version"] == 1
    assert report["artifact"] == "liouville-endpoint-reuse-v0.3"
    results = report["results"]
    assert [
        (result["dimension"], result["propagation_steps"]) for result in results
    ] == list(DEFAULT_WORKLOADS)
    assert all(result["exact_equal"] for result in results)
    assert all(result["max_abs_difference"] == 0.0 for result in results)
