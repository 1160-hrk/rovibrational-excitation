"""Contracts for the real-CUDA evidence recorder."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmarks" / "run_cuda_evidence.py"
ACCEPTED_REPORT = ROOT / "benchmarks" / "real-cuda-v0.3-b9de848.json"


def _load_module():
    spec = importlib.util.spec_from_file_location("cuda_evidence", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _case(name: str, *, passed: bool = True) -> dict[str, object]:
    return {
        "name": name,
        "algorithm": "rk4" if name.startswith("rk4") else "split",
        "interaction_mode": (
            "helicity_projected"
            if name == "split_helicity_projected"
            else "cartesian"
            if name.startswith("split")
            else None
        ),
        "return_trajectory": name == "rk4_trajectory",
        "backend": "cupy",
        "dtype": "complex128",
        "shape": [1, 4],
        "device_result": True,
        "metrics": {
            "max_abs_difference": 1.0e-14 if passed else 1.0e-3,
            "l2_difference": 2.0e-14 if passed else 2.0e-3,
            "maximum_norm_error": 3.0e-14 if passed else 3.0e-3,
        },
        "tolerances": {
            "max_abs_difference": 2.0e-12,
            "maximum_norm_error": 2.0e-12,
        },
        "timing_ms": {
            "cpu_reference_median": 1.0,
            "gpu_device_input_median": 0.5,
            "gpu_host_input_median": 0.7,
            "explicit_input_copy_median": 0.1,
            "explicit_output_copy_median": 0.1,
        },
        "transfer_bytes": {
            "host_to_device_inputs": 512,
            "device_to_host_result": 64,
        },
        "passed": passed,
    }


def _report(module, *, passed: bool = True) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact": "real-cuda-v0.3",
        "status": "pass" if passed else "fail",
        "source": {"commit": "a" * 40, "worktree_dirty": False},
        "environment": {
            "python": "test",
            "platform": "test",
            "numpy": "test",
            "device_count": 1,
            "device_id": 0,
            "device_name": "test-device",
            "compute_capability": "9.0",
            "cupy": "test",
            "cuda_runtime_version": 12000,
            "cuda_driver_version": 12000,
        },
        "methodology": {
            "synchronization": "CUDA stream synchronized around every timed sample",
            "warmup_runs": 1,
            "timed_repeats": 3,
        },
        "cases": [_case(name, passed=passed) for name in module.REQUIRED_CASES],
        "acceptance": {
            "all_cases_passed": passed,
            "required_cases": list(module.REQUIRED_CASES),
            "speed_is_not_an_acceptance_gate": True,
        },
    }


def test_cuda_evidence_schema_accepts_only_complete_real_device_report() -> None:
    module = _load_module()
    report = _report(module)

    module.validate_report_schema(report)
    module.require_accepted_report(report)


def test_committed_real_cuda_report_is_accepted_and_source_bound() -> None:
    module = _load_module()
    report = json.loads(ACCEPTED_REPORT.read_text())

    module.validate_report_schema(report)
    module.require_accepted_report(report)

    assert report["source"] == {
        "commit": "b9de848cf3fa8322e5f685f60191857e528531d4",
        "worktree_dirty": False,
    }
    assert report["environment"]["device_name"] == "NVIDIA GeForce RTX 5070 Ti"
    assert report["acceptance"]["required_cases"] == list(module.REQUIRED_CASES)


def test_cuda_evidence_schema_rejects_missing_case_and_nonfinite_timing() -> None:
    module = _load_module()
    missing = _report(module)
    missing["cases"] = missing["cases"][:-1]
    with pytest.raises(ValueError, match="exactly these cases"):
        module.validate_report_schema(missing)

    nonfinite = _report(module)
    nonfinite["cases"][0]["timing_ms"]["gpu_device_input_median"] = math.nan
    with pytest.raises(ValueError, match="finite non-negative"):
        module.validate_report_schema(nonfinite)


def test_cuda_evidence_schema_recomputes_case_acceptance() -> None:
    module = _load_module()
    report = _report(module)
    report["cases"][0]["metrics"]["max_abs_difference"] = 1.0

    with pytest.raises(ValueError, match="case acceptance is inconsistent"):
        module.validate_report_schema(report)


def test_cuda_evidence_failed_parity_is_never_accepted() -> None:
    module = _load_module()
    report = _report(module, passed=False)

    module.validate_report_schema(report)
    with pytest.raises(RuntimeError, match="did not pass"):
        module.require_accepted_report(report)


def test_cuda_evidence_workloads_execute_all_cpu_references() -> None:
    module = _load_module()
    problem = module._problem(dimension=4, steps=3, dt=0.01)
    cases = module._case_definitions(problem, dt=0.01)

    assert tuple(case["name"] for case in cases) == module.REQUIRED_CASES
    for case in cases:
        result = module._call_case(case, case["values"], backend="numpy")
        assert result.dtype == np.complex128
        assert result.shape[1] == 4
        assert np.all(np.isfinite(result))


def test_cuda_evidence_source_requires_real_device_and_synchronization() -> None:
    source = SCRIPT.read_text()

    assert "getDeviceCount" in source
    assert "Stream.null.synchronize" in source
    assert "device_result" in source
    assert "cp.asnumpy" in source
    assert "CUDA evidence requires" in source
    assert "fallback" not in source.lower()
