"""Record auditable parity and timing evidence on a real CUDA device.

The recorder has no CPU-only success path. It writes a diagnostic JSON report
before raising when a completed CUDA run fails its numerical acceptance gates.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np

from rovibrational_excitation.dynamics.algorithms.rk4.schrodinger import (
    rk4_schrodinger,
)
from rovibrational_excitation.dynamics.algorithms.split_operator.schrodinger import (
    splitop_schrodinger,
)

REQUIRED_CASES = (
    "rk4_final",
    "rk4_trajectory",
    "split_static_cartesian",
    "split_rotating_cartesian",
    "split_helicity_projected",
)
CASE_CONTRACTS = {
    "rk4_final": ("rk4", None, False),
    "rk4_trajectory": ("rk4", None, True),
    "split_static_cartesian": ("split", "cartesian", False),
    "split_rotating_cartesian": ("split", "cartesian", False),
    "split_helicity_projected": ("split", "helicity_projected", False),
}
TIMING_KEYS = (
    "cpu_reference_median",
    "gpu_device_input_median",
    "gpu_host_input_median",
    "explicit_input_copy_median",
    "explicit_output_copy_median",
)
MAX_ABS_TOLERANCE = 2.0e-10
NORM_TOLERANCE = 2.0e-10


def _source_state() -> dict[str, object]:
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "worktree_dirty": bool(git("status", "--porcelain")),
    }


def _require_real_cuda() -> Any:
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError(
            "CUDA evidence requires CuPy and a compatible CUDA runtime"
        ) from exc

    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        raise RuntimeError(
            "CUDA evidence requires a working CUDA runtime and visible device"
        ) from exc
    if device_count < 1:
        raise RuntimeError("CUDA evidence requires at least one visible CUDA device")
    cp.cuda.Device(0).use()
    cp.cuda.Stream.null.synchronize()
    return cp


def _problem(dimension: int, steps: int, dt: float) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260930)
    h0 = np.diag(np.linspace(0.05, 0.95, dimension)).astype(np.complex128)
    mu_x = np.zeros((dimension, dimension), dtype=np.complex128)
    couplings = np.linspace(0.015, 0.045, dimension - 1)
    indices = np.arange(dimension - 1)
    mu_x[indices, indices + 1] = couplings
    mu_x[indices + 1, indices] = couplings

    magnetic = np.arange(dimension, dtype=np.float64)
    quarter_rotation = np.exp(0.5j * np.pi * magnetic)
    mu_y = quarter_rotation[:, None] * mu_x * quarter_rotation.conj()[None, :]

    initial = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
    initial = np.asarray(initial / np.linalg.norm(initial), dtype=np.complex128)

    field_time = np.arange(2 * steps + 1, dtype=np.float64) * (dt / 2.0)
    center = steps * dt / 2.0
    width = max(steps * dt / 5.0, dt)
    envelope = 0.035 * np.exp(-0.5 * ((field_time - center) / width) ** 2)

    rotating_angle = 0.73 * field_time
    rotating_x = envelope * np.cos(rotating_angle)
    rotating_y = envelope * np.sin(rotating_angle)

    fixed_scalar = envelope * np.cos(0.19 * field_time)
    fixed_angle = 0.37
    static_x = fixed_scalar * np.cos(fixed_angle)
    static_y = fixed_scalar * np.sin(fixed_angle)

    projected_scalar = envelope * np.cos(0.41 * field_time)
    polarization = np.array([1.0, 1.0j], dtype=np.complex128) / np.sqrt(2.0)
    zeros = np.zeros_like(projected_scalar)
    return {
        "H0": h0,
        "mu_x": mu_x,
        "mu_y": mu_y,
        "psi": initial,
        "magnetic": magnetic,
        "rotating_x": rotating_x,
        "rotating_y": rotating_y,
        "static_x": static_x,
        "static_y": static_y,
        "projected_scalar": projected_scalar,
        "polarization": polarization,
        "zeros": zeros,
    }


def _case_definitions(
    problem: dict[str, np.ndarray],
    dt: float,
) -> list[dict[str, object]]:
    common = {
        "H0": problem["H0"],
        "mu_x": problem["mu_x"],
        "mu_y": problem["mu_y"],
        "psi": problem["psi"],
    }

    def values(**extra: np.ndarray) -> dict[str, np.ndarray]:
        return {**common, **extra}

    return [
        {
            "name": "rk4_final",
            "algorithm": "rk4",
            "values": values(
                field_x=problem["rotating_x"],
                field_y=problem["rotating_y"],
            ),
            "dt": dt,
            "return_traj": False,
            "stride": 1,
        },
        {
            "name": "rk4_trajectory",
            "algorithm": "rk4",
            "values": values(
                field_x=problem["rotating_x"],
                field_y=problem["rotating_y"],
            ),
            "dt": dt,
            "return_traj": True,
            "stride": 1,
        },
        {
            "name": "split_static_cartesian",
            "algorithm": "split",
            "values": values(
                field_x=problem["static_x"],
                field_y=problem["static_y"],
            ),
            "dt": dt,
            "return_traj": False,
            "stride": 1,
            "interaction_mode": "cartesian",
        },
        {
            "name": "split_rotating_cartesian",
            "algorithm": "split",
            "values": values(
                field_x=problem["rotating_x"],
                field_y=problem["rotating_y"],
                magnetic=problem["magnetic"],
            ),
            "dt": dt,
            "return_traj": False,
            "stride": 1,
            "interaction_mode": "cartesian",
        },
        {
            "name": "split_helicity_projected",
            "algorithm": "split",
            "values": values(
                field_x=problem["zeros"],
                field_y=problem["zeros"],
                polarization=problem["polarization"],
                scalar_field=problem["projected_scalar"],
            ),
            "dt": dt,
            "return_traj": False,
            "stride": 1,
            "interaction_mode": "helicity_projected",
        },
    ]


def _call_case(
    case: dict[str, object],
    values: dict[str, Any],
    *,
    backend: str,
) -> Any:
    common = (
        values["H0"],
        values["mu_x"],
        values["mu_y"],
        values["field_x"],
        values["field_y"],
        values["psi"],
        case["dt"],
    )
    if case["algorithm"] == "rk4":
        return rk4_schrodinger(
            *common,
            return_traj=bool(case["return_traj"]),
            stride=int(case["stride"]),
            renorm=False,
            backend=backend,
        )

    return splitop_schrodinger(
        *common,
        return_traj=bool(case["return_traj"]),
        sample_stride=int(case["stride"]),
        interaction_mode=str(case["interaction_mode"]),
        magnetic_quantum_numbers=values.get("magnetic"),
        polarization=values.get("polarization"),
        scalar_field=values.get("scalar_field"),
        renorm=False,
        backend=backend,
    )


def _measure_cpu(call: Callable[[], Any], repeats: int) -> tuple[float, Any]:
    call()
    samples = []
    output = None
    for _ in range(repeats):
        start = time.perf_counter_ns()
        output = call()
        samples.append((time.perf_counter_ns() - start) / 1.0e6)
    return float(median(samples)), output


def _measure_cuda(
    cp: Any,
    call: Callable[[], Any],
    repeats: int,
) -> tuple[float, Any]:
    output = call()
    cp.cuda.Stream.null.synchronize()
    samples = []
    for _ in range(repeats):
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter_ns()
        output = call()
        cp.cuda.Stream.null.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1.0e6)
    return float(median(samples)), output


def _device_values(cp: Any, values: dict[str, np.ndarray]) -> dict[str, Any]:
    return {name: cp.asarray(value) for name, value in values.items()}


def _case_record(cp: Any, case: dict[str, object], repeats: int) -> dict[str, object]:
    host_values = case["values"]
    assert isinstance(host_values, dict)
    device_values = _device_values(cp, host_values)
    cp.cuda.Stream.null.synchronize()

    cpu_ms, expected = _measure_cpu(
        lambda: _call_case(case, host_values, backend="numpy"),
        repeats,
    )
    device_ms, actual = _measure_cuda(
        cp,
        lambda: _call_case(case, device_values, backend="cupy"),
        repeats,
    )
    host_ms, host_actual = _measure_cuda(
        cp,
        lambda: _call_case(case, host_values, backend="cupy"),
        repeats,
    )
    input_copy_ms, copied_values = _measure_cuda(
        cp,
        lambda: _device_values(cp, host_values),
        repeats,
    )
    del copied_values

    output_copy_ms, actual_host = _measure_cuda(
        cp,
        lambda: cp.asnumpy(actual),
        repeats,
    )
    cp.cuda.Stream.null.synchronize()

    host_actual_array = cp.asnumpy(host_actual)
    cp.cuda.Stream.null.synchronize()
    expected_array = np.asarray(expected)
    actual_array = np.asarray(actual_host)
    device_difference = actual_array - expected_array
    host_difference = host_actual_array - expected_array
    norms = np.linalg.norm(actual_array, axis=1)
    host_norms = np.linalg.norm(host_actual_array, axis=1)
    maximum_norm_error = float(
        max(
            np.max(np.abs(norms - 1.0)),
            np.max(np.abs(host_norms - 1.0)),
        )
    )
    max_abs_difference = float(
        max(
            np.max(np.abs(device_difference)),
            np.max(np.abs(host_difference)),
        )
    )
    l2_difference = float(
        max(
            np.linalg.norm(device_difference),
            np.linalg.norm(host_difference),
        )
    )
    device_result = isinstance(actual, cp.ndarray) and isinstance(
        host_actual, cp.ndarray
    )
    passed = (
        device_result
        and str(actual.dtype) == "complex128"
        and str(host_actual.dtype) == "complex128"
        and actual.shape == expected_array.shape
        and host_actual.shape == expected_array.shape
        and max_abs_difference <= MAX_ABS_TOLERANCE
        and maximum_norm_error <= NORM_TOLERANCE
    )
    return {
        "name": case["name"],
        "algorithm": case["algorithm"],
        "interaction_mode": case.get("interaction_mode"),
        "return_trajectory": case["return_traj"],
        "backend": "cupy",
        "dtype": str(actual.dtype),
        "shape": list(actual.shape),
        "device_result": device_result,
        "metrics": {
            "max_abs_difference": max_abs_difference,
            "l2_difference": l2_difference,
            "maximum_norm_error": maximum_norm_error,
        },
        "tolerances": {
            "max_abs_difference": MAX_ABS_TOLERANCE,
            "maximum_norm_error": NORM_TOLERANCE,
        },
        "timing_ms": {
            "cpu_reference_median": cpu_ms,
            "gpu_device_input_median": device_ms,
            "gpu_host_input_median": host_ms,
            "explicit_input_copy_median": input_copy_ms,
            "explicit_output_copy_median": output_copy_ms,
        },
        "transfer_bytes": {
            "host_to_device_inputs": int(
                sum(value.nbytes for value in host_values.values())
            ),
            "device_to_host_result": int(actual.nbytes),
        },
        "passed": passed,
    }


def _environment(cp: Any) -> dict[str, object]:
    properties = cp.cuda.runtime.getDeviceProperties(0)
    device_name = properties["name"]
    if isinstance(device_name, bytes):
        device_name = device_name.decode(errors="replace")
    major = int(properties["major"])
    minor = int(properties["minor"])
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "cupy": cp.__version__,
        "device_count": int(cp.cuda.runtime.getDeviceCount()),
        "device_id": 0,
        "device_name": str(device_name),
        "compute_capability": f"{major}.{minor}",
        "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "cuda_driver_version": int(cp.cuda.runtime.driverGetVersion()),
    }


def _finite_nonnegative(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def validate_report_schema(report: dict[str, object]) -> None:
    if report.get("schema_version") != 1:
        raise ValueError("CUDA evidence schema_version must be 1")
    if report.get("artifact") != "real-cuda-v0.3":
        raise ValueError("CUDA evidence artifact must be real-cuda-v0.3")

    source = report.get("source")
    if (
        not isinstance(source, dict)
        or not isinstance(source.get("commit"), str)
        or not source["commit"]
        or source.get("worktree_dirty") is not False
    ):
        raise ValueError("accepted CUDA evidence requires a clean source commit")

    environment = report.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("CUDA evidence environment must be a mapping")
    required_strings = (
        "python",
        "platform",
        "numpy",
        "cupy",
        "device_name",
        "compute_capability",
    )
    required_integers = (
        "device_count",
        "device_id",
        "cuda_runtime_version",
        "cuda_driver_version",
    )
    if not all(
        isinstance(environment.get(name), str) and bool(environment[name])
        for name in required_strings
    ):
        raise ValueError("CUDA evidence requires complete device/software identity")
    if not all(
        isinstance(environment.get(name), int)
        and not isinstance(environment[name], bool)
        for name in required_integers
    ):
        raise ValueError("CUDA evidence requires integer CUDA device/version fields")
    if environment["device_count"] < 1 or environment["device_id"] < 0:
        raise ValueError("CUDA evidence requires a real device")

    methodology = report.get("methodology")
    if (
        not isinstance(methodology, dict)
        or methodology.get("warmup_runs") != 1
        or not isinstance(methodology.get("timed_repeats"), int)
        or methodology["timed_repeats"] < 1
        or "synchronized" not in str(methodology.get("synchronization", "")).lower()
    ):
        raise ValueError("CUDA evidence methodology is incomplete")

    cases = report.get("cases")
    if not isinstance(cases, list):
        raise ValueError("CUDA evidence cases must be a list")
    names = [case.get("name") for case in cases if isinstance(case, dict)]
    if len(cases) != len(REQUIRED_CASES) or set(names) != set(REQUIRED_CASES):
        raise ValueError(
            "CUDA evidence must contain exactly these cases: "
            + ", ".join(REQUIRED_CASES)
        )
    if len(names) != len(set(names)):
        raise ValueError("CUDA evidence case names must be unique")

    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("CUDA evidence cases must be mappings")
        expected_contract = CASE_CONTRACTS[str(case["name"])]
        actual_contract = (
            case.get("algorithm"),
            case.get("interaction_mode"),
            case.get("return_trajectory"),
        )
        if actual_contract != expected_contract:
            raise ValueError("CUDA evidence case metadata is inconsistent")
        if case.get("backend") != "cupy" or case.get("device_result") is not True:
            raise ValueError(
                "every CUDA evidence result must remain a CuPy device array"
            )
        if case.get("dtype") != "complex128":
            raise ValueError("every CUDA evidence result must use complex128")
        shape = case.get("shape")
        if (
            not isinstance(shape, list)
            or not shape
            or any(not isinstance(value, int) or value < 1 for value in shape)
        ):
            raise ValueError("every CUDA evidence result requires a positive shape")

        metrics = case.get("metrics")
        tolerances = case.get("tolerances")
        timings = case.get("timing_ms")
        transfers = case.get("transfer_bytes")
        if not all(
            isinstance(value, dict)
            for value in (metrics, tolerances, timings, transfers)
        ):
            raise ValueError("CUDA evidence case sections must be mappings")
        assert isinstance(metrics, dict)
        assert isinstance(tolerances, dict)
        assert isinstance(timings, dict)
        assert isinstance(transfers, dict)
        if set(metrics) != {
            "max_abs_difference",
            "l2_difference",
            "maximum_norm_error",
        }:
            raise ValueError("CUDA evidence metric fields are incomplete")
        if set(tolerances) != {
            "max_abs_difference",
            "maximum_norm_error",
        }:
            raise ValueError("CUDA evidence tolerance fields are incomplete")
        if set(transfers) != {
            "host_to_device_inputs",
            "device_to_host_result",
        }:
            raise ValueError("CUDA evidence transfer fields are incomplete")
        if not all(_finite_nonnegative(value) for value in metrics.values()):
            raise ValueError(
                "CUDA evidence metrics must be finite non-negative numbers"
            )
        if not all(_finite_nonnegative(value) for value in tolerances.values()):
            raise ValueError(
                "CUDA evidence tolerances must be finite non-negative numbers"
            )
        if set(timings) != set(TIMING_KEYS) or not all(
            _finite_nonnegative(value) for value in timings.values()
        ):
            raise ValueError(
                "CUDA evidence timings must be finite non-negative numbers"
            )
        if not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in transfers.values()
        ):
            raise ValueError(
                "CUDA evidence transfer byte counts must be non-negative integers"
            )
        if not isinstance(case.get("passed"), bool):
            raise ValueError("CUDA evidence case acceptance must be boolean")
        expected_passed = (
            metrics["max_abs_difference"] <= tolerances["max_abs_difference"]
            and metrics["maximum_norm_error"] <= tolerances["maximum_norm_error"]
        )
        if case["passed"] is not expected_passed:
            raise ValueError("CUDA evidence case acceptance is inconsistent")

    all_cases_passed = all(bool(case["passed"]) for case in cases)
    acceptance = report.get("acceptance")
    if not isinstance(acceptance, dict):
        raise ValueError("CUDA evidence acceptance must be a mapping")
    if acceptance.get("all_cases_passed") is not all_cases_passed:
        raise ValueError("CUDA evidence aggregate acceptance is inconsistent")
    if acceptance.get("required_cases") != list(REQUIRED_CASES):
        raise ValueError("CUDA evidence required-case declaration is inconsistent")
    if acceptance.get("speed_is_not_an_acceptance_gate") is not True:
        raise ValueError("CUDA evidence must not use speed as an acceptance gate")
    expected_status = "pass" if all_cases_passed else "fail"
    if report.get("status") != expected_status:
        raise ValueError("CUDA evidence status is inconsistent")


def require_accepted_report(report: dict[str, object]) -> None:
    validate_report_schema(report)
    acceptance = report["acceptance"]
    assert isinstance(acceptance, dict)
    if acceptance["all_cases_passed"] is not True:
        raise RuntimeError("real-CUDA numerical evidence did not pass")


def build_report(
    *,
    dimension: int,
    steps: int,
    dt: float,
    repeats: int,
) -> dict[str, object]:
    cp = _require_real_cuda()
    problem = _problem(dimension, steps, dt)
    cases = [_case_record(cp, case, repeats) for case in _case_definitions(problem, dt)]
    all_cases_passed = all(bool(case["passed"]) for case in cases)
    return {
        "schema_version": 1,
        "artifact": "real-cuda-v0.3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if all_cases_passed else "fail",
        "source": _source_state(),
        "environment": _environment(cp),
        "methodology": {
            "synchronization": "CUDA stream synchronized around every timed sample",
            "warmup_runs": 1,
            "timed_repeats": repeats,
            "dimension": dimension,
            "propagation_steps": steps,
            "propagation_dt": dt,
            "gpu_device_input_median": "public low-level call with pre-existing device arrays; includes validation and algorithm setup",
            "gpu_host_input_median": "public low-level call from host arrays through device-native result",
            "explicit_transfer_medians": "isolated cp.asarray input copies and one explicit cp.asnumpy result copy",
            "performance_policy": "record only; no speed threshold or GPU speed claim",
        },
        "cases": cases,
        "acceptance": {
            "all_cases_passed": all_cases_passed,
            "required_cases": list(REQUIRED_CASES),
            "speed_is_not_an_acceptance_gate": True,
        },
    }


def _write_report(output: Path, report: dict[str, object]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/rve-real-cuda-evidence.json"),
    )
    args = parser.parse_args()
    if args.dimension < 2 or args.steps < 1 or args.dt <= 0.0 or args.repeats < 1:
        parser.error(
            "dimension >= 2, steps >= 1, dt > 0, and repeats >= 1 are required"
        )

    try:
        report = build_report(
            dimension=args.dimension,
            steps=args.steps,
            dt=args.dt,
            repeats=args.repeats,
        )
        validate_report_schema(report)
    except Exception as exc:
        diagnostic = {
            "schema_version": 1,
            "artifact": "real-cuda-v0.3",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "error",
            "source": _source_state(),
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
        _write_report(args.output, diagnostic)
        raise

    _write_report(args.output, report)
    print(json.dumps(report, indent=2))
    require_accepted_report(report)


if __name__ == "__main__":
    main()
