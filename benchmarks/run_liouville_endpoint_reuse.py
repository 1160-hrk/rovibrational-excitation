"""Benchmark exact RK4 endpoint-Hamiltonian reuse against the legacy loop."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

for variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np  # noqa: E402
from numba import njit  # noqa: E402

from rovibrational_excitation.dynamics.algorithms.rk4.liouville_numpy import (  # noqa: E402
    rk4_liouville_numpy_dense,
)

DEFAULT_OUTPUT = ROOT / "benchmarks" / "liouville-endpoint-reuse-v0.3.json"
DEFAULT_WORKLOADS = ((4, 1000), (16, 500), (32, 200), (64, 50))
RANDOM_SEED = 20260913


@njit(fastmath=True)  # type: ignore[untyped-decorator]
def legacy_liouville_kernel(
    H0: np.ndarray,
    mu_x: np.ndarray,
    mu_y: np.ndarray,
    Ex: np.ndarray,
    Ey: np.ndarray,
    rho0: np.ndarray,
    dt: float,
    steps: int,
    stride: int,
    record_traj: bool,
) -> np.ndarray:
    """Execute the pre-P5.1-c loop retained only as a benchmark reference."""
    dim = rho0.shape[0]
    n_out = steps // stride + 1 if record_traj else 1
    traj = np.empty((n_out, dim, dim), np.complex128)

    rho = rho0.copy()
    traj[0] = rho
    buf = np.empty_like(rho)
    out_idx = 1

    for s in range(steps):
        idx = 2 * s
        ex1 = Ex[idx]
        ex2 = Ex[idx + 1]
        ex4 = Ex[idx + 2]
        ey1 = Ey[idx]
        ey2 = Ey[idx + 1]
        ey4 = Ey[idx + 2]

        H1 = H0 - mu_x * ex1 - mu_y * ey1
        H2 = H0 - mu_x * ex2 - mu_y * ey2
        H4 = H0 - mu_x * ex4 - mu_y * ey4

        k1 = -1j * (H1 @ rho - rho @ H1)
        buf[:, :] = rho + 0.5 * dt * k1
        k2 = -1j * (H2 @ buf - buf @ H2)
        buf[:, :] = rho + 0.5 * dt * k2
        k3 = -1j * (H2 @ buf - buf @ H2)
        buf[:, :] = rho + dt * k3
        k4 = -1j * (H4 @ buf - buf @ H4)

        rho += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        if record_traj and (s + 1) % stride == 0:
            traj[out_idx] = rho
            out_idx += 1

    if not record_traj:
        traj[0] = rho
    return traj


def build_problem(
    dimension: int,
    steps: int,
    *,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, ...]:
    """Build one deterministic Hermitian two-component Liouville problem."""
    if dimension < 2:
        raise ValueError("dimension must be at least 2")
    if steps < 1:
        raise ValueError("steps must be positive")

    random = np.random.default_rng(seed + dimension)
    operators = []
    for _ in range(3):
        raw = random.normal(size=(dimension, dimension)) + 1j * random.normal(
            size=(dimension, dimension)
        )
        operators.append(
            np.ascontiguousarray((raw + raw.conj().T) / 20.0, dtype=np.complex128)
        )
    H0, mu_x, mu_y = operators

    state_factor = random.normal(size=(dimension, dimension)) + 1j * random.normal(
        size=(dimension, dimension)
    )
    rho0 = state_factor @ state_factor.conj().T
    rho0 = np.ascontiguousarray(rho0 / np.trace(rho0), dtype=np.complex128)
    Ex = np.ascontiguousarray(random.normal(size=2 * steps + 1), dtype=np.float64)
    Ey = np.ascontiguousarray(random.normal(size=2 * steps + 1), dtype=np.float64)
    return H0, mu_x, mu_y, Ex, Ey, rho0


def benchmark_workload(
    dimension: int,
    steps: int,
    repeats: int,
) -> dict[str, object]:
    """Measure one warmed final-state workload and verify exact parity."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    problem = build_problem(dimension, steps)
    arguments = (*problem, 0.001, steps, 1, False)

    legacy_liouville_kernel(*arguments)
    rk4_liouville_numpy_dense(*arguments)

    legacy_times_ns: list[int] = []
    endpoint_reuse_times_ns: list[int] = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for repeat in range(repeats):
            if repeat % 2 == 0:
                start = time.perf_counter_ns()
                legacy_result = legacy_liouville_kernel(*arguments)
                legacy_times_ns.append(time.perf_counter_ns() - start)
                start = time.perf_counter_ns()
                endpoint_result = rk4_liouville_numpy_dense(*arguments)
                endpoint_reuse_times_ns.append(time.perf_counter_ns() - start)
            else:
                start = time.perf_counter_ns()
                endpoint_result = rk4_liouville_numpy_dense(*arguments)
                endpoint_reuse_times_ns.append(time.perf_counter_ns() - start)
                start = time.perf_counter_ns()
                legacy_result = legacy_liouville_kernel(*arguments)
                legacy_times_ns.append(time.perf_counter_ns() - start)
    finally:
        if gc_was_enabled:
            gc.enable()

    legacy_median_ns = int(statistics.median(legacy_times_ns))
    endpoint_median_ns = int(statistics.median(endpoint_reuse_times_ns))
    max_abs_difference = float(np.max(np.abs(legacy_result - endpoint_result)))
    exact_equal = bool(np.array_equal(legacy_result, endpoint_result))

    avoided_hamiltonian_builds = steps - 1
    avoided_hamiltonian_bytes = (
        avoided_hamiltonian_builds
        * dimension
        * dimension
        * np.dtype(np.complex128).itemsize
    )
    return {
        "dimension": dimension,
        "propagation_steps": steps,
        "field_points": 2 * steps + 1,
        "repeats": repeats,
        "legacy_median_ns": legacy_median_ns,
        "endpoint_reuse_median_ns": endpoint_median_ns,
        "speedup": legacy_median_ns / endpoint_median_ns,
        "exact_equal": exact_equal,
        "max_abs_difference": max_abs_difference,
        "legacy_hamiltonian_builds": 3 * steps,
        "endpoint_reuse_hamiltonian_builds": 2 * steps + 1,
        "avoided_hamiltonian_builds": avoided_hamiltonian_builds,
        "avoided_hamiltonian_allocation_traffic_bytes": avoided_hamiltonian_bytes,
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_value(*arguments: str) -> str | None:
    try:
        return subprocess.run(
            ("git", *arguments),
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_report(repeats: int = 11) -> dict[str, object]:
    """Run the fixed workload matrix and construct a serializable report."""
    results = [
        benchmark_workload(dimension, steps, repeats)
        for dimension, steps in DEFAULT_WORKLOADS
    ]
    return {
        "schema_version": 1,
        "artifact": "liouville-endpoint-reuse-v0.3",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "random_seed": RANDOM_SEED,
            "dt": 0.001,
            "record_trajectory": False,
            "timer": "time.perf_counter_ns",
            "statistic": "median",
            "alternating_measurement_order": True,
            "allocation_metric": (
                "analytical complex128 bytes for the eliminated source-level "
                "endpoint Hamiltonian arrays; not process RSS"
            ),
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "numpy": _package_version("numpy"),
            "numba": _package_version("numba"),
            "source_commit": _git_value("rev-parse", "HEAD"),
            "source_status": _git_value("status", "--short"),
        },
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args(argv)

    report = build_report(repeats=arguments.repeats)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
