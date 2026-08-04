"""Benchmark the explicit split-operator polarization contracts.

Wall-clock values are diagnostic only. Run this script on the same machine and
dependency stack when comparing revisions.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

# Avoid comparing different BLAS thread counts on small/medium matrices.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import numba
import numpy as np
import scipy

from rovibrational_excitation.core.basis import LinMolBasis
from rovibrational_excitation.core.propagation.algorithms.rk4.schrodinger import (
    rk4_schrodinger,
)
from rovibrational_excitation.core.propagation.algorithms.split_operator.schrodinger import (
    splitop_schrodinger,
)
from rovibrational_excitation.dipole.linmol import LinMolDipoleMatrix


def _source_state() -> dict[str, object]:
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], check=False, capture_output=True, text=True
        )
        return result.stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "worktree_dirty": bool(git("status", "--porcelain")),
    }


def _problem(j_max: int, steps: int, dt: float):
    basis = LinMolBasis(
        V_max=1,
        J_max=j_max,
        use_M=True,
        omega=1.0,
        B=0.01,
        input_units="rad/fs",
        output_units="rad/fs",
    )
    h0 = np.asarray(basis.generate_H0().matrix)
    dipole = LinMolDipoleMatrix(basis, mu0=1.0)
    mu_x = np.asarray(dipole.mu_x)
    mu_y = np.asarray(dipole.mu_y)
    initial = np.zeros(basis.size(), dtype=np.complex128)
    initial[basis.get_index((0, 0, 0))] = 1.0

    time_grid = np.arange(2 * steps + 1, dtype=np.float64) * (dt / 2.0)
    center = steps * dt / 2.0
    width = max(steps * dt / 5.0, dt)
    envelope = 0.08 * np.exp(-0.5 * ((time_grid - center) / width) ** 2)
    angle = 1.02 * time_grid
    field_x = envelope * np.cos(angle) / np.sqrt(2.0)
    field_y = -envelope * np.sin(angle) / np.sqrt(2.0)
    projected_scalar = envelope * np.cos(angle)
    polarization = np.array([1.0, 1.0j]) / np.sqrt(2.0)
    return (
        basis,
        h0,
        mu_x,
        mu_y,
        initial,
        field_x,
        field_y,
        projected_scalar,
        polarization,
    )


def _measure(callable_, repeats: int) -> tuple[float, np.ndarray]:
    callable_()  # compile and warm caches outside the timed region
    elapsed_ms = []
    final = None
    for _ in range(repeats):
        start = time.perf_counter_ns()
        final = callable_()
        elapsed_ms.append((time.perf_counter_ns() - start) / 1.0e6)
    assert final is not None
    return float(median(elapsed_ms)), final[0]


def _workload(j_max: int, steps: int, dt: float, repeats: int) -> dict[str, object]:
    (
        basis,
        h0,
        mu_x,
        mu_y,
        initial,
        field_x,
        field_y,
        projected_scalar,
        polarization,
    ) = _problem(j_max, steps, dt)

    def run_rk4():
        return rk4_schrodinger(
            h0,
            mu_x,
            mu_y,
            field_x,
            field_y,
            initial,
            dt,
            return_traj=False,
        )

    def run_cartesian():
        return splitop_schrodinger(
            h0,
            mu_x,
            mu_y,
            field_x,
            field_y,
            initial,
            dt,
            return_traj=False,
            magnetic_quantum_numbers=basis.M_array,
        )

    def run_projected():
        return splitop_schrodinger(
            h0,
            mu_x,
            mu_y,
            field_x,
            field_y,
            initial,
            dt,
            return_traj=False,
            interaction_mode="helicity_projected",
            polarization=polarization,
            scalar_field=projected_scalar,
        )

    rk4_ms, rk4_final = _measure(run_rk4, repeats)
    cartesian_ms, cartesian_final = _measure(run_cartesian, repeats)
    projected_ms, projected_final = _measure(run_projected, repeats)
    return {
        "j_max": j_max,
        "dimension": basis.size(),
        "steps": steps,
        "dt": dt,
        "median_ms": {
            "rk4_dense": rk4_ms,
            "split_cartesian": cartesian_ms,
            "split_helicity_projected": projected_ms,
        },
        "speedup_vs_rk4": {
            "split_cartesian": rk4_ms / cartesian_ms,
            "split_helicity_projected": rk4_ms / projected_ms,
        },
        "final_norm_error": {
            "rk4_dense": abs(float(np.linalg.norm(rk4_final)) - 1.0),
            "split_cartesian": abs(float(np.linalg.norm(cartesian_final)) - 1.0),
            "split_helicity_projected": abs(
                float(np.linalg.norm(projected_final)) - 1.0
            ),
        },
        "same_grid_cartesian_vs_rk4_l2": float(
            np.linalg.norm(cartesian_final - rk4_final)
        ),
        "projected_vs_cartesian_model_difference_l2": float(
            np.linalg.norm(projected_final - cartesian_final)
        ),
    }


def _convergence_check() -> dict[str, object]:
    errors = []
    for dt in (0.02, 0.01):
        steps = round(0.4 / dt)
        basis, h0, mu_x, mu_y, initial, ex, ey, _scalar, _pol = _problem(1, steps, dt)
        rk4_final = rk4_schrodinger(
            h0, mu_x, mu_y, ex, ey, initial, dt, return_traj=False
        )[0]
        split_final = splitop_schrodinger(
            h0,
            mu_x,
            mu_y,
            ex,
            ey,
            initial,
            dt,
            return_traj=False,
            magnetic_quantum_numbers=basis.M_array,
        )[0]
        errors.append(float(np.linalg.norm(split_final - rk4_final)))
    return {
        "coarse_dt": 0.02,
        "fine_dt": 0.01,
        "coarse_l2": errors[0],
        "fine_l2": errors[1],
        "coarse_to_fine_error_ratio": errors[0] / errors[1],
        "interpretation": "approximately 4 indicates second-order convergence",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--j-max", nargs="+", type=int, default=[3, 5])
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/split-polarization-v0.3.json"),
    )
    args = parser.parse_args()
    if args.steps < 1 or args.repeats < 1 or args.dt <= 0.0:
        parser.error("steps and repeats must be positive, and dt must be > 0")

    report = {
        "artifact": "split-polarization-v0.3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "numba": numba.__version__,
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "gpu": "not run; CUDA hardware unavailable in this environment",
        },
        "source": _source_state(),
        "methodology": {
            "timing": "median perf_counter_ns after one untimed warmup",
            "timed_scope": "public final-state propagation, including eigendecomposition",
            "comparison": "same field grid; projected difference is a model difference",
        },
        "workloads": [
            _workload(j_max, args.steps, args.dt, args.repeats) for j_max in args.j_max
        ],
        "cartesian_convergence": _convergence_check(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
