"""Record the non-blocking v0.2.10 propagation benchmark baseline."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import importlib.util
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np  # noqa: E402

from rovibrational_excitation.core.basis import (  # noqa: E402
    LinMolBasis,
    TwoLevelBasis,
    VibLadderBasis,
)
from rovibrational_excitation.core.electric_field import (  # noqa: E402
    ElectricField,
    gaussian_fwhm,
)
from rovibrational_excitation.core.propagation import (  # noqa: E402
    LiouvillePropagator,
    SchrodingerPropagator,
)
from rovibrational_excitation.dipole import (  # noqa: E402
    LinMolDipoleMatrix,
    TwoLevelDipoleMatrix,
    VibLadderDipoleMatrix,
)

DEFAULT_OUTPUT = ROOT / "benchmarks" / "baseline-v0.2.10.json"
MODEL_NAMES = ("two_level", "vib_ladder", "linear_molecule")


@dataclass(frozen=True)
class Workload:
    """Prepared propagation case whose setup is excluded from timing."""

    name: str
    model: str
    equation: str
    storage: str
    dimension: int
    field_points: int
    propagation_steps: int
    state_kind: str
    run: Callable[[], np.ndarray]


def _field(
    field_points: int,
    *,
    polarization: np.ndarray,
    amplitude: float = 8.0e8,
) -> ElectricField:
    if field_points < 3 or field_points % 2 == 0:
        raise ValueError("field_points must be an odd integer >= 3")
    time_fs = np.linspace(0.0, 4.0, field_points)
    field = ElectricField(time_fs)
    field.add_dispersed_Efield(
        gaussian_fwhm,
        duration=1.5,
        t_center=2.0,
        carrier_freq=0.0,
        amplitude=amplitude,
        polarization=polarization,
        const_polarisation=True,
    )
    return field


def _pure_workload(
    *,
    name: str,
    model: str,
    storage: str,
    hamiltonian,
    field: ElectricField,
    dipole,
    initial: np.ndarray,
    coupling_mode: str,
    coupling_axis: str | None = None,
    axes: str = "xy",
) -> Workload:
    sparse = storage == "sparse"
    solver = SchrodingerPropagator(
        backend="numpy",
        validate_units=False,
        sparse=sparse,
    )

    def run() -> np.ndarray:
        return solver.propagate(
            hamiltonian,
            field,
            dipole,
            initial,
            axes=axes,
            coupling_mode=coupling_mode,
            coupling_axis=coupling_axis,
            return_traj=True,
            sample_stride=1,
            sparse=sparse,
        )

    return Workload(
        name=name,
        model=model,
        equation="schrodinger",
        storage=storage,
        dimension=initial.size,
        field_points=field.tlist.size,
        propagation_steps=(field.tlist.size - 1) // 2,
        state_kind="wavefunction",
        run=run,
    )


def build_workloads(field_points: int = 4001) -> list[Workload]:
    """Build deterministic model workloads without running propagation."""
    scalar_field = _field(
        field_points,
        polarization=np.array([0.0, 1.0], dtype=np.float64),
    )
    cartesian_field = _field(
        field_points,
        polarization=np.array([1.0, 0.0], dtype=np.float64),
    )

    two_basis = TwoLevelBasis(
        energy_gap=0.37,
        input_units="rad/fs",
        output_units="rad/fs",
    )
    two_h0 = two_basis.generate_H0()
    two_initial = np.array(
        [np.sqrt(0.4), np.sqrt(0.6) * np.exp(0.2j)],
        dtype=np.complex128,
    )
    two_dipole = TwoLevelDipoleMatrix(two_basis, mu0=1.0e-30)

    vib_basis = VibLadderBasis(
        V_max=15,
        omega=0.37,
        delta_omega=0.01,
        input_units="rad/fs",
        output_units="rad/fs",
    )
    vib_h0 = vib_basis.generate_H0()
    vib_initial = np.zeros(vib_basis.size(), dtype=np.complex128)
    vib_initial[0] = 1.0
    vib_dipole = VibLadderDipoleMatrix(
        vib_basis,
        mu0=1.0e-30,
        potential_type="harmonic",
        units="C*m",
        units_input="C*m",
    )

    lin_basis = LinMolBasis(
        V_max=1,
        J_max=2,
        use_M=True,
        omega=0.37,
        delta_omega=0.01,
        B=0.003,
        alpha=0.0001,
        input_units="rad/fs",
        output_units="rad/fs",
    )
    lin_h0 = lin_basis.generate_H0()
    lin_initial = np.zeros(lin_basis.size(), dtype=np.complex128)
    lin_initial[lin_basis.get_index((0, 0, 0))] = 1.0

    workloads: list[Workload] = []
    for storage in ("dense", "sparse"):
        workloads.append(
            _pure_workload(
                name=f"two_level_schrodinger_numpy_{storage}",
                model="two_level",
                storage=storage,
                hamiltonian=two_h0,
                field=scalar_field,
                dipole=two_dipole,
                initial=two_initial,
                coupling_mode="scalar",
                coupling_axis="z",
            )
        )
        workloads.append(
            _pure_workload(
                name=f"vib_ladder_schrodinger_numpy_{storage}",
                model="vib_ladder",
                storage=storage,
                hamiltonian=vib_h0,
                field=scalar_field,
                dipole=vib_dipole,
                initial=vib_initial,
                coupling_mode="scalar",
                coupling_axis="z",
            )
        )
        lin_dipole = LinMolDipoleMatrix(
            lin_basis,
            mu0=1.0e-30,
            dense=storage == "dense",
        )
        workloads.append(
            _pure_workload(
                name=f"linear_molecule_schrodinger_numpy_{storage}",
                model="linear_molecule",
                storage=storage,
                hamiltonian=lin_h0,
                field=cartesian_field,
                dipole=lin_dipole,
                initial=lin_initial,
                coupling_mode="cartesian",
                axes="xz",
            )
        )

    rho0 = np.outer(two_initial, two_initial.conj())
    liouville = LiouvillePropagator(backend="numpy", validate_units=False)

    def run_liouville() -> np.ndarray:
        return liouville.propagate(
            two_h0,
            scalar_field,
            two_dipole,
            rho0,
            coupling_mode="scalar",
            coupling_axis="z",
            return_traj=True,
            sample_stride=1,
        )

    workloads.append(
        Workload(
            name="two_level_liouville_numpy_dense",
            model="two_level",
            equation="liouville",
            storage="dense",
            dimension=two_basis.size(),
            field_points=field_points,
            propagation_steps=(field_points - 1) // 2,
            state_kind="density_matrix",
            run=run_liouville,
        )
    )
    return workloads


def trajectory_memory_estimate_bytes(
    *,
    saved_states: int,
    dimension: int,
    state_kind: str,
) -> int:
    """Return the allocated complex128 trajectory size, excluding temporaries."""
    if saved_states < 1 or dimension < 1:
        raise ValueError("saved_states and dimension must be positive")
    if state_kind == "wavefunction":
        elements_per_state = dimension
    elif state_kind == "density_matrix":
        elements_per_state = dimension * dimension
    else:
        raise ValueError("state_kind must be 'wavefunction' or 'density_matrix'")
    return saved_states * elements_per_state * np.dtype(np.complex128).itemsize


def serialize_complex_array(value: np.ndarray) -> dict:
    """Serialize a small complex reference output without losing precision."""
    array = np.asarray(value, dtype=np.complex128)
    return {
        "shape": list(array.shape),
        "real": array.real.tolist(),
        "imag": array.imag.tolist(),
    }


def _benchmark_workload(
    workload: Workload,
    *,
    repeats: int,
    warmup_runs: int,
) -> tuple[dict, np.ndarray]:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if warmup_runs < 1:
        raise ValueError("warmup_runs must be at least one")

    for _ in range(warmup_runs):
        workload.run()

    samples: list[float] = []
    trajectory = None
    for _ in range(repeats):
        gc.collect()
        start_ns = time.perf_counter_ns()
        trajectory = np.asarray(workload.run())
        samples.append((time.perf_counter_ns() - start_ns) / 1.0e9)

    assert trajectory is not None
    final = np.asarray(trajectory[-1], dtype=np.complex128)
    saved_states = workload.propagation_steps + 1
    estimated_bytes = trajectory_memory_estimate_bytes(
        saved_states=saved_states,
        dimension=workload.dimension,
        state_kind=workload.state_kind,
    )

    if workload.state_kind == "wavefunction":
        final_norm = float(np.linalg.norm(final))
        final_norm_error = abs(final_norm - 1.0)
        final_trace_error = None
        hermiticity_error = None
    else:
        final_norm = None
        final_norm_error = None
        final_trace_error = float(abs(np.trace(final) - 1.0))
        hermiticity_error = float(np.linalg.norm(final - final.conj().T))

    record = {
        "name": workload.name,
        "model": workload.model,
        "equation": workload.equation,
        "algorithm": "rk4",
        "backend": "numpy",
        "storage": workload.storage,
        "dimension": workload.dimension,
        "field_points": workload.field_points,
        "propagation_steps": workload.propagation_steps,
        "saved_states": int(trajectory.shape[0]),
        "samples_seconds": samples,
        "median_seconds": float(statistics.median(samples)),
        "minimum_seconds": min(samples),
        "maximum_seconds": max(samples),
        "final_norm": final_norm,
        "final_norm_error": final_norm_error,
        "final_trace_error": final_trace_error,
        "final_hermiticity_error": hermiticity_error,
        "peak_trajectory_memory_estimate_bytes": estimated_bytes,
        "trajectory_array_bytes": int(trajectory.nbytes),
        "final_state": serialize_complex_array(final),
    }
    if record["saved_states"] != saved_states:
        raise RuntimeError(
            f"{workload.name}: expected {saved_states} saved states, "
            f"received {record['saved_states']}"
        )
    if record["trajectory_array_bytes"] != estimated_bytes:
        raise RuntimeError(
            f"{workload.name}: trajectory allocation does not match the estimate"
        )
    return record, final


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_metadata() -> dict:
    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        return {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "worktree_dirty": bool(git("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "worktree_dirty": None}


def _cuda_metadata() -> dict:
    if importlib.util.find_spec("cupy") is None:
        return {
            "status": "not_run",
            "reason": "CuPy is not installed; no GPU result is claimed.",
        }
    try:
        import cupy as cp

        device_count = int(cp.cuda.runtime.getDeviceCount())
        if device_count < 1:
            return {
                "status": "not_run",
                "reason": "CuPy is installed but no CUDA device is available.",
            }
        properties = cp.cuda.runtime.getDeviceProperties(0)
        device_name = properties["name"]
        if isinstance(device_name, bytes):
            device_name = device_name.decode()
        return {
            "status": "available_not_measured",
            "device_count": device_count,
            "device_name": device_name,
            "cupy": cp.__version__,
            "reason": (
                "This artifact is the NumPy baseline. GPU timing must be recorded "
                "in a dedicated real-CUDA run before it is claimed."
            ),
        }
    except Exception as exc:
        return {
            "status": "not_run",
            "reason": f"CUDA runtime check failed: {type(exc).__name__}: {exc}",
        }


def _environment_metadata() -> dict:
    dependencies = {
        name: _distribution_version(name)
        for name in (
            "numpy",
            "scipy",
            "numba",
            "pandas",
            "tqdm",
            "PyYAML",
            "rovibrational-excitation",
        )
    }
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "logical_cpu_count": os.cpu_count(),
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "dependencies": dependencies,
        "cuda": _cuda_metadata(),
    }


def _dense_sparse_comparisons(
    records: list[dict],
    finals: dict[str, np.ndarray],
) -> list[dict]:
    by_name = {record["name"]: record for record in records}
    comparisons = []
    for model in MODEL_NAMES:
        dense_name = f"{model}_schrodinger_numpy_dense"
        sparse_name = f"{model}_schrodinger_numpy_sparse"
        dense = by_name[dense_name]
        sparse = by_name[sparse_name]
        comparisons.append(
            {
                "model": model,
                "dense_sparse_final_l2_difference": float(
                    np.linalg.norm(finals[dense_name] - finals[sparse_name])
                ),
                "sparse_to_dense_median_time_ratio": (
                    sparse["median_seconds"] / dense["median_seconds"]
                ),
            }
        )
    return comparisons


def validate_report(report: dict) -> None:
    """Reject incomplete artifacts before they are written."""
    if report.get("schema_version") != 1:
        raise ValueError("benchmark schema_version must be 1")
    records = report.get("workloads", [])
    names = {record.get("name") for record in records}
    for model in MODEL_NAMES:
        for storage in ("dense", "sparse"):
            required = f"{model}_schrodinger_numpy_{storage}"
            if required not in names:
                raise ValueError(f"missing required workload: {required}")
    liouville = next(
        (
            record
            for record in records
            if record.get("name") == "two_level_liouville_numpy_dense"
        ),
        None,
    )
    if liouville is None or liouville.get("final_trace_error") is None:
        raise ValueError("missing TwoLevel Liouville trace measurement")
    if not report["methodology"].get("jit_warmup_excluded"):
        raise ValueError("JIT warmup must be excluded")
    if report["methodology"].get("summary_statistic") != "median":
        raise ValueError("timing summary must use the median")


def run_baseline(
    *,
    repeats: int = 7,
    warmup_runs: int = 1,
    field_points: int = 4001,
) -> dict:
    """Measure all CPU workloads and return a validated JSON-ready report."""
    records: list[dict] = []
    finals: dict[str, np.ndarray] = {}
    for workload in build_workloads(field_points):
        record, final = _benchmark_workload(
            workload,
            repeats=repeats,
            warmup_runs=warmup_runs,
        )
        records.append(record)
        finals[workload.name] = final

    report = {
        "schema_version": 1,
        "artifact": "baseline-v0.2.10",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": _git_metadata(),
        "environment": _environment_metadata(),
        "methodology": {
            "timer": "time.perf_counter_ns",
            "warmup_runs_per_workload": warmup_runs,
            "jit_warmup_excluded": True,
            "repeats": repeats,
            "summary_statistic": "median",
            "timed_scope": (
                "Propagator.propagate only; model, dipole, field, and initial-state "
                "construction are excluded."
            ),
            "absolute_runtime_gate": False,
            "trajectory_memory_definition": (
                "Returned complex128 trajectory allocation only; excludes operators, "
                "JIT/runtime allocations, temporaries, and Python overhead."
            ),
        },
        "workloads": records,
        "comparisons": _dense_sparse_comparisons(records, finals),
    }
    validate_report(report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--field-points", type=int, default=4001)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run_baseline(
        repeats=args.repeats,
        warmup_runs=args.warmup_runs,
        field_points=args.field_points,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
