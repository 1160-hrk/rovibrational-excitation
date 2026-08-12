"""Record the deterministic four-level Krotov V=0 -> V=3 reference.

The saved NPZ contains the optimized field and an independently propagated
trajectory. The adjacent JSON contains scalar checks and source provenance.
This is a reproducible end-to-end regression reference, not an independent
proof that the Krotov objective or update equation is physically complete.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
# ruff: noqa: E402
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from rovibrational_excitation.core.propagation import SchrodingerPropagator
from rovibrational_excitation.simulation.optimize_runner import run_from_config

DEFAULT_CONFIG = ROOT / "configs" / "reference_krotov_viblad_v3.yaml"
DEFAULT_NPZ = ROOT / "benchmarks" / "krotov-v0-v3-v0.3.npz"
DEFAULT_JSON = ROOT / "benchmarks" / "krotov-v0-v3-v0.3.json"


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], check=False, capture_output=True, text=True, cwd=ROOT
    )
    return result.stdout.strip()


def record_reference(
    *, config_path: Path, npz_path: Path, json_path: Path
) -> dict[str, object]:
    source = {
        "commit": _git("rev-parse", "HEAD"),
        "worktree_dirty": bool(_git("status", "--porcelain")),
    }
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    run = run_from_config(config=config, out_dir=npz_path.parent, do_plot=False)
    result = run["result"]
    basis = run["basis"]

    initial = np.zeros(basis.size(), dtype=np.complex128)
    initial[basis.get_index(tuple(config["states"]["initial"]))] = 1.0
    check_time, check_trajectory = SchrodingerPropagator(
        backend="numpy", validate_units=True, renorm=True
    ).propagate(
        hamiltonian=run["H0"],
        efield=result["efield"],
        dipole_matrix=run["dipole"],
        initial_state=initial,
        axes=config["algorithms"]["krotov"]["control_axes"],
        return_traj=True,
        return_time_psi=True,
        sample_stride=1,
        algorithm="rk4",
        sparse=False,
    )

    optimized_trajectory = np.asarray(result["psi_traj"])
    check_trajectory = np.asarray(check_trajectory)
    populations = np.abs(check_trajectory) ** 2
    field = np.asarray(result["field_data"])
    field_times = np.asarray(result["tlist"])
    target_index = basis.get_index(tuple(config["states"]["target"]))

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        time_fs=np.asarray(check_time),
        psi_traj=check_trajectory,
        field_time_fs=field_times,
        field_V_per_m=field,
        final_populations=populations[-1],
    )

    summary: dict[str, object] = {
        "artifact": "krotov-v0-v3-v0.3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "configuration": _display_path(config_path),
        "dimension": basis.size(),
        "target_index": target_index,
        "fidelity": float(populations[-1, target_index]),
        "reported_fidelity": float(result["metrics"]["fidelity"]),
        "final_populations": populations[-1].tolist(),
        "maximum_non_target_population": float(
            np.max(np.delete(populations[-1], target_index))
        ),
        "norm_min": float(np.min(np.sum(populations, axis=1))),
        "norm_max": float(np.max(np.sum(populations, axis=1))),
        "independent_trajectory_max_abs_difference": float(
            np.max(np.abs(optimized_trajectory - check_trajectory))
        ),
        "field_peak_V_per_m": np.max(np.abs(field), axis=0).tolist(),
        "field_rms_V_per_m": np.sqrt(np.mean(field**2, axis=0)).tolist(),
        "trajectory_shape": list(check_trajectory.shape),
        "field_shape": list(field.shape),
        "npz": _display_path(npz_path),
        "interpretation": (
            "Deterministic end-to-end regression reference; not an independent "
            "validation of the optimization objective or update equation."
        ),
    }
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    args = parser.parse_args()
    summary = record_reference(
        config_path=args.config.resolve(),
        npz_path=args.npz.resolve(),
        json_path=args.json.resolve(),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
