"""Contract checks for the non-blocking benchmark recorder."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.run_baseline import (  # noqa: E402
    MODEL_NAMES,
    build_workloads,
    serialize_complex_array,
    trajectory_memory_estimate_bytes,
)

pytestmark = pytest.mark.performance


def test_workload_matrix_covers_required_models_and_storage_paths():
    nine_points = 9
    workloads = build_workloads(field_points=nine_points)
    by_name = {workload.name: workload for workload in workloads}

    assert tuple(MODEL_NAMES) == ("two_level", "vib_ladder", "linear_molecule")
    for model in MODEL_NAMES:
        for storage in ("dense", "sparse"):
            name = f"{model}_schrodinger_numpy_{storage}"
            assert name in by_name
            assert by_name[name].field_points == nine_points
            assert by_name[name].propagation_steps == 4

    assert by_name["two_level_schrodinger_numpy_dense"].dimension == 2
    assert by_name["vib_ladder_schrodinger_numpy_dense"].dimension == 16
    assert by_name["linear_molecule_schrodinger_numpy_dense"].dimension == 18

    liouville = by_name["two_level_liouville_numpy_dense"]
    assert liouville.state_kind == "density_matrix"
    assert liouville.storage == "dense"


@pytest.mark.parametrize(
    ("state_kind", "dimension", "expected"),
    [
        ("wavefunction", 18, 11 * 18 * 16),
        ("density_matrix", 2, 11 * 2 * 2 * 16),
    ],
)
def test_trajectory_memory_estimate_is_complex128_allocation(
    state_kind: str,
    dimension: int,
    expected: int,
):
    assert (
        trajectory_memory_estimate_bytes(
            saved_states=11,
            dimension=dimension,
            state_kind=state_kind,
        )
        == expected
    )


def test_complex_reference_serialization_preserves_shape_and_components():
    value = np.array([[1.0 + 2.0j, -3.5j]], dtype=np.complex128)

    serialized = serialize_complex_array(value)

    assert serialized == {
        "shape": [1, 2],
        "real": [[1.0, -0.0]],
        "imag": [[2.0, -3.5]],
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"saved_states": 0, "dimension": 2, "state_kind": "wavefunction"},
        {"saved_states": 1, "dimension": 0, "state_kind": "wavefunction"},
        {"saved_states": 1, "dimension": 2, "state_kind": "unknown"},
    ],
)
def test_trajectory_memory_estimate_rejects_invalid_contract(kwargs):
    with pytest.raises(ValueError):
        trajectory_memory_estimate_bytes(**kwargs)
