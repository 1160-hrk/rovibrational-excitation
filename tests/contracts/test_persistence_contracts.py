"""File-format and overwrite contracts for persistence helpers."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rovibrational_excitation.core.time import TimeGrid
from rovibrational_excitation.fields import ScalarField
from rovibrational_excitation.io import (
    CheckpointManager,
    deserialize_polarization,
    json_safe,
    storage,
)
from rovibrational_excitation.io.result_schema import ResultFormatError
from rovibrational_excitation.simulation.result_persistence import (
    persist_wavefunction_result,
)


def test_json_safe_preserves_the_current_recursive_representation():
    value = {
        "complex": 3.0 - 4.0j,
        "scalar": np.float64(1.25),
        "array": np.array([1 + 2j, 3 - 1j]),
        "tuple": (np.int64(2),),
    }

    assert json_safe(value) == {
        "complex": {"__complex__": True, "r": 3.0, "i": -4.0},
        "scalar": 1.25,
        "array": [
            {"__complex__": True, "r": 1.0, "i": 2.0},
            {"__complex__": True, "r": 3.0, "i": -1.0},
        ],
        "tuple": [2],
    }


def test_deserialize_polarization_preserves_scalar_and_vector_forms():
    np.testing.assert_array_equal(
        deserialize_polarization({"real": 0.5, "imag": -0.25}),
        np.array([0.5 - 0.25j, 0.0j]),
    )
    np.testing.assert_array_equal(
        deserialize_polarization([{"r": 0.0, "i": 1.0}, 2.0]),
        np.array([1.0j, 2.0 + 0.0j]),
    )


def test_checkpoint_schema_filenames_deduplication_and_overwrite(tmp_path):
    manager = CheckpointManager(tmp_path)
    completed = {"amplitude": 1.0, "outdir": "first", "save": True}
    duplicate = {**completed, "outdir": "second"}
    failed = {"amplitude": 2.0, "error": "failure"}

    manager.save_checkpoint([completed, duplicate], [failed], 3, 12.5)

    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text())
    failed_cases = json.loads((tmp_path / "failed_cases.json").read_text())
    assert set(checkpoint) == {
        "timestamp",
        "start_time",
        "total_cases",
        "completed_cases",
        "failed_cases",
        "completed_case_hashes",
        "failed_case_data",
    }
    assert datetime.fromisoformat(checkpoint["timestamp"])
    assert checkpoint["start_time"] == 12.5
    assert checkpoint["total_cases"] == 3
    assert checkpoint["completed_cases"] == 1
    assert checkpoint["failed_cases"] == 1
    assert checkpoint["failed_case_data"] == [failed]
    assert failed_cases == [failed]

    replacement = {"amplitude": 3.0}
    manager.save_checkpoint([replacement], [], 1, 20.0)

    overwritten = manager.load_checkpoint()
    assert overwritten is not None
    assert overwritten["start_time"] == 20.0
    assert overwritten["total_cases"] == 1
    assert overwritten["completed_cases"] == 1
    assert overwritten["failed_cases"] == 0
    assert overwritten["failed_case_data"] == []
    assert json.loads((tmp_path / "failed_cases.json").read_text()) == []


def test_versioned_result_keeps_exact_numeric_arrays_and_json_regime_info(tmp_path):
    grid = TimeGrid.from_bounds(-1.0, 1.0, 0.5)
    field_samples = np.array([0.0, 2.0, -3.0, 2.0, 0.0])
    field = ScalarField(grid, field_samples)
    times_fs = np.array([-1.0, 0.0, 1.0])
    state = np.array(
        [[1.0, 0.0], [0.5 + 0.5j, 0.5 - 0.5j], [0.0, 1.0]],
        dtype=np.complex128,
    )
    population = np.abs(state) ** 2
    params = {"basis_type": "twolevel", "amplitude": 2.0}
    regime_info = {"energy_scale_eV": 0.25}

    persist_wavefunction_result(
        outdir=tmp_path,
        params=params,
        field_times_fs=grid.field_times_fs,
        sampled_field=field,
        times_fs=times_fs,
        state=state,
        population=population,
        regime_info=regime_info,
    )

    with np.load(tmp_path / "result.npz", allow_pickle=False) as saved:
        assert set(saved.files) == {"t_E", "psi", "pop", "E", "t_p"}
        for key, expected in {
            "t_E": grid.field_times_fs,
            "psi": state,
            "pop": population,
            "E": field_samples,
            "t_p": times_fs,
        }.items():
            np.testing.assert_array_equal(saved[key], expected)
        assert all(saved[key].dtype.kind != "O" for key in saved.files)
    assert json.loads((tmp_path / "parameters.json").read_text()) == params
    assert json.loads((tmp_path / "regime_analysis.json").read_text()) == regime_info
    assert (
        json.loads((tmp_path / "result_manifest.json").read_text())["schema_version"]
        == 1
    )


def test_summary_rejects_unversioned_result_without_overwriting_existing_csv(tmp_path):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    np.savez_compressed(case_dir / "result.npz", pop=np.array([[0.2, 0.8]]))
    summary = tmp_path / "summary.csv"
    summary.write_text("previous summary\n")

    with pytest.raises(ResultFormatError, match="unversioned"):
        storage.update_summary(tmp_path, [{"outdir": case_dir, "save": True}])

    assert summary.read_text() == "previous summary\n"


def test_summary_rejects_tampered_versioned_result(tmp_path):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    grid = TimeGrid.from_bounds(-1.0, 1.0, 0.5)
    persist_wavefunction_result(
        outdir=case_dir,
        params={"basis_type": "twolevel"},
        field_times_fs=grid.field_times_fs,
        sampled_field=ScalarField(grid, np.zeros(5)),
        times_fs=np.array([1.0]),
        state=np.array([1.0 + 0j, 0.0 + 0j]),
        population=np.array([[1.0, 0.0]]),
        regime_info=None,
    )
    (case_dir / "result.npz").write_bytes(b"changed after manifest")

    with pytest.raises(ResultFormatError, match="digest mismatch"):
        storage.update_summary(tmp_path, [{"outdir": case_dir, "save": True}])

    assert not (tmp_path / "summary.csv").exists()


def test_summary_rejects_published_manifest_with_missing_npz(tmp_path):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    grid = TimeGrid.from_bounds(-1.0, 1.0, 0.5)
    persist_wavefunction_result(
        outdir=case_dir,
        params={"basis_type": "twolevel"},
        field_times_fs=grid.field_times_fs,
        sampled_field=ScalarField(grid, np.zeros(5)),
        times_fs=np.array([1.0]),
        state=np.array([1.0 + 0j, 0.0 + 0j]),
        population=np.array([[1.0, 0.0]]),
        regime_info=None,
    )
    (case_dir / "result.npz").unlink()

    with pytest.raises(ResultFormatError, match="missing result payload"):
        storage.update_summary(tmp_path, [{"outdir": case_dir, "save": True}])

    assert not (tmp_path / "summary.csv").exists()


def test_results_root_name_and_summary_files_are_preserved(tmp_path, monkeypatch):
    class FixedDatetime:
        @classmethod
        def now(cls):
            return datetime(2026, 8, 15, 12, 34, 56)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(storage, "datetime", FixedDatetime)
    results_root = storage.make_results_root("persistence_contract")
    assert results_root == Path("results/20260815_123456_persistence_contract")
    assert results_root.is_dir()

    success_dir = tmp_path / "case_success"
    missing_dir = tmp_path / "case_missing"
    success_dir.mkdir()
    grid = TimeGrid.from_bounds(-1.0, 1.0, 0.5)
    persist_wavefunction_result(
        outdir=success_dir,
        params={"basis_type": "twolevel"},
        field_times_fs=grid.field_times_fs,
        sampled_field=ScalarField(grid, np.zeros(5)),
        times_fs=np.array([-1.0, 1.0]),
        state=np.zeros((2, 2), dtype=np.complex128),
        population=np.array([[1.0, 0.0], [0.25, 0.75]]),
        regime_info=None,
    )
    cases = [
        {"amplitude": 1.0, "outdir": success_dir, "save": True},
        {"amplitude": 3.0, "outdir": missing_dir, "save": True},
    ]

    storage.update_summary(results_root, cases)

    summary = pd.read_csv(results_root / "summary.csv")
    assert summary["status"].tolist() == ["success", "failed"]
    assert "outdir" not in summary.columns
    assert "save" not in summary.columns
    assert summary.loc[0, "pop_0"] == 0.25
    assert summary.loc[0, "pop_1"] == 0.75
    successful = pd.read_csv(results_root / "summary_success.csv")
    assert successful["amplitude"].tolist() == [1.0]
