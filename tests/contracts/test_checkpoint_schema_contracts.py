"""Strict checkpoint payload and resume-provenance contracts."""

from __future__ import annotations

import json

import pytest

from rovibrational_excitation.io.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointFormatError,
    CheckpointManager,
    checkpoint_run_fingerprint,
    resolve_checkpoint_directory,
)


def _cases() -> list[dict[str, object]]:
    return [
        {"amplitude": 1.0, "save": True, "outdir": "first"},
        {"amplitude": 2.0, "save": True, "outdir": "second"},
    ]


def _manager_with_checkpoint(tmp_path) -> CheckpointManager:
    cases = _cases()
    manager = CheckpointManager(tmp_path, all_cases=cases)
    manager.save_checkpoint([cases[0]], [], len(cases), 1.25)
    return manager


def test_checkpoint_v1_records_exact_declared_run_fingerprint(tmp_path):
    manager = _manager_with_checkpoint(tmp_path)

    checkpoint = manager.load_checkpoint()
    assert checkpoint is not None
    assert checkpoint["checkpoint_schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert checkpoint["run_provenance"] == {
        "digest_algorithm": "sha256",
        "scope": "ordered_declared_cases_excluding_outdir_save_error",
        "case_count": 2,
        "cases_sha256": checkpoint_run_fingerprint(_cases()),
    }


def test_checkpoint_reader_rejects_unknown_and_unversioned_payloads(tmp_path):
    generation = tmp_path / ".checkpoint_generations" / ("a" * 32)
    generation.mkdir(parents=True)
    (generation / "failed_cases.json").write_text("[]")
    (tmp_path / "checkpoint_current.json").write_text(
        json.dumps({"publication_schema_version": 1, "generation": "a" * 32})
    )
    manager = CheckpointManager(tmp_path)

    (generation / "checkpoint.json").write_text(json.dumps({"completed_cases": 0}))
    with pytest.raises(CheckpointFormatError, match="unversioned checkpoint"):
        manager.load_checkpoint()

    (generation / "checkpoint.json").write_text(
        json.dumps({"checkpoint_schema_version": 999})
    )
    with pytest.raises(CheckpointFormatError, match="unsupported checkpoint schema"):
        manager.load_checkpoint()


def test_checkpoint_reader_rejects_tampered_failure_sidecar(tmp_path):
    manager = _manager_with_checkpoint(tmp_path)
    generation = resolve_checkpoint_directory(tmp_path)
    (generation / "failed_cases.json").write_text('[{"amplitude": 99.0}]')

    with pytest.raises(CheckpointFormatError, match="failed_cases.json"):
        manager.load_checkpoint()


def test_resume_case_binding_rejects_changed_scientific_inputs(tmp_path):
    manager = _manager_with_checkpoint(tmp_path)
    changed = _cases()
    changed[1]["amplitude"] = 3.0

    with pytest.raises(CheckpointFormatError, match="run provenance mismatch"):
        manager.filter_remaining_cases(changed)


def test_resume_rejects_completed_hash_outside_declared_run(tmp_path):
    _manager_with_checkpoint(tmp_path)
    generation = resolve_checkpoint_directory(tmp_path)
    checkpoint_path = generation / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["completed_case_hashes"] = ["b" * 32]
    checkpoint_path.write_text(json.dumps(checkpoint))
    resumed = CheckpointManager(tmp_path)

    with pytest.raises(CheckpointFormatError, match="do not belong"):
        resumed.filter_remaining_cases(_cases())


def test_runtime_paths_do_not_change_run_fingerprint():
    first = _cases()
    second = _cases()
    second[0].update({"save": False, "outdir": "elsewhere", "error": "old"})

    assert checkpoint_run_fingerprint(first) == checkpoint_run_fingerprint(second)


def test_saving_requires_bound_complete_run_provenance(tmp_path):
    manager = CheckpointManager(tmp_path)

    with pytest.raises(RuntimeError, match="all_cases"):
        manager.save_checkpoint([], [], 0, 0.0)


def test_writer_rejects_invalid_payload_before_creating_generation(tmp_path):
    manager = CheckpointManager(tmp_path, all_cases=[])

    with pytest.raises(CheckpointFormatError, match="start_time"):
        manager.save_checkpoint([], [], 0, float("nan"))

    assert not (tmp_path / "checkpoint_current.json").exists()
    assert not (tmp_path / ".checkpoint_generations").exists()


def test_incomplete_direct_pair_cannot_be_silently_replaced(tmp_path):
    (tmp_path / "failed_cases.json").write_text("[]")
    manager = CheckpointManager(tmp_path, all_cases=[])

    assert manager.is_resumable()
    with pytest.raises(CheckpointFormatError, match="payload pair"):
        manager.load_checkpoint()
    with pytest.raises(CheckpointFormatError, match="payload pair"):
        manager.save_checkpoint([], [], 0, 0.0)

    assert not (tmp_path / "checkpoint_current.json").exists()
    assert not (tmp_path / ".checkpoint_generations").exists()


def test_legacy_direct_checkpoint_requires_explicit_migration(tmp_path):
    (tmp_path / "checkpoint.json").write_text(json.dumps({"completed_cases": 0}))
    (tmp_path / "failed_cases.json").write_text("[]")
    manager = CheckpointManager(tmp_path, all_cases=[])

    with pytest.raises(CheckpointFormatError, match="unversioned checkpoint"):
        manager.load_checkpoint()
    with pytest.raises(CheckpointFormatError, match="explicit migration"):
        manager.save_checkpoint([], [], 0, 0.0)
