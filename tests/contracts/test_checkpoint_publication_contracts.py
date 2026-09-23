"""Atomic publication contracts for checkpoint/failure-list pairs."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from rovibrational_excitation.io.checkpoint import (
    CHECKPOINT_CURRENT_NAME,
    CheckpointManager,
    resolve_checkpoint_directory,
)


def _save(
    manager: CheckpointManager, amplitude: float, *, failed: bool = False
) -> None:
    case = {"amplitude": amplitude}
    manager.save_checkpoint(
        [] if failed else [case],
        [{**case, "error": "failure"}] if failed else [],
        1,
        amplitude,
    )


def test_complete_checkpoint_pair_is_published_by_one_pointer(tmp_path):
    manager = CheckpointManager(tmp_path)
    _save(manager, 1.0, failed=True)

    pointer = json.loads((tmp_path / CHECKPOINT_CURRENT_NAME).read_text())
    generation = resolve_checkpoint_directory(tmp_path)
    assert pointer == {
        "publication_schema_version": 1,
        "generation": generation.name,
    }
    assert generation.parent == tmp_path / ".checkpoint_generations"
    assert {path.name for path in generation.iterdir()} == {
        "checkpoint.json",
        "failed_cases.json",
    }
    assert not (tmp_path / "checkpoint.json").exists()
    assert manager.load_checkpoint()["failed_case_data"] == [
        {"amplitude": 1.0, "error": "failure"}
    ]


def test_second_checkpoint_switches_pair_without_mutating_previous(tmp_path):
    manager = CheckpointManager(tmp_path)
    _save(manager, 1.0)
    previous = resolve_checkpoint_directory(tmp_path)
    previous_files = {
        name: (previous / name).read_bytes()
        for name in ("checkpoint.json", "failed_cases.json")
    }

    _save(manager, 2.0, failed=True)

    current = resolve_checkpoint_directory(tmp_path)
    assert current != previous
    assert {
        name: (previous / name).read_bytes()
        for name in ("checkpoint.json", "failed_cases.json")
    } == previous_files
    assert manager.load_checkpoint()["start_time"] == 2.0


@pytest.mark.parametrize("failed_name", ["checkpoint.json", "failed_cases.json"])
def test_failed_pair_payload_keeps_previous_checkpoint(
    tmp_path, monkeypatch, failed_name
):
    manager = CheckpointManager(tmp_path)
    _save(manager, 1.0)
    pointer = tmp_path / CHECKPOINT_CURRENT_NAME
    previous_pointer = pointer.read_bytes()
    previous_checkpoint = manager.load_checkpoint()
    from rovibrational_excitation.io import checkpoint as checkpoint_module

    original_write = checkpoint_module.atomic_write_json

    def fail_selected(path, value, **kwargs):
        if path.name == failed_name:
            raise OSError("simulated checkpoint payload failure")
        return original_write(path, value, **kwargs)

    monkeypatch.setattr(checkpoint_module, "atomic_write_json", fail_selected)
    with pytest.raises(OSError, match="simulated checkpoint payload failure"):
        _save(manager, 2.0, failed=True)

    assert pointer.read_bytes() == previous_pointer
    assert manager.load_checkpoint() == previous_checkpoint


def test_failed_pointer_replace_keeps_previous_checkpoint(tmp_path, monkeypatch):
    manager = CheckpointManager(tmp_path)
    _save(manager, 1.0)
    pointer = tmp_path / CHECKPOINT_CURRENT_NAME
    previous_pointer = pointer.read_bytes()
    previous_checkpoint = manager.load_checkpoint()
    original_replace = os.replace

    def fail_pointer(source, destination):
        if Path(destination) == pointer:
            raise OSError("simulated checkpoint pointer failure")
        return original_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_pointer)
    with pytest.raises(OSError, match="simulated checkpoint pointer failure"):
        _save(manager, 2.0, failed=True)

    assert pointer.read_bytes() == previous_pointer
    assert manager.load_checkpoint() == previous_checkpoint


@pytest.mark.parametrize(
    "pointer",
    [
        {"publication_schema_version": 2, "generation": "a" * 32},
        {"publication_schema_version": 1, "generation": "a" * 32},
        {"publication_schema_version": 1, "generation": "../other"},
        {"publication_schema_version": 1, "generation": "a" * 32, "extra": 1},
    ],
)
def test_invalid_pointer_never_falls_back_to_legacy_checkpoint(
    tmp_path, capsys, pointer
):
    (tmp_path / "checkpoint.json").write_text('{"completed_cases": 99}')
    pointer_path = tmp_path / CHECKPOINT_CURRENT_NAME
    pointer_path.write_text(json.dumps(pointer))
    previous_pointer = pointer_path.read_bytes()
    manager = CheckpointManager(tmp_path)

    assert manager.is_resumable()
    assert manager.load_checkpoint() is None
    assert "チェックポイント読み込み失敗" in capsys.readouterr().out
    with pytest.raises((ValueError, json.JSONDecodeError), match="checkpoint"):
        _save(manager, 2.0)
    assert pointer_path.read_bytes() == previous_pointer


def test_malformed_pointer_is_not_silently_repaired(tmp_path, capsys):
    pointer_path = tmp_path / CHECKPOINT_CURRENT_NAME
    pointer_path.write_text("{invalid json")
    previous_pointer = pointer_path.read_bytes()
    manager = CheckpointManager(tmp_path)

    assert manager.load_checkpoint() is None
    assert "チェックポイント読み込み失敗" in capsys.readouterr().out
    with pytest.raises(json.JSONDecodeError):
        _save(manager, 2.0)
    assert pointer_path.read_bytes() == previous_pointer


def test_legacy_direct_checkpoint_is_readable_and_next_save_upgrades(tmp_path):
    legacy = {
        "timestamp": "2026-09-23T00:00:00",
        "start_time": 1.0,
        "total_cases": 1,
        "completed_cases": 1,
        "failed_cases": 0,
        "completed_case_hashes": ["legacy"],
        "failed_case_data": [],
    }
    (tmp_path / "checkpoint.json").write_text(json.dumps(legacy))
    (tmp_path / "failed_cases.json").write_text("[]")
    manager = CheckpointManager(tmp_path)

    assert manager.load_checkpoint() == legacy
    _save(manager, 2.0)

    assert resolve_checkpoint_directory(tmp_path) != tmp_path
    assert manager.load_checkpoint()["start_time"] == 2.0
