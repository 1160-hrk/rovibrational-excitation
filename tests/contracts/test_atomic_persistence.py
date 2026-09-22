"""Atomic single-file persistence contracts for large result payloads."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from rovibrational_excitation.io.atomic import atomic_write_json, atomic_write_npz


def test_atomic_npz_replaces_complete_file_without_changing_arrays(tmp_path):
    path = tmp_path / "result.npz"
    np.savez_compressed(path, pop=np.array([[1.0, 0.0]]))
    expected = np.array([[0.25, 0.75]], dtype=np.float64)

    atomic_write_npz(path, {"pop": expected})

    with np.load(path, allow_pickle=False) as saved:
        assert saved.files == ["pop"]
        np.testing.assert_array_equal(saved["pop"], expected)
    assert sorted(tmp_path.iterdir()) == [path]


def test_npz_write_failure_preserves_previous_file_and_removes_temp(
    tmp_path, monkeypatch
):
    path = tmp_path / "result.npz"
    previous = b"previous complete result"
    path.write_bytes(previous)

    def fail_after_partial_write(temporary, **arrays):
        Path(temporary).write_bytes(b"partial")
        raise OSError("simulated NPZ write failure")

    monkeypatch.setattr(np, "savez_compressed", fail_after_partial_write)
    with pytest.raises(OSError, match="simulated NPZ write failure"):
        atomic_write_npz(path, {"pop": np.array([[1.0]])})

    assert path.read_bytes() == previous
    assert sorted(tmp_path.iterdir()) == [path]


def test_json_replace_failure_preserves_previous_file_and_removes_temp(
    tmp_path, monkeypatch
):
    path = tmp_path / "parameters.json"
    previous = b'{"old": true}'
    path.write_bytes(previous)

    def fail_replace(source, destination):
        assert Path(destination) == path
        assert Path(source).parent == tmp_path
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        atomic_write_json(path, {"new": True})

    assert path.read_bytes() == previous
    assert sorted(tmp_path.iterdir()) == [path]


def test_atomic_json_keeps_existing_serialization_shape(tmp_path):
    path = tmp_path / "parameters.json"
    atomic_write_json(path, {"complex": {"__complex__": True, "r": 1.0, "i": -2.0}})

    assert json.loads(path.read_text()) == {
        "complex": {"__complex__": True, "r": 1.0, "i": -2.0}
    }
    assert sorted(tmp_path.iterdir()) == [path]
