"""Atomic publication contracts for complete normal-simulation result bundles."""

from __future__ import annotations

import json
import os
import shutil

import numpy as np
import pytest

from rovibrational_excitation.core.time import TimeGrid
from rovibrational_excitation.fields import ScalarField
from rovibrational_excitation.io import result_schema
from rovibrational_excitation.io.result_schema import (
    CURRENT_RESULT_NAME,
    ResultFormatError,
    load_simulation_result,
    resolve_result_directory,
)
from rovibrational_excitation.simulation import result_persistence


def _write_result(outdir, *, final_population):
    grid = TimeGrid.from_bounds(-1.0, 1.0, 0.5)
    state = np.array([1.0 + 0j, 0.0 + 0j])
    result_persistence.persist_wavefunction_result(
        outdir=outdir,
        params={"basis_type": "twolevel"},
        field_times_fs=grid.field_times_fs,
        sampled_field=ScalarField(grid, np.zeros(5)),
        times_fs=np.array([1.0]),
        state=state,
        population=np.array([[1.0 - final_population, final_population]]),
        regime_info=None,
    )


def test_complete_result_is_published_by_one_pointer(tmp_path):
    _write_result(tmp_path, final_population=0.25)

    pointer = json.loads((tmp_path / CURRENT_RESULT_NAME).read_text())
    generation = resolve_result_directory(tmp_path)
    assert pointer == {
        "publication_schema_version": 1,
        "generation": generation.name,
    }
    assert generation.parent == tmp_path / ".result_generations"
    assert generation != tmp_path
    assert not (tmp_path / "result.npz").exists()
    assert {path.name for path in generation.iterdir()} == {
        "result.npz",
        "parameters.json",
        "result_manifest.json",
    }
    np.testing.assert_array_equal(
        load_simulation_result(tmp_path).arrays["pop"], [[0.75, 0.25]]
    )


def test_second_publication_switches_complete_result_without_mutating_prior(tmp_path):
    _write_result(tmp_path, final_population=0.25)
    previous_dir = resolve_result_directory(tmp_path)

    _write_result(tmp_path, final_population=0.75)

    current_dir = resolve_result_directory(tmp_path)
    assert current_dir != previous_dir
    np.testing.assert_array_equal(
        load_simulation_result(tmp_path).arrays["pop"], [[0.25, 0.75]]
    )
    np.testing.assert_array_equal(
        load_simulation_result(previous_dir).arrays["pop"], [[0.75, 0.25]]
    )


def test_failed_payload_write_keeps_previous_complete_result(tmp_path, monkeypatch):
    _write_result(tmp_path, final_population=0.25)
    pointer_path = tmp_path / CURRENT_RESULT_NAME
    previous = pointer_path.read_bytes()

    def fail_parameters(path, value, **kwargs):
        if path.name == "parameters.json":
            raise OSError("simulated payload write failure")
        raise AssertionError("unexpected JSON write")

    monkeypatch.setattr(result_persistence, "atomic_write_json", fail_parameters)
    with pytest.raises(OSError, match="simulated payload write failure"):
        _write_result(tmp_path, final_population=0.75)

    assert pointer_path.read_bytes() == previous
    np.testing.assert_array_equal(
        load_simulation_result(tmp_path).arrays["pop"], [[0.75, 0.25]]
    )


def test_failed_manifest_write_keeps_previous_complete_result(tmp_path, monkeypatch):
    _write_result(tmp_path, final_population=0.25)
    pointer_path = tmp_path / CURRENT_RESULT_NAME
    previous = pointer_path.read_bytes()
    original_json_write = result_schema.atomic_write_json

    def fail_manifest(path, value, **kwargs):
        if path.name == "result_manifest.json":
            raise OSError("simulated manifest write failure")
        return original_json_write(path, value, **kwargs)

    monkeypatch.setattr(result_schema, "atomic_write_json", fail_manifest)
    with pytest.raises(OSError, match="simulated manifest write failure"):
        _write_result(tmp_path, final_population=0.75)

    assert pointer_path.read_bytes() == previous
    np.testing.assert_array_equal(
        load_simulation_result(tmp_path).arrays["pop"], [[0.75, 0.25]]
    )


def test_failed_pointer_replace_keeps_previous_complete_result(tmp_path, monkeypatch):
    _write_result(tmp_path, final_population=0.25)
    pointer_path = tmp_path / CURRENT_RESULT_NAME
    previous = pointer_path.read_bytes()
    original_replace = os.replace

    def fail_pointer_replace(source, destination):
        if destination == pointer_path:
            raise OSError("simulated pointer replace failure")
        return original_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_pointer_replace)
    with pytest.raises(OSError, match="simulated pointer replace failure"):
        _write_result(tmp_path, final_population=0.75)

    assert pointer_path.read_bytes() == previous
    np.testing.assert_array_equal(
        load_simulation_result(tmp_path).arrays["pop"], [[0.75, 0.25]]
    )


@pytest.mark.parametrize(
    "pointer",
    [
        {"publication_schema_version": 2, "generation": "a" * 32},
        {"publication_schema_version": 1, "generation": "a" * 32},
        {"publication_schema_version": 1, "generation": "../other"},
        {"publication_schema_version": 1, "generation": "a" * 32, "extra": 1},
    ],
)
def test_invalid_pointer_never_falls_back_to_legacy_files(tmp_path, pointer):
    (tmp_path / "result.npz").write_bytes(b"legacy placeholder")
    (tmp_path / CURRENT_RESULT_NAME).write_text(json.dumps(pointer))
    with pytest.raises(ResultFormatError, match="publication"):
        load_simulation_result(tmp_path)


def test_existing_valid_v1_direct_layout_remains_readable(tmp_path):
    published = tmp_path / "published"
    published.mkdir()
    _write_result(published, final_population=0.25)
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    for name in ("result.npz", "parameters.json", "result_manifest.json"):
        shutil.copy2(resolve_result_directory(published) / name, legacy / name)

    assert resolve_result_directory(legacy) == legacy
    np.testing.assert_array_equal(
        load_simulation_result(legacy).arrays["pop"], [[0.75, 0.25]]
    )


def test_legacy_direct_layout_requires_explicit_overwrite_migration(tmp_path):
    (tmp_path / "result.npz").write_bytes(b"legacy placeholder")
    with pytest.raises(ResultFormatError, match="explicit migration"):
        _write_result(tmp_path, final_population=0.25)
    assert not (tmp_path / CURRENT_RESULT_NAME).exists()
