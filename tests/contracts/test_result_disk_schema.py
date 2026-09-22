"""Contracts for an explicit, independently versioned simulation disk format."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from rovibrational_excitation.core.time import TimeGrid
from rovibrational_excitation.fields import ScalarField
from rovibrational_excitation.io.result_schema import (
    DISK_RESULT_SCHEMA_VERSION,
    ResultFormatError,
    load_simulation_result,
)
from rovibrational_excitation.simulation.m_average import MAveragePropagationResult
from rovibrational_excitation.simulation.result_persistence import (
    persist_m_average_result,
    persist_wavefunction_result,
)


def _example(tmp_path, *, regime_info=None):
    grid = TimeGrid.from_bounds(-1.0, 1.0, 0.5)
    field = ScalarField(grid, np.array([0.0, 2.0, -3.0, 2.0, 0.0]))
    times = np.array([-1.0, 0.0, 1.0])
    state = np.array(
        [[1.0, 0.0], [0.5 + 0.5j, 0.5 - 0.5j], [0.0, 1.0]],
        dtype=np.complex128,
    )
    population = np.abs(state) ** 2
    params = {
        "basis_type": "twolevel",
        "algorithm": "rk4",
        "backend": "numpy",
        "storage": "dense",
        "nondimensional": regime_info is not None,
        "renorm": False,
        "return_traj": True,
        "sample_stride": 1,
    }
    persist_wavefunction_result(
        outdir=tmp_path,
        params=params,
        field_times_fs=grid.field_times_fs,
        sampled_field=field,
        times_fs=times,
        state=state,
        population=population,
        regime_info=regime_info,
    )
    return params, grid, field, times, state, population


def test_versioned_wavefunction_round_trip_preserves_numeric_arrays(tmp_path):
    regime_info = {"energy_scale_eV": 0.25}
    params, grid, field, times, state, population = _example(
        tmp_path, regime_info=regime_info
    )

    saved = load_simulation_result(tmp_path)
    assert DISK_RESULT_SCHEMA_VERSION == 1
    assert saved.manifest["schema_version"] == 1
    assert saved.representation == "wavefunction"
    assert saved.parameters == params
    assert saved.regime_info == regime_info
    assert saved.manifest["units"] == {
        "t_E": "fs",
        "t_p": "fs",
        "E": "V/m",
        "psi": "1",
        "pop": "1",
    }
    assert saved.manifest["declared"]["model"] == "twolevel"
    assert saved.manifest["declared"]["backend"] == "numpy"
    for key, expected in {
        "t_E": grid.field_times_fs,
        "t_p": times,
        "E": field.samples_v_per_m,
        "psi": state,
        "pop": population,
    }.items():
        np.testing.assert_array_equal(saved.arrays[key], expected)

    with np.load(tmp_path / "result.npz", allow_pickle=False) as raw:
        assert set(raw.files) == {"t_E", "t_p", "E", "psi", "pop"}
        for key in raw.files:
            assert raw[key].dtype.kind != "O"


def test_versioned_m_average_round_trip_preserves_block_arrays(tmp_path):
    grid = TimeGrid.from_bounds(-1.0, 1.0, 0.5)
    field = ScalarField(grid, np.zeros(5))
    times = np.array([-1.0, 0.0, 1.0])
    population = np.array([[1.0, 0.0], [0.8, 0.2], [0.6, 0.4]])
    trajectory = np.array([[1.0, 0.0], [0.8 + 0.1j, 0.5], [0.7, 0.6j]])
    result = MAveragePropagationResult(
        time_fs=times,
        population=population,
        blocks=(SimpleNamespace(abs_m=0, multiplicity=1, weight=1.0),),
        block_wavefunctions=(trajectory,),
    )
    persist_m_average_result(
        outdir=tmp_path,
        params={"basis_type": "linmol", "representation": "m_incoherent_average"},
        field_times_fs=grid.field_times_fs,
        sampled_field=field,
        result=result,
    )

    saved = load_simulation_result(tmp_path)
    assert saved.representation == "m_incoherent_average"
    np.testing.assert_array_equal(saved.arrays["pop"], population)
    np.testing.assert_array_equal(saved.arrays["psi_abs_m_0"], trajectory)
    np.testing.assert_array_equal(saved.arrays["m_weight"], np.array([1.0]))


def test_missing_and_unknown_disk_schema_raise_actionable_errors(tmp_path):
    np.savez_compressed(tmp_path / "result.npz", pop=np.array([[1.0, 0.0]]))
    with pytest.raises(ResultFormatError, match="unversioned"):
        load_simulation_result(tmp_path)

    _example(tmp_path)
    manifest_path = tmp_path / "result_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = 999
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ResultFormatError, match="unsupported.*999"):
        load_simulation_result(tmp_path)


@pytest.mark.parametrize("invalid", [["wavefunction"], {"kind": "wavefunction"}])
def test_loader_rejects_malformed_representation_with_format_error(tmp_path, invalid):
    _example(tmp_path)
    manifest_path = tmp_path / "result_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["representation"] = invalid
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ResultFormatError, match="representation"):
        load_simulation_result(tmp_path)


def test_loader_rejects_invalid_npz_even_with_matching_digest(tmp_path):
    _example(tmp_path)
    payload = b"not a zip archive"
    (tmp_path / "result.npz").write_bytes(payload)
    manifest_path = tmp_path / "result_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["result.npz"] = hashlib.sha256(payload).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ResultFormatError, match="cannot safely read result.npz"):
        load_simulation_result(tmp_path)


def test_loader_rejects_corrupted_result_without_guessing_from_keys(tmp_path):
    _example(tmp_path)
    (tmp_path / "result.npz").write_bytes(b"not the committed result")
    with pytest.raises(ResultFormatError, match="digest mismatch"):
        load_simulation_result(tmp_path)
