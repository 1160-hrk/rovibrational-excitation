"""Strict versioned disk format for normal simulation results.

The manifest is written after the established numerical NPZ and JSON sidecars.
Each file is atomically replaced; cross-file publication remains a separate unit.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

import numpy as np

from .atomic import atomic_write_json

DISK_RESULT_SCHEMA_VERSION = 1
MANIFEST_NAME = "result_manifest.json"
_DECLARED_PARAMETERS = {
    "model": "basis_type",
    "algorithm": "algorithm",
    "backend": "backend",
    "storage": "storage",
    "nondimensional": "nondimensional",
    "renorm": "renorm",
    "trajectory": "return_traj",
    "sample_stride": "sample_stride",
}
_BASE_ARRAYS = {"t_E", "t_p", "E", "pop"}


class ResultFormatError(ValueError):
    """The on-disk result is absent, unsupported, incomplete, or inconsistent."""


@dataclass(frozen=True)
class StoredSimulationResult:
    """Validated numeric arrays and JSON sidecars from one disk result."""

    representation: str
    arrays: dict[str, np.ndarray]
    parameters: dict[str, Any]
    regime_info: Any | None
    manifest: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _declared(parameters: dict[str, Any]) -> dict[str, Any]:
    """Project only caller-declared metadata; missing values remain explicit nulls."""
    return {
        name: parameters.get(parameter_key)
        for name, parameter_key in _DECLARED_PARAMETERS.items()
    }


def _units(representation: str, array_keys: set[str]) -> dict[str, str]:
    units = {"t_E": "fs", "t_p": "fs", "E": "V/m", "pop": "1"}
    if representation == "wavefunction":
        units["psi"] = "1"
    else:
        for key in array_keys:
            if key.startswith("psi_abs_m_"):
                units[key] = "1"
        units.update({"abs_m": "1", "m_multiplicity": "1", "m_weight": "1"})
    return units


def write_result_manifest(
    outdir: Path,
    *,
    representation: str,
    arrays: dict[str, Any],
    has_regime_info: bool,
) -> None:
    """Publish a versioned description after the existing payload files exist."""
    if representation not in {"wavefunction", "m_incoherent_average"}:
        raise ValueError(f"unsupported result representation: {representation}")
    parameter_path = outdir / "parameters.json"
    parameters = json.loads(parameter_path.read_text(encoding="utf-8"))
    if not isinstance(parameters, dict):
        raise ValueError("parameters.json must contain a JSON object")

    array_metadata: dict[str, dict[str, Any]] = {}
    for key, value in arrays.items():
        array = np.asarray(value)
        if array.dtype.kind == "O":
            raise ValueError(f"result array {key!r} must not have object dtype")
        array_metadata[key] = {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
        }

    file_names = ["result.npz", "parameters.json"]
    if has_regime_info:
        file_names.append("regime_analysis.json")
    manifest = {
        "schema_version": DISK_RESULT_SCHEMA_VERSION,
        "representation": representation,
        "files": {name: _sha256(outdir / name) for name in file_names},
        "arrays": array_metadata,
        "units": _units(representation, set(array_metadata)),
        "declared": _declared(parameters),
    }
    atomic_write_json(outdir / MANIFEST_NAME, manifest, sort_keys=True)


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ResultFormatError(f"cannot read {label}: {path}") from exc


def _validate_manifest(manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise ResultFormatError("result manifest must be a JSON object")
    version = manifest.get("schema_version")
    if version is None:
        raise ResultFormatError(
            "unversioned result manifest; explicit migration required"
        )
    if type(version) is not int or version != DISK_RESULT_SCHEMA_VERSION:
        raise ResultFormatError(
            f"unsupported result disk schema version {version!r}; "
            f"supported version: {DISK_RESULT_SCHEMA_VERSION}"
        )
    expected = {
        "schema_version",
        "representation",
        "files",
        "arrays",
        "units",
        "declared",
    }
    if set(manifest) != expected:
        raise ResultFormatError("result manifest has missing or unknown fields")
    representation = manifest["representation"]
    if not isinstance(representation, str) or representation not in {
        "wavefunction",
        "m_incoherent_average",
    }:
        raise ResultFormatError("result manifest has unknown representation")
    if not isinstance(manifest["files"], dict):
        raise ResultFormatError("result manifest files must be a mapping")
    if not isinstance(manifest["arrays"], dict):
        raise ResultFormatError("result manifest arrays must be a mapping")
    return manifest


def _validate_array_keys(representation: str, arrays: dict[str, np.ndarray]) -> None:
    keys = set(arrays)
    if representation == "wavefunction":
        if keys != _BASE_ARRAYS | {"psi"}:
            raise ResultFormatError("wavefunction result has unexpected array keys")
        return

    required = _BASE_ARRAYS | {
        "representation",
        "abs_m",
        "m_multiplicity",
        "m_weight",
    }
    if not required <= keys:
        raise ResultFormatError("M-average result is missing required arrays")
    stored_representation = arrays["representation"]
    if (
        stored_representation.shape != ()
        or stored_representation.item() != representation
    ):
        raise ResultFormatError("M-average representation disagrees with manifest")
    abs_m = arrays["abs_m"]
    if abs_m.ndim != 1 or abs_m.dtype.kind not in "iu":
        raise ResultFormatError(
            "M-average abs_m must be a one-dimensional integer array"
        )
    indices = [int(value) for value in abs_m]
    if len(indices) != len(set(indices)) or any(value < 0 for value in indices):
        raise ResultFormatError("M-average abs_m must be unique and nonnegative")
    expected = required | {f"psi_abs_m_{value}" for value in indices}
    if keys != expected:
        raise ResultFormatError("M-average block arrays disagree with abs_m")


def load_simulation_result(outdir: Path) -> StoredSimulationResult:
    """Load only a known complete disk schema; never infer legacy layouts."""
    outdir = Path(outdir)
    manifest_path = outdir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ResultFormatError(
            f"unversioned or incomplete result at {outdir}: {MANIFEST_NAME} "
            "is missing; explicit migration required"
        )
    manifest = _validate_manifest(_read_json(manifest_path, MANIFEST_NAME))
    file_hashes = manifest["files"]
    expected_files = {"result.npz", "parameters.json"}
    if "regime_analysis.json" in file_hashes:
        expected_files.add("regime_analysis.json")
    if set(file_hashes) != expected_files:
        raise ResultFormatError("result manifest has missing or unknown payload files")
    for name, expected_hash in file_hashes.items():
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or any(char not in "0123456789abcdef" for char in expected_hash)
        ):
            raise ResultFormatError(f"invalid digest for {name}")
        path = outdir / name
        try:
            actual_hash = _sha256(path)
        except OSError as exc:
            raise ResultFormatError(f"missing result payload: {path}") from exc
        if actual_hash != expected_hash:
            raise ResultFormatError(f"result payload digest mismatch: {path}")

    parameters = _read_json(outdir / "parameters.json", "parameters.json")
    if not isinstance(parameters, dict):
        raise ResultFormatError("parameters.json must contain a JSON object")
    if manifest["declared"] != _declared(parameters):
        raise ResultFormatError("declared metadata disagrees with parameters.json")
    regime_info = (
        _read_json(outdir / "regime_analysis.json", "regime_analysis.json")
        if "regime_analysis.json" in file_hashes
        else None
    )

    metadata = manifest["arrays"]
    if not isinstance(metadata, dict) or any(
        not isinstance(key, str) or not isinstance(value, dict)
        for key, value in metadata.items()
    ):
        raise ResultFormatError("result array metadata must be a mapping")
    arrays: dict[str, np.ndarray] = {}
    try:
        with np.load(outdir / "result.npz", allow_pickle=False) as saved:
            if set(saved.files) != set(metadata):
                raise ResultFormatError("result array keys disagree with manifest")
            for key, declared in metadata.items():
                array = saved[key]
                if array.dtype.kind == "O":
                    raise ResultFormatError(f"object array {key!r} is not supported")
                if declared != {
                    "dtype": array.dtype.str,
                    "shape": list(array.shape),
                }:
                    raise ResultFormatError(
                        f"result array {key!r} shape or dtype disagrees with manifest"
                    )
                arrays[key] = array
    except (OSError, ValueError, KeyError, BadZipFile) as exc:
        if isinstance(exc, ResultFormatError):
            raise
        raise ResultFormatError("cannot safely read result.npz") from exc

    representation = manifest["representation"]
    _validate_array_keys(representation, arrays)
    if manifest["units"] != _units(representation, set(arrays)):
        raise ResultFormatError("result units disagree with disk schema")
    return StoredSimulationResult(
        representation=representation,
        arrays=arrays,
        parameters=parameters,
        regime_info=regime_info,
        manifest=manifest,
    )


__all__ = [
    "DISK_RESULT_SCHEMA_VERSION",
    "ResultFormatError",
    "StoredSimulationResult",
    "load_simulation_result",
    "write_result_manifest",
]
