"""Checkpoint persistence for parameter sweeps."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .atomic import atomic_write_json
from .serialization import json_safe

CHECKPOINT_CURRENT_NAME = "checkpoint_current.json"
CHECKPOINT_SCHEMA_VERSION = 1
_CHECKPOINT_GENERATIONS_NAME = ".checkpoint_generations"
_CHECKPOINT_PUBLICATION_SCHEMA_VERSION = 1
_CHECKPOINT_PAYLOAD_NAMES = ("checkpoint.json", "failed_cases.json")
_RUNTIME_CASE_KEYS = {"outdir", "save", "error"}
_RUN_PROVENANCE_SCOPE = "ordered_declared_cases_excluding_outdir_save_error"
_CHECKPOINT_FIELDS = {
    "checkpoint_schema_version",
    "timestamp",
    "start_time",
    "total_cases",
    "completed_cases",
    "failed_cases",
    "completed_case_hashes",
    "failed_case_data",
    "run_provenance",
}
_PROVENANCE_FIELDS = {
    "digest_algorithm",
    "scope",
    "case_count",
    "cases_sha256",
}


class CheckpointFormatError(ValueError):
    """The checkpoint is unsupported, incomplete, or internally inconsistent."""


def _case_without_runtime(case: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in case.items() if key not in _RUNTIME_CASE_KEYS}


def _canonical_cases(all_cases: list[dict[str, Any]]) -> bytes:
    try:
        return json.dumps(
            json_safe([_case_without_runtime(case) for case in all_cases]),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "all_cases must have a finite JSON-safe representation for provenance"
        ) from exc


def checkpoint_run_fingerprint(all_cases: list[dict[str, Any]]) -> str:
    """Hash the ordered declared cases while excluding runtime-only fields."""
    return hashlib.sha256(_canonical_cases(all_cases)).hexdigest()


def _run_provenance(all_cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "digest_algorithm": "sha256",
        "scope": _RUN_PROVENANCE_SCOPE,
        "case_count": len(all_cases),
        "cases_sha256": checkpoint_run_fingerprint(all_cases),
    }


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CheckpointFormatError(f"cannot read {label}: {path}") from exc


def resolve_checkpoint_directory(root_dir: Path) -> Path:
    """Resolve one published checkpoint pair or the legacy direct layout."""
    root_dir = Path(root_dir)
    pointer_path = root_dir / CHECKPOINT_CURRENT_NAME
    if not pointer_path.exists() and not pointer_path.is_symlink():
        return root_dir
    if pointer_path.is_symlink():
        raise CheckpointFormatError(
            "checkpoint publication pointer must not be a symlink"
        )
    pointer = _read_json(pointer_path, CHECKPOINT_CURRENT_NAME)
    if not isinstance(pointer, dict) or set(pointer) != {
        "publication_schema_version",
        "generation",
    }:
        raise CheckpointFormatError("checkpoint publication pointer has invalid fields")
    version = pointer["publication_schema_version"]
    if type(version) is not int or version != _CHECKPOINT_PUBLICATION_SCHEMA_VERSION:
        raise CheckpointFormatError(
            f"unsupported checkpoint publication schema version {version!r}"
        )
    generation = pointer["generation"]
    if (
        not isinstance(generation, str)
        or re.fullmatch(r"[0-9a-f]{32}", generation) is None
    ):
        raise CheckpointFormatError("checkpoint publication generation is invalid")
    generations_root = root_dir / _CHECKPOINT_GENERATIONS_NAME
    generation_dir = generations_root / generation
    if generations_root.is_symlink() or generation_dir.is_symlink():
        raise CheckpointFormatError("checkpoint publication generation is unsafe")
    if not generation_dir.is_dir():
        raise CheckpointFormatError("checkpoint publication generation is missing")
    for name in _CHECKPOINT_PAYLOAD_NAMES:
        path = generation_dir / name
        if path.is_symlink() or not path.is_file():
            raise CheckpointFormatError(
                f"checkpoint publication payload is missing or unsafe: {name}"
            )
    return generation_dir


def _create_checkpoint_generation(root_dir: Path) -> Path:
    pointer_path = root_dir / CHECKPOINT_CURRENT_NAME
    if pointer_path.exists() or pointer_path.is_symlink():
        resolve_checkpoint_directory(root_dir)
    generations_root = root_dir / _CHECKPOINT_GENERATIONS_NAME
    if generations_root.is_symlink():
        raise ValueError("checkpoint generation directory must not be a symlink")
    generations_root.mkdir(exist_ok=True)
    generation_dir = generations_root / uuid4().hex
    generation_dir.mkdir()
    return generation_dir


def _publish_checkpoint_generation(root_dir: Path, generation_dir: Path) -> None:
    for name in _CHECKPOINT_PAYLOAD_NAMES:
        path = generation_dir / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"cannot publish incomplete checkpoint pair: {name}")
    atomic_write_json(
        root_dir / CHECKPOINT_CURRENT_NAME,
        {
            "publication_schema_version": _CHECKPOINT_PUBLICATION_SCHEMA_VERSION,
            "generation": generation_dir.name,
        },
        sort_keys=True,
    )


def _require_count(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise CheckpointFormatError(f"checkpoint {name} must be a nonnegative integer")
    return value


def _require_finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CheckpointFormatError(f"checkpoint {name} must be finite")
    try:
        number = float(value)
    except OverflowError as exc:
        raise CheckpointFormatError(f"checkpoint {name} must be finite") from exc
    if not math.isfinite(number):
        raise CheckpointFormatError(f"checkpoint {name} must be finite")
    return number


def _validate_provenance(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _PROVENANCE_FIELDS:
        raise CheckpointFormatError("checkpoint run_provenance has invalid fields")
    if value["digest_algorithm"] != "sha256":
        raise CheckpointFormatError("checkpoint provenance uses an unknown digest")
    if value["scope"] != _RUN_PROVENANCE_SCOPE:
        raise CheckpointFormatError("checkpoint provenance uses an unknown scope")
    _require_count(value["case_count"], "provenance case_count")
    digest = value["cases_sha256"]
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise CheckpointFormatError("checkpoint provenance SHA-256 is invalid")
    return value


def _case_identity_hash(case: dict[str, Any]) -> str:
    encoded = json.dumps(
        json_safe(_case_without_runtime(case)), sort_keys=True
    ).encode()
    return hashlib.md5(encoded).hexdigest()


def _validate_checkpoint_payload(
    checkpoint: Any, failed_sidecar: Any
) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise CheckpointFormatError("checkpoint payload must be a JSON object")
    version = checkpoint.get("checkpoint_schema_version")
    if version is None:
        raise CheckpointFormatError(
            "unversioned checkpoint; explicit migration or a new run is required"
        )
    if type(version) is not int or version != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointFormatError(
            f"unsupported checkpoint schema version {version!r}; "
            f"supported version: {CHECKPOINT_SCHEMA_VERSION}"
        )
    if set(checkpoint) != _CHECKPOINT_FIELDS:
        raise CheckpointFormatError("checkpoint payload has missing or unknown fields")

    timestamp = checkpoint["timestamp"]
    if not isinstance(timestamp, str):
        raise CheckpointFormatError("checkpoint timestamp must be an ISO string")
    try:
        datetime.fromisoformat(timestamp)
    except ValueError as exc:
        raise CheckpointFormatError("checkpoint timestamp is invalid") from exc
    _require_finite_number(checkpoint["start_time"], "start_time")

    total = _require_count(checkpoint["total_cases"], "total_cases")
    completed = _require_count(checkpoint["completed_cases"], "completed_cases")
    failed = _require_count(checkpoint["failed_cases"], "failed_cases")
    if completed + failed > total:
        raise CheckpointFormatError("checkpoint case counts are inconsistent")

    hashes = checkpoint["completed_case_hashes"]
    if not isinstance(hashes, list) or any(
        not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{32}", value) is None
        for value in hashes
    ):
        raise CheckpointFormatError("checkpoint completed-case hashes are invalid")
    if len(hashes) != len(set(hashes)) or len(hashes) != completed:
        raise CheckpointFormatError("checkpoint completed-case count is inconsistent")

    failed_data = checkpoint["failed_case_data"]
    if not isinstance(failed_data, list) or any(
        not isinstance(case, dict) for case in failed_data
    ):
        raise CheckpointFormatError(
            "checkpoint failed_case_data must be a list of cases"
        )
    if len(failed_data) != failed:
        raise CheckpointFormatError("checkpoint failed-case count is inconsistent")
    if not isinstance(failed_sidecar, list) or failed_sidecar != failed_data:
        raise CheckpointFormatError(
            "failed_cases.json does not match checkpoint failed_case_data"
        )
    failed_hashes = [_case_identity_hash(case) for case in failed_data]
    if len(failed_hashes) != len(set(failed_hashes)) or set(failed_hashes) & set(
        hashes
    ):
        raise CheckpointFormatError("checkpoint completed and failed cases conflict")

    provenance = _validate_provenance(checkpoint["run_provenance"])
    if provenance["case_count"] != total:
        raise CheckpointFormatError("checkpoint provenance case count is inconsistent")
    return checkpoint


class CheckpointManager:
    """Track completed cases and resume interrupted parameter sweeps."""

    def __init__(
        self, root_dir: Path, *, all_cases: list[dict[str, Any]] | None = None
    ):
        self.root_dir = Path(root_dir)
        self.checkpoint_file = self.root_dir / "checkpoint.json"
        self.failed_cases_file = self.root_dir / "failed_cases.json"
        self._run_provenance = (
            _run_provenance(all_cases) if all_cases is not None else None
        )
        self._declared_case_hashes = (
            frozenset(self._case_hash(case) for case in all_cases)
            if all_cases is not None
            else None
        )

    def save_checkpoint(
        self,
        completed_cases: list[dict[str, Any]],
        failed_cases: list[dict[str, Any]],
        total_cases: int,
        start_time: float,
    ) -> None:
        if self._run_provenance is None or self._declared_case_hashes is None:
            raise RuntimeError("all_cases must be bound before saving a checkpoint")
        if total_cases != self._run_provenance["case_count"]:
            raise ValueError("total_cases does not match bound all_cases")
        existing = self.load_checkpoint() if self.is_resumable() else None
        if existing is not None and existing["run_provenance"] != self._run_provenance:
            raise CheckpointFormatError(
                "checkpoint run provenance mismatch; refusing to overwrite"
            )

        unique_completed = {self._case_hash(case): case for case in completed_cases}
        completed_hashes = set(unique_completed)
        unique_failed = {
            self._case_hash(case): case
            for case in failed_cases
            if self._case_hash(case) not in completed_hashes
        }
        if (
            not completed_hashes <= self._declared_case_hashes
            or not set(unique_failed) <= self._declared_case_hashes
        ):
            raise ValueError("checkpoint cases must belong to bound all_cases")
        checkpoint_data = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "timestamp": datetime.now().isoformat(),
            "start_time": start_time,
            "total_cases": total_cases,
            "completed_cases": len(unique_completed),
            "failed_cases": len(unique_failed),
            "completed_case_hashes": list(unique_completed),
            "failed_case_data": list(unique_failed.values()),
            "run_provenance": self._run_provenance,
        }
        checkpoint_payload = json_safe(checkpoint_data)
        failed_payload = json_safe(list(unique_failed.values()))
        _validate_checkpoint_payload(checkpoint_payload, failed_payload)
        generation_dir = _create_checkpoint_generation(self.root_dir)
        atomic_write_json(
            generation_dir / self.checkpoint_file.name, checkpoint_payload
        )
        atomic_write_json(
            generation_dir / self.failed_cases_file.name,
            failed_payload,
        )
        _publish_checkpoint_generation(self.root_dir, generation_dir)
        print(f"✓ チェックポイント保存: {len(unique_completed)}/{total_cases} 完了")

    def load_checkpoint(self) -> dict[str, Any] | None:
        checkpoint_dir = resolve_checkpoint_directory(self.root_dir)
        checkpoint_file = checkpoint_dir / self.checkpoint_file.name
        failed_cases_file = checkpoint_dir / self.failed_cases_file.name
        checkpoint_exists = checkpoint_file.exists() or checkpoint_file.is_symlink()
        failed_exists = failed_cases_file.exists() or failed_cases_file.is_symlink()
        if not checkpoint_exists and not failed_exists:
            return None
        if (
            checkpoint_file.is_symlink()
            or failed_cases_file.is_symlink()
            or not checkpoint_file.is_file()
            or not failed_cases_file.is_file()
        ):
            raise CheckpointFormatError("checkpoint payload pair is missing or unsafe")
        checkpoint = _read_json(checkpoint_file, "checkpoint.json")
        failed_sidecar = _read_json(failed_cases_file, "failed_cases.json")
        return _validate_checkpoint_payload(checkpoint, failed_sidecar)

    def _case_hash(self, case: dict[str, Any]) -> str:
        return _case_identity_hash(case)

    def bind_cases(self, all_cases: list[dict[str, Any]]) -> None:
        """Bind and validate the complete ordered run before filtering or saving."""
        provenance = _run_provenance(all_cases)
        if self._run_provenance is not None and self._run_provenance != provenance:
            raise CheckpointFormatError(
                "checkpoint run provenance mismatch; all_cases changed"
            )
        checkpoint = self.load_checkpoint()
        if checkpoint is not None and checkpoint["run_provenance"] != provenance:
            raise CheckpointFormatError(
                "checkpoint run provenance mismatch; parameters changed since save"
            )
        declared_case_hashes = frozenset(self._case_hash(case) for case in all_cases)
        if checkpoint is not None:
            completed_hashes = set(checkpoint["completed_case_hashes"])
            failed_hashes = {
                self._case_hash(case) for case in checkpoint["failed_case_data"]
            }
            if not completed_hashes <= declared_case_hashes:
                raise CheckpointFormatError(
                    "checkpoint completed cases do not belong to the declared run"
                )
            if not failed_hashes <= declared_case_hashes:
                raise CheckpointFormatError(
                    "checkpoint failed cases do not belong to the declared run"
                )
        self._run_provenance = provenance
        self._declared_case_hashes = declared_case_hashes

    def filter_remaining_cases(
        self, all_cases: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        self.bind_cases(all_cases)
        checkpoint = self.load_checkpoint()
        if checkpoint is None:
            return all_cases
        completed_hashes = set(checkpoint["completed_case_hashes"])
        return [
            case for case in all_cases if self._case_hash(case) not in completed_hashes
        ]

    def is_resumable(self) -> bool:
        pointer = self.root_dir / CHECKPOINT_CURRENT_NAME
        candidates = (
            pointer,
            self.checkpoint_file,
            self.failed_cases_file,
        )
        return any(path.exists() or path.is_symlink() for path in candidates)


__all__ = [
    "CHECKPOINT_CURRENT_NAME",
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointFormatError",
    "CheckpointManager",
    "checkpoint_run_fingerprint",
    "resolve_checkpoint_directory",
]
