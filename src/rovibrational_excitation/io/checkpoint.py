"""Checkpoint persistence for parameter sweeps."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from .atomic import atomic_write_json
from .serialization import json_safe

CHECKPOINT_CURRENT_NAME = "checkpoint_current.json"
_CHECKPOINT_GENERATIONS_NAME = ".checkpoint_generations"
_CHECKPOINT_PUBLICATION_SCHEMA_VERSION = 1
_CHECKPOINT_PAYLOAD_NAMES = ("checkpoint.json", "failed_cases.json")


def _read_pointer(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_checkpoint_directory(root_dir: Path) -> Path:
    """Resolve one published checkpoint pair or the legacy direct layout."""
    root_dir = Path(root_dir)
    pointer_path = root_dir / CHECKPOINT_CURRENT_NAME
    if not pointer_path.exists() and not pointer_path.is_symlink():
        return root_dir
    if pointer_path.is_symlink():
        raise ValueError("checkpoint publication pointer must not be a symlink")
    pointer = _read_pointer(pointer_path)
    if not isinstance(pointer, dict) or set(pointer) != {
        "publication_schema_version",
        "generation",
    }:
        raise ValueError("checkpoint publication pointer has invalid fields")
    version = pointer["publication_schema_version"]
    if type(version) is not int or version != _CHECKPOINT_PUBLICATION_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported checkpoint publication schema version {version!r}"
        )
    generation = pointer["generation"]
    if (
        not isinstance(generation, str)
        or re.fullmatch(r"[0-9a-f]{32}", generation) is None
    ):
        raise ValueError("checkpoint publication generation is invalid")
    generations_root = root_dir / _CHECKPOINT_GENERATIONS_NAME
    generation_dir = generations_root / generation
    if generations_root.is_symlink() or generation_dir.is_symlink():
        raise ValueError("checkpoint publication generation is unsafe")
    if not generation_dir.is_dir():
        raise ValueError("checkpoint publication generation is missing")
    for name in _CHECKPOINT_PAYLOAD_NAMES:
        path = generation_dir / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(
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


class CheckpointManager:
    """Track completed cases and resume interrupted parameter sweeps."""

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.checkpoint_file = root_dir / "checkpoint.json"
        self.failed_cases_file = root_dir / "failed_cases.json"

    def save_checkpoint(
        self,
        completed_cases: list[dict[str, Any]],
        failed_cases: list[dict[str, Any]],
        total_cases: int,
        start_time: float,
    ) -> None:
        unique_completed = {self._case_hash(case): case for case in completed_cases}
        completed_hashes = set(unique_completed)
        unique_failed = {
            self._case_hash(case): case
            for case in failed_cases
            if self._case_hash(case) not in completed_hashes
        }
        checkpoint_data = {
            "timestamp": datetime.now().isoformat(),
            "start_time": start_time,
            "total_cases": total_cases,
            "completed_cases": len(unique_completed),
            "failed_cases": len(unique_failed),
            "completed_case_hashes": list(unique_completed),
            "failed_case_data": list(unique_failed.values()),
        }
        generation_dir = _create_checkpoint_generation(self.root_dir)
        atomic_write_json(
            generation_dir / self.checkpoint_file.name, json_safe(checkpoint_data)
        )
        atomic_write_json(
            generation_dir / self.failed_cases_file.name,
            json_safe(list(unique_failed.values())),
        )
        _publish_checkpoint_generation(self.root_dir, generation_dir)
        print(f"✓ チェックポイント保存: {len(unique_completed)}/{total_cases} 完了")

    def load_checkpoint(self) -> dict[str, Any] | None:
        try:
            checkpoint_file = (
                resolve_checkpoint_directory(self.root_dir) / self.checkpoint_file.name
            )
            if not checkpoint_file.exists():
                return None
            with checkpoint_file.open() as file:
                return cast(dict[str, Any], json.load(file))
        except Exception as exc:
            print(f"⚠ チェックポイント読み込み失敗: {exc}")
            return None

    def _case_hash(self, case: dict[str, Any]) -> str:
        case_without_runtime = {
            key: value
            for key, value in case.items()
            if key not in ["outdir", "save", "error"]
        }
        encoded = json.dumps(json_safe(case_without_runtime), sort_keys=True).encode()
        return hashlib.md5(encoded).hexdigest()

    def filter_remaining_cases(
        self, all_cases: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        checkpoint = self.load_checkpoint()
        if checkpoint is None:
            return all_cases
        completed_hashes = set(checkpoint.get("completed_case_hashes", []))
        return [
            case for case in all_cases if self._case_hash(case) not in completed_hashes
        ]

    def is_resumable(self) -> bool:
        pointer = self.root_dir / CHECKPOINT_CURRENT_NAME
        return self.checkpoint_file.exists() or pointer.exists() or pointer.is_symlink()


__all__ = [
    "CHECKPOINT_CURRENT_NAME",
    "CheckpointManager",
    "resolve_checkpoint_directory",
]
