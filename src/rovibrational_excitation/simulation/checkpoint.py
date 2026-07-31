"""Checkpoint persistence for parameter sweeps."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .serialization import json_safe


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
        with self.checkpoint_file.open("w", encoding="utf-8") as file:
            json.dump(json_safe(checkpoint_data), file, indent=2)
        with self.failed_cases_file.open("w", encoding="utf-8") as file:
            json.dump(json_safe(list(unique_failed.values())), file, indent=2)
        print(f"✓ チェックポイント保存: {len(unique_completed)}/{total_cases} 完了")

    def load_checkpoint(self) -> dict[str, Any] | None:
        if not self.checkpoint_file.exists():
            return None
        try:
            with self.checkpoint_file.open() as file:
                return json.load(file)
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
        return self.checkpoint_file.exists()
