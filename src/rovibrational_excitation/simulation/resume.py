"""Prepare an existing simulation run for resume without changing its policy."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

from ..io import CheckpointManager
from .case_paths import materialize_sweep_cases


class ResumePreparation(NamedTuple):
    """Existing checkpoint and reconstructed cases in their current order."""

    results_dir: Path
    checkpoint_manager: CheckpointManager
    all_cases: list[dict[str, Any]]
    remaining_cases: list[dict[str, Any]]


def prepare_resume_run(
    results_dir: str | Path,
    *,
    checkpoint_manager_factory: Callable[[Path], CheckpointManager],
    load_params_file: Callable[[str], dict[str, Any]],
) -> ResumePreparation:
    """Preserve resume validation, prints, and eager case-path reconstruction."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"結果ディレクトリが見つかりません: {results_dir}")

    checkpoint_manager = checkpoint_manager_factory(results_dir)
    if not checkpoint_manager.is_resumable():
        raise ValueError(f"再開可能なチェックポイントが見つかりません: {results_dir}")

    checkpoint = checkpoint_manager.load_checkpoint()
    if checkpoint is None:
        raise ValueError("チェックポイントの読み込みに失敗")

    print(f"📁 再開: {results_dir}")
    print(
        f"🔄 前回の進捗: {checkpoint['completed_cases']}/{checkpoint['total_cases']} 完了"
    )

    params_file = results_dir / "params.py"
    if not params_file.exists():
        raise FileNotFoundError(f"パラメータファイルが見つかりません: {params_file}")

    base_dict = load_params_file(str(params_file))
    base_dict.get("description", "resumed_run")

    all_cases = materialize_sweep_cases(base_dict, root=results_dir, save=True)
    remaining_cases = checkpoint_manager.filter_remaining_cases(all_cases)
    return ResumePreparation(
        results_dir, checkpoint_manager, all_cases, remaining_cases
    )
