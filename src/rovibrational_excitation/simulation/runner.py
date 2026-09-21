"""
rovibrational_excitation/simulation/runner.py
============================================
・パラメータ sweep → 逐次／並列実行
・結果を results/<timestamp>_<desc>/… に保存
・JSON 変換安全化／進捗バー／npz 圧縮など改善
・チェックポイント・復旧機能追加

依存：
    numpy, pandas, (tqdm は任意)
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Mapping
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import numpy as np

from ..fields import SampledField
from ..io import (
    CheckpointManager,
)
from ..io import (
    make_results_root as _make_root,
)
from ..io import (
    update_summary as _update_summary,
)
from .batch import execute_case_batches
from .case_paths import materialize_sweep_cases
from .config import (
    load_params_file as _load_params_file,
)
from .execution import prepare_simulation_case, propagate_simulation_case
from .m_average import MAveragePropagationResult
from .reporting import report_normal_batch, report_resumed_batch
from .result_persistence import (
    persist_m_average_result,
    persist_wavefunction_result,
)
from .resume import prepare_resume_run
from .safe_execution import CaseRunOutcome, run_case_safely
from .sweep import expand_cases as _expand_cases

try:
    from tqdm import tqdm as _tqdm_impl

    def _tqdm(x, **k):  # type: ignore
        return _tqdm_impl(x, **k)
except ImportError:  # 進捗バーが無くても動く

    def _tqdm(x, **k):  # type: ignore
        return x


# ---------------------------------------------------------------------
# エラーハンドリング付き実行関数
# ---------------------------------------------------------------------
def _run_one_safe(params: dict[str, Any], max_retries: int = 2) -> CaseRunOutcome:
    """
    1ケース実行（エラーハンドリング付き）

    Returns:
        (result, error_message): 成功時は(result, None)、失敗時は(None, error_message)
    """
    return run_case_safely(params, execute=_run_one, max_retries=max_retries)


def _parallel_run_safe(
    case_list: list[dict[str, Any]],
) -> list[CaseRunOutcome]:
    """並列実行用のラッパー関数"""
    return [_run_one_safe(case) for case in case_list]


# ---------------------------------------------------------------------
# 1 ケース実行
# ---------------------------------------------------------------------
def run_simulation_case(
    params: Mapping[str, Any],
    *,
    field: SampledField,
) -> np.ndarray:
    """Run one case with an explicitly sampled scalar or Cartesian field."""
    if not isinstance(params, Mapping):
        raise TypeError("params must be a mapping")
    return _execute_one(dict(params), field=field)


def _run_one(params: dict[str, Any]) -> np.ndarray:
    """Run one generated-field parameter set and return population(t)."""
    return _execute_one(params, field=None)


def _execute_one(params: dict[str, Any], *, field: SampledField | None) -> np.ndarray:
    """Execute one validated generated-field or sampled-field case."""
    simulation_case = prepare_simulation_case(params, field=field)
    field_times_fs = simulation_case.time_grid.field_times_fs
    sampled_field = simulation_case.field
    result = propagate_simulation_case(simulation_case)

    if isinstance(result, MAveragePropagationResult):
        if params.get("save", True):
            persist_m_average_result(
                outdir=Path(params["outdir"]),
                params=params,
                field_times_fs=field_times_fs,
                sampled_field=sampled_field,
                result=result,
            )
        return result.population

    if params.get("save", True):
        persist_wavefunction_result(
            outdir=Path(params["outdir"]),
            params=params,
            field_times_fs=field_times_fs,
            sampled_field=sampled_field,
            times_fs=result.propagation.times_fs,
            state=result.propagation.state,
            population=result.population,
            regime_info=result.regime_info,
        )

    return result.population


# ---------------------------------------------------------------------
# チェックポイント付きバッチ実行
# ---------------------------------------------------------------------
def run_all_with_checkpoint(
    params: str | Mapping[str, Any],
    *,
    nproc: int | None = None,
    save: bool = True,
    dry_run: bool = False,
    checkpoint_interval: int = 10,
) -> list[Any]:
    """チェックポイント機能付きのバッチ実行"""
    if not isinstance(checkpoint_interval, int) or checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be a positive integer")

    # ---------- パラメータ読み込み ---------------------------------
    if isinstance(params, str):
        base_dict = _load_params_file(params)
        description = base_dict.get("description", Path(params).stem)
        param_file_path = Path(params)
    elif isinstance(params, Mapping):
        print("📊 Loading parameters from dict")
        base_dict = dict(params)
        print("📋 Values and explicit unit labels loaded unchanged.")
        description = base_dict.get("description", "run")
        param_file_path = None
    else:
        raise TypeError("params must be filepath str or dict-like")

    # ---------- ルートディレクトリ ---------------------------------
    root = _make_root(description) if save else None
    if save and root is not None and param_file_path is not None:
        shutil.copy(param_file_path, root / "params.py")

    # ---------- ケース展開 -----------------------------------------
    cases = materialize_sweep_cases(base_dict, root=root, save=save)

    if dry_run:
        print(f"[Dry-run] would execute {len(cases)} cases")
        return []

    # ---------- チェックポイント管理 -------------------------------
    checkpoint_manager = CheckpointManager(root) if save and root else None

    # ---------- 実行 -----------------------------------------------
    start_time = time.perf_counter()
    nproc = min(cpu_count(), nproc or 1)

    print(f"📊 実行開始: {len(cases)} ケース、{nproc} プロセス")

    batch_run = execute_case_batches(
        cases,
        all_cases=cases,
        checkpoint_manager=checkpoint_manager,
        checkpoint_interval=checkpoint_interval,
        nproc=nproc,
        start_time=start_time,
        execute_case=_run_one_safe,
        progress=_tqdm,
        pool_factory=Pool,
        progress_label="Batch",
    )
    completed_cases = batch_run.completed_cases
    failed_cases = batch_run.failed_cases
    results = batch_run.results
    outcomes = batch_run.outcomes

    report_normal_batch(
        total_cases=len(cases),
        completed_cases=completed_cases,
        failed_cases=failed_cases,
        outcomes=outcomes,
        save=save,
        root=root,
    )

    return [r for r in results if r is not None]


def resume_run(
    results_dir: str | Path,
    *,
    nproc: int | None = None,
    checkpoint_interval: int = 10,
) -> list[Any]:
    """中断された計算を途中から再開"""

    resume_preparation = prepare_resume_run(
        results_dir,
        checkpoint_manager_factory=CheckpointManager,
        load_params_file=_load_params_file,
    )
    results_dir = resume_preparation.results_dir
    checkpoint_manager = resume_preparation.checkpoint_manager
    all_cases = resume_preparation.all_cases
    remaining_cases = resume_preparation.remaining_cases

    if not remaining_cases:
        print("✅ 全ケースが既に完了しています")
        return []

    print(f"🔄 残り {len(remaining_cases)} ケースを実行中...")

    # 残りケースを実行
    start_time = time.perf_counter()
    nproc = min(cpu_count(), nproc or 1)

    existing_checkpoint = checkpoint_manager.load_checkpoint()
    existing_failed = (
        existing_checkpoint.get("failed_case_data", []) if existing_checkpoint else []
    )
    batch_run = execute_case_batches(
        remaining_cases,
        all_cases=all_cases,
        checkpoint_manager=checkpoint_manager,
        checkpoint_interval=checkpoint_interval,
        nproc=nproc,
        start_time=start_time,
        execute_case=_run_one_safe,
        progress=_tqdm,
        pool_factory=Pool,
        progress_label="Resume Batch",
        initial_failed_cases=existing_failed,
    )
    completed_cases = batch_run.completed_cases
    failed_cases = batch_run.failed_cases
    results = batch_run.results

    report_resumed_batch(
        results_dir=results_dir,
        all_cases=all_cases,
        completed_cases=completed_cases,
        failed_cases=failed_cases,
        update_summary=_update_summary,
    )

    return results


# ---------------------------------------------------------------------
# 元のrun_all関数（後方互換性のため）
# ---------------------------------------------------------------------
def run_all(
    params: str | Mapping[str, Any],
    *,
    nproc: int | None = None,
    save: bool = True,
    dry_run: bool = False,
) -> list[Any]:
    """元のrun_all関数（チェックポイント無し）"""
    return run_all_with_checkpoint(
        params,
        nproc=nproc,
        save=save,
        dry_run=dry_run,
        checkpoint_interval=len(
            list(
                _expand_cases(
                    _load_params_file(params)
                    if isinstance(params, str)
                    else dict(params)
                )
            )
        )
        + 1,  # 全て一度に実行（チェックポイント無し）
    )
