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
import pandas as pd

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
from .config import (
    load_params_file as _load_params_file,
)
from .execution import prepare_simulation_case, propagate_simulation_case
from .m_average import MAveragePropagationResult
from .result_persistence import (
    persist_m_average_result,
    persist_wavefunction_result,
)
from .safe_execution import CaseRunOutcome, run_case_safely
from .sweep import expand_cases as _expand_cases
from .sweep import label as _label

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
    cases: list[dict[str, Any]] = []
    for case, sweep_keys in _expand_cases(base_dict):
        case["save"] = save
        if save and root is not None:
            rel = Path(*[f"{k}_{_label(case[k])}" for k in sweep_keys])
            outdir = root / rel
            outdir.mkdir(parents=True, exist_ok=True)
            case["outdir"] = str(outdir)
        cases.append(case)

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

    # ---------- 最終結果整理 ---------------------------------------
    print(
        f"✅ 実行完了: {len(completed_cases)}/{len(cases)} 成功, {len(failed_cases)} 失敗"
    )

    if failed_cases:
        print(f"⚠ 失敗ケース: {len(failed_cases)} 件")
        for i, failed_case in enumerate(failed_cases[:5]):  # 最初の5件のみ表示
            error_preview = failed_case.get("error", "Unknown error")[:100]
            print(f"  {i + 1}. {error_preview}...")
        if len(failed_cases) > 5:
            print(f"  ... (他 {len(failed_cases) - 5} 件)")

    # ---------- summary.csv ----------------------------------------
    if save and root is not None:
        rows: list[dict[str, Any]] = []
        for case, result, error in outcomes:
            row = {k: v for k, v in case.items() if k not in ["outdir", "save"]}
            if result is not None:
                vals = result
                if isinstance(vals, np.ndarray):
                    if vals.ndim == 0:
                        vals = np.array([float(vals)])
                    elif vals.ndim == 1:
                        pass
                    else:
                        vals = vals[-1]
                else:
                    vals = [vals]
                row.update({f"pop_{i}": float(p) for i, p in enumerate(vals)})
                row["status"] = "success"
            else:
                row["status"] = "failed"
                row["error"] = error
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(root / "summary.csv", index=False)

        # 成功ケースのみのサマリー
        success_df = df[df["status"] == "success"]
        if not success_df.empty:
            success_df.to_csv(root / "summary_success.csv", index=False)

    return [r for r in results if r is not None]


def resume_run(
    results_dir: str | Path,
    *,
    nproc: int | None = None,
    checkpoint_interval: int = 10,
) -> list[Any]:
    """中断された計算を途中から再開"""

    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"結果ディレクトリが見つかりません: {results_dir}")

    checkpoint_manager = CheckpointManager(results_dir)
    if not checkpoint_manager.is_resumable():
        raise ValueError(f"再開可能なチェックポイントが見つかりません: {results_dir}")

    # チェックポイントから情報を読み込み
    checkpoint = checkpoint_manager.load_checkpoint()
    if checkpoint is None:
        raise ValueError("チェックポイントの読み込みに失敗")

    print(f"📁 再開: {results_dir}")
    print(
        f"🔄 前回の進捗: {checkpoint['completed_cases']}/{checkpoint['total_cases']} 完了"
    )

    # 元のパラメータファイルを読み込み
    params_file = results_dir / "params.py"
    if not params_file.exists():
        raise FileNotFoundError(f"パラメータファイルが見つかりません: {params_file}")

    base_dict = _load_params_file(str(params_file))
    base_dict.get("description", "resumed_run")

    # 全ケースを再構築
    all_cases: list[dict[str, Any]] = []
    for case, sweep_keys in _expand_cases(base_dict):
        case["save"] = True
        rel = Path(*[f"{k}_{_label(case[k])}" for k in sweep_keys])
        outdir = results_dir / rel
        outdir.mkdir(parents=True, exist_ok=True)
        case["outdir"] = str(outdir)
        all_cases.append(case)

    # 残りのケースをフィルタリング
    remaining_cases = checkpoint_manager.filter_remaining_cases(all_cases)

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

    print(f"✅ 再開完了: {len(completed_cases)} 新規完了, {len(failed_cases)} 失敗")

    # 最終サマリー更新
    _update_summary(results_dir, all_cases)

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
