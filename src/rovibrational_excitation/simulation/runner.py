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

import json
import shutil
import time
import traceback
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
    json_safe as _json_safe,
)
from ..io import (
    make_results_root as _make_root,
)
from ..io import (
    update_summary as _update_summary,
)
from ..models.factory import build_model_from_parameters
from ..models.linear_molecule import LinMolParameters
from ..models.validation import LinMolRepresentation
from .case import SimulationCase
from .config import (
    load_params_file as _load_params_file,
)
from .field_preparation import _generated_sampled_field
from .result_persistence import (
    persist_m_average_result,
    persist_wavefunction_result,
)
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
def _run_one_safe(
    params: dict[str, Any], max_retries: int = 2
) -> tuple[np.ndarray | None, str | None]:
    """
    1ケース実行（エラーハンドリング付き）

    Returns:
        (result, error_message): 成功時は(result, None)、失敗時は(None, error_message)
    """
    for attempt in range(max_retries + 1):
        try:
            result = _run_one(params)
            return result, None

        except Exception as e:
            error_msg = f"Attempt {attempt + 1}/{max_retries + 1} failed: {str(e)}"
            if isinstance(e, OSError) and attempt < max_retries:
                print(f"⚠ {error_msg} (再試行中...)")
                time.sleep(2**attempt)  # 指数バックオフ
            else:
                full_error = f"{error_msg}\nTraceback:\n{traceback.format_exc()}"
                print(f"✗ ケース失敗: {full_error}")

                # 失敗ケースの情報を保存
                if params.get("save", True) and "outdir" in params:
                    outdir = Path(params["outdir"])
                    outdir.mkdir(parents=True, exist_ok=True)
                    with open(outdir / "error.txt", "w", encoding="utf-8") as f:
                        f.write(full_error)
                        f.write(
                            f"\nParameters:\n{json.dumps(_json_safe(params), indent=2)}"
                        )

                return None, full_error

    # この行に到達することはないが、型チェッカーのため
    return None, "Unknown error"


def _parallel_run_safe(
    case_list: list[dict[str, Any]],
) -> list[tuple[np.ndarray | None, str | None]]:
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
    from rovibrational_excitation.core.states import PureState
    from rovibrational_excitation.dynamics.problem import PropagationProblem
    from rovibrational_excitation.dynamics.scaling.reporting import analyze_regime
    from rovibrational_excitation.dynamics.schrodinger import SchrodingerPropagator

    from .validation import _resolve_simulation_case

    validated = _resolve_simulation_case(params, field=field)
    options = validated.options
    execution_policy = options.execution
    use_m_average = (
        params["basis_type"].lower() == "linmol"
        and params["representation"] == LinMolRepresentation.M_INCOHERENT_AVERAGE.value
    )
    expects_cartesian = params["basis_type"].lower() == "symtop" or (
        params["basis_type"].lower() == "linmol"
        and params["representation"] == LinMolRepresentation.M_RESOLVED.value
    )
    if field is None:
        generated_parameters = validated.generated_field
        if generated_parameters is None:
            raise RuntimeError("validated generated field parameters are unavailable")
        time_grid = generated_parameters.time_grid
        E = _generated_sampled_field(
            params,
            generated_parameters=generated_parameters,
            use_m_average=use_m_average,
            expects_cartesian=expects_cartesian,
        )
    else:
        time_grid = field.time_grid
        E = field
    simulation_case = SimulationCase.from_validated_mapping(
        params,
        field=E,
        options=options,
    )
    time_grid = simulation_case.time_grid
    E = simulation_case.field
    t_E = time_grid.field_times_fs

    if simulation_case.uses_m_average:
        from .m_average import propagate_m_average

        model_parameters = simulation_case.model_parameters
        if not isinstance(model_parameters, LinMolParameters):
            raise RuntimeError("M-average case does not contain LinMolParameters")
        result = propagate_m_average(
            model_parameters,
            simulation_case.initial_states,
            E,
            time_grid=time_grid,
            options=simulation_case.options,
            validate_units=simulation_case.validate_units,
            verbose=simulation_case.verbose,
        )
        if params.get("save", True):
            persist_m_average_result(
                outdir=Path(params["outdir"]),
                params=params,
                field_times_fs=t_E,
                sampled_field=E,
                result=result,
            )
        return result.population

    model = build_model_from_parameters(
        simulation_case.model_parameters,
        initial_states=simulation_case.initial_states,
        representation=simulation_case.representation,
        axes=simulation_case.axes,
        execution_policy=execution_policy,
    )
    problem = PropagationProblem(
        model=model.to_system_model(),
        field=E,
        time_grid=time_grid,
        initial_state=PureState(model.state.data.ravel()),
    )
    H0 = problem.model.hamiltonian
    dip = problem.model.dipole

    use_nondimensional = options.nondimensional
    backend = options.backend_name
    algorithm_name = options.algorithm_name
    sparse = options.sparse
    split_interaction = simulation_case.split_interaction
    prop = SchrodingerPropagator(
        backend=backend,
        algorithm=algorithm_name,
        split_interaction=split_interaction,
        validate_units=simulation_case.validate_units,
        renorm=options.renorm,
        sparse=sparse,
    )
    propagation_result = prop.propagate(
        problem,
        options=options,
        verbose=simulation_case.verbose,
        split_interaction=(
            split_interaction if algorithm_name == "split_operator" else None
        ),
    )

    regime_info = None
    if use_nondimensional:
        from rovibrational_excitation.dynamics.scaling.converter import (
            nondimensionalize_from_objects,
        )

        coupling_axes = problem.coupling.axes
        *_, scales = nondimensionalize_from_objects(
            H0,
            dip,
            E,
            coupling_axes=coupling_axes,
            scalar_coupling=problem.coupling_mode == "scalar",
            verbose=False,
        )
        regime_info = analyze_regime(scales)

    host_result = propagation_result.to_numpy()
    t_p = host_result.times_fs
    psi_t = host_result.state

    pop_t = np.abs(psi_t) ** 2
    if isinstance(pop_t, np.ndarray):
        if pop_t.ndim == 0:
            pop_t = np.array([[float(pop_t)]], dtype=float)
        elif pop_t.ndim == 1:
            pop_t = pop_t.reshape(1, -1)

    if params.get("save", True):
        persist_wavefunction_result(
            outdir=Path(params["outdir"]),
            params=params,
            field_times_fs=t_E,
            sampled_field=E,
            times_fs=t_p,
            state=psi_t,
            population=pop_t,
            regime_info=regime_info,
        )

    return pop_t


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

    completed_cases = []
    failed_cases = []
    results = []
    outcomes = []

    # バッチ処理（チェックポイント間隔で分割）
    for i in range(0, len(cases), checkpoint_interval):
        batch = cases[i : i + checkpoint_interval]
        batch_num = i // checkpoint_interval + 1
        total_batches = (len(cases) + checkpoint_interval - 1) // checkpoint_interval

        print(
            f"🔄 バッチ {batch_num}/{total_batches} を実行中... ({len(batch)} ケース)"
        )

        # バッチ実行
        if nproc > 1:
            with Pool(nproc) as pool:
                batch_results = list(
                    _tqdm(
                        pool.imap(_run_one_safe, batch),
                        total=len(batch),
                        desc=f"Batch {batch_num}",
                    )
                )
        else:
            batch_results = [
                _run_one_safe(case) for case in _tqdm(batch, desc=f"Batch {batch_num}")
            ]

        # 結果を分類
        for case, (result, error) in zip(batch, batch_results):
            outcomes.append((case, result, error))
            if error is None:
                completed_cases.append(case)
                results.append(result)
            else:
                failed_case = case.copy()
                failed_case["error"] = error
                failed_cases.append(failed_case)

        # チェックポイント更新
        if batch_num % 2 == 0 or batch_num == total_batches:
            # 既存の完了ケースも含めて保存
            all_completed = []
            if checkpoint_manager is not None:
                existing_checkpoint = checkpoint_manager.load_checkpoint()
                if existing_checkpoint:
                    completed_hashes = set(
                        existing_checkpoint.get("completed_case_hashes", [])
                    )
                    for case in cases:
                        case_hash = checkpoint_manager._case_hash(case)
                        if case_hash in completed_hashes:
                            all_completed.append(case)
            all_completed.extend(completed_cases)

            if checkpoint_manager is not None:
                checkpoint_manager.save_checkpoint(
                    all_completed, failed_cases, len(cases), start_time
                )

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

    completed_cases = []
    failed_cases = []
    results = []

    # 既存の完了・失敗ケースを読み込み
    existing_checkpoint = checkpoint_manager.load_checkpoint()
    if existing_checkpoint:
        existing_failed = existing_checkpoint.get("failed_case_data", [])
        failed_cases.extend(existing_failed)

    # バッチ処理
    for i in range(0, len(remaining_cases), checkpoint_interval):
        batch = remaining_cases[i : i + checkpoint_interval]
        batch_num = i // checkpoint_interval + 1
        total_batches = (
            len(remaining_cases) + checkpoint_interval - 1
        ) // checkpoint_interval

        print(
            f"🔄 バッチ {batch_num}/{total_batches} を実行中... ({len(batch)} ケース)"
        )

        # バッチ実行
        if nproc > 1:
            with Pool(nproc) as pool:
                batch_results = list(
                    _tqdm(
                        pool.imap(_run_one_safe, batch),
                        total=len(batch),
                        desc=f"Resume Batch {batch_num}",
                    )
                )
        else:
            batch_results = [
                _run_one_safe(case)
                for case in _tqdm(batch, desc=f"Resume Batch {batch_num}")
            ]

        # 結果を分類
        for case, (result, error) in zip(batch, batch_results):
            if error is None:
                completed_cases.append(case)
                results.append(result)
            else:
                failed_case = case.copy()
                failed_case["error"] = error
                failed_cases.append(failed_case)

        # チェックポイント更新
        if batch_num % 2 == 0 or batch_num == total_batches:
            # 既存の完了ケースも含めて保存
            all_completed = []
            existing_checkpoint = checkpoint_manager.load_checkpoint()
            if existing_checkpoint:
                completed_hashes = set(
                    existing_checkpoint.get("completed_case_hashes", [])
                )
                for case in all_cases:
                    case_hash = checkpoint_manager._case_hash(case)
                    if case_hash in completed_hashes:
                        all_completed.append(case)
            all_completed.extend(completed_cases)

            checkpoint_manager.save_checkpoint(
                all_completed, failed_cases, len(all_cases), start_time
            )

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
):
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
