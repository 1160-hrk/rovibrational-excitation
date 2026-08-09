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

from .checkpoint import CheckpointManager
from .config import (
    load_params_file as _load_params_file,
)
from .config import (
    process_params as _process_params,
)
from .models import build_model
from .serialization import (
    deserialize_polarization as _deserialize_pol,
)
from .serialization import (
    json_safe as _json_safe,
)
from .storage import (
    make_results_root as _make_root,
)
from .storage import (
    update_summary as _update_summary,
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
def _run_one(params: dict[str, Any]) -> np.ndarray:
    """
    1 パラメータセット実行し population(t) を返す。
    系タイプ（basis_type）に応じて汎用的に対応
    """
    # --- 必要なimportは関数内で ---
    from rovibrational_excitation.core.electric_field import (
        ElectricField,
        gaussian_fwhm,
    )
    from rovibrational_excitation.core.nondimensional.analysis import analyze_regime
    from rovibrational_excitation.core.propagation.schrodinger import (
        SchrodingerPropagator,
    )
    from rovibrational_excitation.core.propagation.utils import validate_axes

    from .timegrid import build_time_grid
    from .validation import validate_simulation_case

    # --- Electric field 共通 ---
    validate_simulation_case(params)
    polarization = _deserialize_pol(params["polarization"])
    use_m_average = params.get(
        "basis_type", "linmol"
    ).lower() == "linmol" and not params.get("use_M", True)
    if use_m_average:
        from .models.linmol_m_average import (
            canonicalize_fixed_linear_polarization,
        )

        polarization = canonicalize_fixed_linear_polarization(polarization)
    t_E = build_time_grid(params["t_start"], params["t_end"], params["dt"])
    E = ElectricField(tlist=t_E)
    E.add_dispersed_Efield(
        envelope_func=params.get("envelope_func", gaussian_fwhm),
        duration=params.get(
            "duration",
            params.get(
                "pulse_duration",
                (params["t_end"] - params["t_start"]) / 2,
            ),
        ),
        t_center=params.get("t_center", 0.0),
        carrier_freq=params["carrier_freq"],
        amplitude=params["amplitude"],
        polarization=polarization,
        phase_rad=params.get("phase_rad", 0.0),
        gdd=params.get("gdd", 0.0),
        tod=params.get("tod", 0.0),
    )
    if params.get("Sinusoidal_modulation", False):
        E.apply_sinusoidal_mod(
            center_freq=params["carrier_freq"],
            amplitude=params["amplitude_sin_mod"],
            carrier_freq=params["carrier_freq_sin_mod"],
            phase_rad=params.get("phase_rad_sin_mod", 0.0),
            type_mod=params.get("type_mod_sin_mod", "phase"),
        )

    if use_m_average:
        from .models.linmol_m_average import propagate_m_average

        result = propagate_m_average(params, E)
        if params.get("save", True):
            outdir = Path(params["outdir"])
            save_data: dict[str, Any] = {
                "t_E": t_E,
                "pop": result.population,
                "E": np.array(E.Efield),
                "t_p": result.time_fs,
                "representation": np.array("m_incoherent_average"),
                "abs_m": np.array([block.abs_m for block in result.blocks]),
                "m_multiplicity": np.array(
                    [block.multiplicity for block in result.blocks]
                ),
                "m_weight": np.array([block.weight for block in result.blocks]),
            }
            for block, wavefunction in zip(result.blocks, result.block_wavefunctions):
                save_data[f"psi_abs_m_{block.abs_m}"] = wavefunction
            np.savez_compressed(outdir / "result.npz", **save_data)
            with open(outdir / "parameters.json", "w") as f:
                json.dump(_json_safe(params), f, indent=2)
        return result.population

    # --- 系タイプ別の構築 ---
    model = build_model(params)
    sv = model.state
    H0 = model.hamiltonian
    dip = model.dipole

    # ---------- Propagation 共通 ----------
    use_nondimensional = params.get("nondimensional", False)
    backend = params.get("backend", "numpy")
    algorithm = params.get("algorithm", "rk4")
    sparse = params.get("sparse", not params.get("dense", True))
    prop = SchrodingerPropagator(
        backend=backend,
        algorithm=algorithm,
        split_interaction=params.get("split_interaction", "cartesian"),
        validate_units=params.get("validate_units", True),
        renorm=params.get("renorm", False),
        sparse=sparse,
    )
    psi_t = prop.propagate(
        hamiltonian=H0,
        efield=E,
        dipole_matrix=dip,
        initial_state=sv.data,
        coupling_mode=model.coupling.mode,
        **(
            {"axes": params.get("axes", model.coupling.default_axes)}
            if model.coupling.mode == "cartesian"
            else {"coupling_axis": model.coupling.axis}
        ),
        return_traj=params.get("return_traj", True),
        return_time_psi=True,
        sample_stride=params.get("sample_stride", 1),
        nondimensional=use_nondimensional,
        verbose=params.get("verbose", False),
        algorithm=algorithm,
        **(
            {"split_interaction": params.get("split_interaction", "cartesian")}
            if algorithm == "split_operator"
            else {}
        ),
        sparse=sparse,
        renorm=params.get("renorm", False),
    )

    # 無次元化使用時は物理レジーム情報も保存
    regime_info = None
    if use_nondimensional:
        from rovibrational_excitation.core.nondimensional.converter import (
            nondimensionalize_from_objects,
        )

        if model.coupling.mode == "scalar":
            axis = model.coupling.axis
            assert axis is not None
            coupling_axes = (axis,)
        else:
            coupling_axes = validate_axes(params.get("axes", "xy"))
        *_, scales = nondimensionalize_from_objects(
            H0,
            dip,
            E,
            coupling_axes=coupling_axes,
            scalar_coupling=model.coupling.mode == "scalar",
            verbose=False,
        )
        regime_info = analyze_regime(scales)

    if isinstance(psi_t, tuple) and len(psi_t) == 2:
        t_p, psi_t = psi_t
    else:
        raise RuntimeError("propagator did not return the requested physical time grid")

    pop_t = np.abs(psi_t) ** 2  # ideally (t, dim)
    # Ensure shape is always (t, dim)
    if isinstance(pop_t, np.ndarray):
        if pop_t.ndim == 0:
            pop_t = np.array([[float(pop_t)]], dtype=float)
        elif pop_t.ndim == 1:
            pop_t = pop_t.reshape(1, -1)

    # ---------- Save (npz 圧縮) 共通 ----------
    if params.get("save", True):
        outdir = Path(params["outdir"])
        save_data = {
            "t_E": t_E,
            "psi": psi_t,
            "pop": pop_t,
            "E": np.array(E.Efield),
            "t_p": t_p,
        }
        if regime_info is not None:
            save_data["regime_info"] = regime_info
        np.savez_compressed(outdir / "result.npz", **save_data)
        with open(outdir / "parameters.json", "w") as f:
            json.dump(_json_safe(params), f, indent=2)
        if regime_info is not None:
            with open(outdir / "regime_analysis.json", "w") as f:
                json.dump(_json_safe(regime_info), f, indent=2)

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
        raw_dict = dict(params)

        # Apply default units and automatic conversion
        # デフォルト単位を適用（後方互換性のため一時的にスキップ）
        dict_with_defaults = raw_dict.copy()
        base_dict = _process_params(dict_with_defaults)

        if raw_dict != base_dict:
            print("📋 Unit processing completed.")
        else:
            print("📋 No unit processing needed.")
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
