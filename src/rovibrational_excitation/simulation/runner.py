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

import importlib.util
import itertools
import json
import os
import shutil
import time
import types
import traceback
import hashlib
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # 進捗バーが無くても動く
    tqdm = lambda x, **k: x  # type: ignore


# ---------------------------------------------------------------------
# チェックポイント管理クラス
# ---------------------------------------------------------------------
class CheckpointManager:
    """パラメータ探索の進捗を管理し、途中から再開可能にする"""
    
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.checkpoint_file = root_dir / "checkpoint.json"
        self.failed_cases_file = root_dir / "failed_cases.json"
        
    def save_checkpoint(self, 
                       completed_cases: List[Dict[str, Any]], 
                       failed_cases: List[Dict[str, Any]], 
                       total_cases: int,
                       start_time: float):
        """進捗をチェックポイントファイルに保存"""
        checkpoint_data = {
            "timestamp": datetime.now().isoformat(),
            "start_time": start_time,
            "total_cases": total_cases,
            "completed_cases": len(completed_cases),
            "failed_cases": len(failed_cases),
            "completed_case_hashes": [self._case_hash(case) for case in completed_cases],
            "failed_case_data": failed_cases,
        }
        
        with open(self.checkpoint_file, "w") as f:
            json.dump(_json_safe(checkpoint_data), f, indent=2)
            
        print(f"✓ チェックポイント保存: {len(completed_cases)}/{total_cases} 完了")
    
    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """チェックポイントファイルから進捗を読み込み"""
        if not self.checkpoint_file.exists():
            return None
            
        try:
            with open(self.checkpoint_file, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠ チェックポイント読み込み失敗: {e}")
            return None
    
    def _case_hash(self, case: Dict[str, Any]) -> str:
        """ケースのユニークハッシュを生成"""
        # outdirとsaveを除いたパラメータでハッシュ生成
        case_copy = {k: v for k, v in case.items() if k not in ["outdir", "save"]}
        case_str = json.dumps(_json_safe(case_copy), sort_keys=True)
        return hashlib.md5(case_str.encode()).hexdigest()
    
    def filter_remaining_cases(self, all_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """完了済みケースを除いた残りのケースを返す"""
        checkpoint = self.load_checkpoint()
        if checkpoint is None:
            return all_cases
            
        completed_hashes = set(checkpoint.get("completed_case_hashes", []))
        remaining_cases = []
        
        for case in all_cases:
            case_hash = self._case_hash(case)
            if case_hash not in completed_hashes:
                remaining_cases.append(case)
                
        return remaining_cases
    
    def is_resumable(self) -> bool:
        """再開可能かチェック"""
        return self.checkpoint_file.exists()


# ---------------------------------------------------------------------
# エラーハンドリング付き実行関数
# ---------------------------------------------------------------------
def _run_one_safe(params: Dict[str, Any], max_retries: int = 2) -> Tuple[Optional[np.ndarray], Optional[str]]:
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
            if attempt < max_retries:
                print(f"⚠ {error_msg} (再試行中...)")
                time.sleep(2 ** attempt)  # 指数バックオフ
            else:
                full_error = f"{error_msg}\nTraceback:\n{traceback.format_exc()}"
                print(f"✗ ケース失敗: {full_error}")
                
                # 失敗ケースの情報を保存
                if params.get("save", True):
                    outdir = Path(params["outdir"])
                    outdir.mkdir(parents=True, exist_ok=True)
                    with open(outdir / "error.txt", "w") as f:
                        f.write(full_error)
                        f.write(f"\nParameters:\n{json.dumps(_json_safe(params), indent=2)}")
                
                return None, full_error
    
    # この行に到達することはないが、型チェッカーのため
    return None, "Unknown error"


def _parallel_run_safe(case_list: List[Dict[str, Any]]) -> List[Tuple[Optional[np.ndarray], Optional[str]]]:
    """並列実行用のラッパー関数"""
    return [_run_one_safe(case) for case in case_list]


# ---------------------------------------------------------------------
# JSON変換の安全化
# 複素数に対応していないので、実部・虚部を分けて辞書化
# list, tuple, np.ndarrayなども再帰的に変換
# ---------------------------------------------------------------------
def _json_safe(obj: Any) -> Any:
    """complex / ndarray などを JSON 可能へ再帰変換"""
    if isinstance(obj, complex):
        return {"__complex__": True, "r": obj.real, "i": obj.imag}

    if callable(obj):  # 関数・メソッド・クラス等
        return f"{getattr(obj, '__module__', 'builtins')}.{getattr(obj, '__qualname__', str(obj))}"

    if isinstance(obj, types.ModuleType):
        return obj.__name__

    if isinstance(obj, np.generic):
        return obj.item()  # np.float64 → float など

    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]

    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]

    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}

    return obj  # str, int, float, bool, None などはそのまま


# ---------------------------------------------------------------------
# polarization dict ⇄ complex array
# ---------------------------------------------------------------------
def _deserialize_pol(seq) -> np.ndarray:
    def to_complex(d):
        if isinstance(d, dict):
            r = d.get("r", d.get("real", 0))
            i = d.get("i", d.get("imag", 0))
            return complex(r, i)
        elif isinstance(d, (float, int, complex)):
            return complex(d)
        else:
            raise TypeError(f"Invalid type in polarization sequence: {type(d)}")
    
    # seqが単一の値の場合
    if isinstance(seq, (int, float, complex, dict)):
        return np.array([to_complex(seq), 0], dtype=complex)
    
    # seqがiterableの場合
    if hasattr(seq, '__iter__') and not isinstance(seq, (str, bytes)):
        seq_list = list(seq)
        if len(seq_list) == 1:
            # [x] → [x, 0]
            return np.array([to_complex(seq_list[0]), 0], dtype=complex)
        elif len(seq_list) == 2:
            # [x, y] → [x, y]
            return np.array([to_complex(d) for d in seq_list], dtype=complex)
        else:
            raise ValueError(f"Polarization must have 1 or 2 elements, got {len(seq_list)}")
    
    # 何も該当しない場合はデフォルト
    return np.array([1.0, 0.0], dtype=complex)


# ---------------------------------------------------------------------
# パラメータファイル読み込み
# ---------------------------------------------------------------------
def _load_params_file(path: str) -> Dict[str, Any]:
    spec = importlib.util.spec_from_file_location("params", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[arg-type]
    return {k: getattr(mod, k) for k in dir(mod) if not k.startswith("__")}


# ---------------------------------------------------------------------
# sweep する / しない変数を分離してデータ点展開
# ---------------------------------------------------------------------
def _expand_cases(base: Dict[str, Any]):
    sweep_keys: list[str] = []
    static: Dict[str, Any] = {}

    for k, v in base.items():
        if isinstance(v, (str, bytes)):
            static[k] = v
            continue
        if hasattr(v, "__iter__"):
            try:
                if len(v) > 1:              # ★ ここが「>1」で判定
                    sweep_keys.append(k)
                    continue
            except TypeError:               # len() 不可 (ジェネレータ等)
                pass
        static[k] = v                       # 固定値に回す

    if not sweep_keys:                      # sweep 無し → 1 ケースのみ
        yield static, []
        return

    iterables = (base[k] for k in sweep_keys)
    for combo in itertools.product(*iterables):
        d = static.copy()
        d.update(dict(zip(sweep_keys, combo)))
        yield d, sweep_keys                 # sweep_keys を返す


# --- ラベル整形 -------------------------------------------
def _label(v: Any) -> str:
    if isinstance(v, (int, float)):
        return f"{v:g}"
    return str(v).replace(" ", "").replace("\n", "")

# ---------------------------------------------------------------------
# 結果ルート作成
# ---------------------------------------------------------------------
def _make_root(desc: str) -> Path:
    root = Path("results") / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{desc}"
    root.mkdir(parents=True, exist_ok=True)
    return root


# ---------------------------------------------------------------------
# 1 ケース実行
# ---------------------------------------------------------------------
def _run_one(params: Dict[str, Any]) -> np.ndarray:
    """
    1 パラメータセット実行し population(t) を返す。
    heavy import は関数内 (fork 後キャッシュされる) に移動
    """
    from rovibrational_excitation.core.basis import LinMolBasis
    from rovibrational_excitation.core.hamiltonian import generate_H0_LinMol
    from rovibrational_excitation.core.states import StateVector
    from rovibrational_excitation.core.electric_field import (
        ElectricField,
        gaussian_fwhm,
    )
    from rovibrational_excitation.core.propagator import schrodinger_propagation
    from rovibrational_excitation.dipole.linmol import LinMolDipoleMatrix
    from rovibrational_excitation.dipole.vib.morse import omega01_domega_to_N

    # ---------- Electric field -------------------------------------
    t_E = np.arange(params["t_start"], params["t_end"] + params["dt"], params["dt"])
    E = ElectricField(tlist=t_E)
    E.add_dispersed_Efield(
        envelope_func=params.get("envelope_func", gaussian_fwhm),
        duration=params["duration"],
        t_center=params["t_center"],
        carrier_freq=params["carrier_freq"],
        amplitude=params["amplitude"],
        polarization=_deserialize_pol(params["polarization"]),
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

    # ---------- Basis / initial state ------------------------------
    basis = LinMolBasis(
        params["V_max"],
        params["J_max"],
        use_M=params.get("use_M", True),
    )
    sv = StateVector(basis)
    for idx in params.get("initial_states", [0]):
        sv.set_state(basis.get_state(idx), 1)
    sv.normalize()

    # ---------- Hamiltonian & dipole -------------------------------
    delta_omega_rad_phz = params.get("delta_omega_rad_phz", 0.0)
    potential_type = params.get("potential_type", "harmonic")
    
    if delta_omega_rad_phz == 0.0:
        params.update({"potential_type": "harmonic"})

    if potential_type == 'morse':
        omega01_domega_to_N(params["omega_rad_phz"], delta_omega_rad_phz)
    
    H0 = generate_H0_LinMol(
        basis,
        omega_rad_phz=params["omega_rad_phz"],
        delta_omega_rad_phz=delta_omega_rad_phz,
        B_rad_phz=params.get("B_rad_phz", 0.0),
        alpha_rad_phz=params.get("alpha_rad_phz", 0.0),
    )
    
    dip = LinMolDipoleMatrix(
        basis,
        mu0=params["mu0_Cm"],
        potential_type=params.get("potential_type", "harmonic"),
        backend=params.get("backend", "numpy"),
        dense=params.get("dense", True),
    )

    # ---------- Propagation ----------------------------------------
    psi_t = schrodinger_propagation(
        H0,
        E,
        dip,
        sv.data,
        axes=params.get("axes", "xy"),
        return_traj=params.get("return_traj", True),
        return_time_psi=params.get("return_time_psi", True),
        backend=params.get("backend", "numpy"),
        sample_stride=params.get("sample_stride", 1),
    )
    if isinstance(psi_t, (list, tuple)) and len(psi_t) == 2:
        t_p, psi_t = psi_t
    else:
        t_p = np.array([0.0])  # dummy

    pop_t = np.abs(psi_t) ** 2  # (t, dim)

    # ---------- Save (npz 圧縮) ------------------------------------
    if params.get("save", True):
        outdir = Path(params["outdir"])
        np.savez_compressed(outdir / "result.npz", t_E=t_E, psi=psi_t, pop=pop_t, E=np.array(E.Efield), t_p=t_p)
        with open(outdir / "parameters.json", "w") as f:
            json.dump(_json_safe(params), f, indent=2)

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
    ) -> List[Any]:
    """チェックポイント機能付きのバッチ実行"""
    
    # ---------- パラメータ読み込み ---------------------------------
    if isinstance(params, str):
        base_dict = _load_params_file(params)
        description = base_dict.get("description", Path(params).stem)
        param_file_path = Path(params)
    elif isinstance(params, Mapping):
        base_dict = dict(params)
        description = base_dict.get("description", "run")
        param_file_path = None
    else:
        raise TypeError("params must be filepath str or dict-like")
    
    # ---------- ルートディレクトリ ---------------------------------
    root = _make_root(description) if save else None
    if save and root is not None and param_file_path is not None:
        shutil.copy(param_file_path, root / "params.py")

    # ---------- ケース展開 -----------------------------------------
    cases: List[Dict[str, Any]] = []
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
    
    # バッチ処理（チェックポイント間隔で分割）
    for i in range(0, len(cases), checkpoint_interval):
        batch = cases[i:i + checkpoint_interval]
        batch_num = i // checkpoint_interval + 1
        total_batches = (len(cases) + checkpoint_interval - 1) // checkpoint_interval
        
        print(f"🔄 バッチ {batch_num}/{total_batches} を実行中... ({len(batch)} ケース)")
        
        # バッチ実行
        if nproc > 1:
            with Pool(nproc) as pool:
                batch_results = list(tqdm(
                    pool.imap(_run_one_safe, batch), 
                    total=len(batch), 
                    desc=f"Batch {batch_num}"
                ))
        else:
            batch_results = [_run_one_safe(case) for case in tqdm(batch, desc=f"Batch {batch_num}")]
        
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
                completed_hashes = set(existing_checkpoint.get("completed_case_hashes", []))
                for case in cases:
                    case_hash = checkpoint_manager._case_hash(case)
                    if case_hash in completed_hashes:
                        all_completed.append(case)
            all_completed.extend(completed_cases)
            
            checkpoint_manager.save_checkpoint(
                all_completed, failed_cases, len(cases), start_time
            )
    
    # ---------- 最終結果整理 ---------------------------------------
    print(f"✅ 実行完了: {len(completed_cases)}/{len(cases)} 成功, {len(failed_cases)} 失敗")
    
    if failed_cases:
        print(f"⚠ 失敗ケース: {len(failed_cases)} 件")
        for i, failed_case in enumerate(failed_cases[:5]):  # 最初の5件のみ表示
            error_preview = failed_case.get("error", "Unknown error")[:100]
            print(f"  {i+1}. {error_preview}...")
        if len(failed_cases) > 5:
            print(f"  ... (他 {len(failed_cases) - 5} 件)")
    
    # ---------- summary.csv ----------------------------------------
    if save and root is not None:
        rows: List[Dict[str, Any]] = []
        for case, result in zip(cases, results):
            row = {k: v for k, v in case.items() if k not in ["outdir", "save"]}
            if result is not None:
                row.update({f"pop_{i}": float(p) for i, p in enumerate(result[-1])})
                row["status"] = "success"
            else:
                row["status"] = "failed"
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
    ) -> List[Any]:
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
    print(f"🔄 前回の進捗: {checkpoint['completed_cases']}/{checkpoint['total_cases']} 完了")
    
    # 元のパラメータファイルを読み込み
    params_file = results_dir / "params.py"
    if not params_file.exists():
        raise FileNotFoundError(f"パラメータファイルが見つかりません: {params_file}")
    
    base_dict = _load_params_file(str(params_file))
    description = base_dict.get("description", "resumed_run")
    
    # 全ケースを再構築
    all_cases: List[Dict[str, Any]] = []
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
        batch = remaining_cases[i:i + checkpoint_interval]
        batch_num = i // checkpoint_interval + 1
        total_batches = (len(remaining_cases) + checkpoint_interval - 1) // checkpoint_interval
        
        print(f"🔄 バッチ {batch_num}/{total_batches} を実行中... ({len(batch)} ケース)")
        
        # バッチ実行
        if nproc > 1:
            with Pool(nproc) as pool:
                batch_results = list(tqdm(
                    pool.imap(_run_one_safe, batch), 
                    total=len(batch), 
                    desc=f"Resume Batch {batch_num}"
                ))
        else:
            batch_results = [_run_one_safe(case) for case in tqdm(batch, desc=f"Resume Batch {batch_num}")]
        
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
                completed_hashes = set(existing_checkpoint.get("completed_case_hashes", []))
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


def _update_summary(results_dir: Path, all_cases: List[Dict[str, Any]]):
    """サマリーファイルを更新"""
    try:
        rows = []
        for case in all_cases:
            row = {k: v for k, v in case.items() if k not in ["outdir", "save"]}
            
            outdir = Path(case["outdir"])
            result_file = outdir / "result.npz"
            
            if result_file.exists():
                try:
                    data = np.load(result_file)
                    if "pop" in data:
                        pop_final = data["pop"][-1]
                        row.update({f"pop_{i}": float(p) for i, p in enumerate(pop_final)})
                    row["status"] = "success"
                except Exception:
                    row["status"] = "corrupted"
            else:
                row["status"] = "failed"
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(results_dir / "summary.csv", index=False)
        
        # 成功ケースのみ
        success_df = df[df["status"] == "success"]
        if not success_df.empty:
            success_df.to_csv(results_dir / "summary_success.csv", index=False)
            
        print(f"📊 サマリー更新完了: {len(success_df)}/{len(df)} 成功")
        
    except Exception as e:
        print(f"⚠ サマリー更新失敗: {e}")


# ---------------------------------------------------------------------
# 元のrun_all関数（後方互換性のため）
# ---------------------------------------------------------------------
def run_all(
    params: str | Mapping[str, Any],
    *,
    nproc: int | None = None,
    save: bool = True,
    dry_run: bool = False
    ):
    """元のrun_all関数（チェックポイント無し）"""
    return run_all_with_checkpoint(
        params, 
        nproc=nproc, 
        save=save, 
        dry_run=dry_run,
        checkpoint_interval=len(list(_expand_cases(
            _load_params_file(params) if isinstance(params, str) else dict(params)
        ))) + 1  # 全て一度に実行（チェックポイント無し）
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Run rovibrational simulation batch")
    ap.add_argument("paramfile", nargs="?", help=".py file with parameter definitions")
    ap.add_argument("-j", "--nproc", type=int, help="processes (default=1)")
    ap.add_argument("--no-save", action="store_true", help="do not write any files")
    ap.add_argument("--dry-run", action="store_true", help="list cases only (no run)")
    ap.add_argument("--resume", type=str, metavar="RESULTS_DIR", 
                    help="resume from checkpoint in specified results directory")
    ap.add_argument("--checkpoint-interval", type=int, default=10,
                    help="save checkpoint every N cases (default=10)")
    args = ap.parse_args()

    if args.resume:
        # 再開モード
        print(f"🔄 チェックポイントから再開: {args.resume}")
        t0 = time.perf_counter()
        try:
            resume_run(args.resume, nproc=args.nproc, checkpoint_interval=args.checkpoint_interval)
            print(f"✅ Resumed and finished in {time.perf_counter() - t0:.1f} s")
        except Exception as e:
            print(f"❌ 再開に失敗: {e}")
            exit(1)
    else:
        # 通常実行モード
        if not args.paramfile:
            ap.error("paramfile is required when not using --resume")
        
        t0 = time.perf_counter()
        run_all_with_checkpoint(
            args.paramfile, 
            nproc=args.nproc, 
            save=not args.no_save, 
            dry_run=args.dry_run,
            checkpoint_interval=args.checkpoint_interval
        )
        print(f"Finished in {time.perf_counter() - t0:.1f} s")
