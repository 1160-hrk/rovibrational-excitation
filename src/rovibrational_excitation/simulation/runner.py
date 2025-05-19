"""
rovibrational_excitation/simulation/runner.py
============================================
・パラメータ sweep → 逐次／並列実行
・結果を results/<timestamp>_<desc>/… に保存
・JSON 変換安全化／進捗バー／npz 圧縮など改善

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
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # 進捗バーが無くても動く
    tqdm = lambda x, **k: x  # type: ignore


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
def _deserialize_pol(seq: list[dict | float | int | complex]) -> np.ndarray:
    return np.asarray(
        [
            complex(d["r"], d["i"]) if isinstance(d, dict) and d.get("__complex__")
            else complex(d)
            for d in seq
        ],
        dtype=complex,
    )


# ---------------------------------------------------------------------
# パラメータファイル読み込み
# ---------------------------------------------------------------------
def _load_params(path: str):
    spec = importlib.util.spec_from_file_location("params", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[arg-type]
    return mod


# ---------------------------------------------------------------------
# sweep する / しない変数を分離してデータ点展開
# ---------------------------------------------------------------------
def _expand_cases(base: Dict[str, Any]):
    """list, tuple, ndarray など *イテラブル* を sweep 対象とみなす"""
    sweep_keys = [k for k, v in base.items()
                  if hasattr(v, "__iter__") and not isinstance(v, (str, bytes))]
    static = {k: v for k, v in base.items() if k not in sweep_keys}

    iterables = (base[k] for k in sweep_keys)
    for combo in itertools.product(*iterables):
        d = static.copy()
        d.update(dict(zip(sweep_keys, combo)))
        yield d, sweep_keys   # sweep_keys を戻しておく

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
    from linmol_dipole import LinMolDipoleMatrix

    # ---------- Electric field -------------------------------------
    t_E = np.arange(params["t_start"], params["t_end"] + params["dt"], params["dt"])
    E = ElectricField(tlist=t_E)
    E.add_Efield_disp(
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
        E.add_sinusoidal_mod(
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
    H0 = generate_H0_LinMol(
        basis,
        omega_rad_phz=params["omega_rad_phz"],
        delta_omega_rad_phz=params.get("delta_omega_rad_phz", 0.0),
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
        np.savez_compressed(outdir / "result.npz", t_E=t_E, psi=psi_t, pop=pop_t, E=E.Efield, t_p=t_p)
        with open(outdir / "parameters.json", "w") as f:
            json.dump(_json_safe(params), f, indent=2)

    return pop_t


# ---------------------------------------------------------------------
# メイン：全ケース実行
# ---------------------------------------------------------------------
def run_all(param_path: str, *, nproc: int | None = None, save: bool = True, dry_run: bool = False):
    mod = _load_params(param_path)
    base_dict = {k: getattr(mod, k) for k in dir(mod) if not k.startswith("__")}
    
    if save:
        root = _make_root(mod.description)
        shutil.copy(param_path, root / "params.py")

    # --------------------------------------------------------------
    # ケース展開 & 出力ディレクトリ決定
    # --------------------------------------------------------------
    cases: List[Dict[str, Any]] = []
    for case, sweep_keys in _expand_cases(base_dict):
        case["save"] = save
        if save:
            rel = Path(*[f"{k}_{_label(case[k])}" for k in sweep_keys])
            outdir = root / rel
            outdir.mkdir(parents=True, exist_ok=True)
            case["outdir"] = str(outdir)
        cases.append(case)

    if dry_run:
        print(f"[Dry-run] would execute {len(cases)} cases")
        return

    # --------------------------------------------------------------
    # 実行
    # --------------------------------------------------------------
    nproc = min(cpu_count(), nproc or 1)
    runner = Pool(nproc).map if nproc > 1 else map
    results = list(tqdm(runner(_run_one, cases), total=len(cases), desc="Cases"))

    # --------------------------------------------------------------
    # summary.csv
    # --------------------------------------------------------------
    if save:
        rows: List[Dict[str, Any]] = []
        for case, pop in zip(cases, results):
            row = {k: v for k, v in case.items() if k != "outdir"}
            row.update({f"pop_{i}": float(p) for i, p in enumerate(pop[-1])})
            rows.append(row)
        pd.DataFrame(rows).to_csv(root / "summary.csv", index=False)
    return results


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Run rovibrational simulation batch")
    ap.add_argument("paramfile", help=".py file with parameter definitions")
    ap.add_argument("-j", "--nproc", type=int, help="processes (default=1)")
    ap.add_argument("--no-save", action="store_true", help="do not write any files")
    ap.add_argument("--dry-run", action="store_true", help="list cases only (no run)")
    args = ap.parse_args()

    t0 = time.perf_counter()
    run_all(args.paramfile, nproc=args.nproc, save=not args.no_save, dry_run=args.dry_run)
    print(f"Finished in {time.perf_counter() - t0:.1f} s")
