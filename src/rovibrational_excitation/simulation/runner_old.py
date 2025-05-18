"""
rovibrational_excitation/simulation/runner.py
============================================
1 ケース実行 → 保存、バッチ & 並列ラッパ

* 依存モジュール（相対 import）は **core. …** ではなく
  `linmol_dipole`, `rovibrational_excitation.core` を使用
* `LinMolDipoleMatrix` で μ 行列を lazy-build & キャッシュ
* `propagator` には axes="xy" 等で電場 ↔ μ 行列の対応を渡す
"""

from __future__ import annotations

import itertools
import importlib.util
import json
import os
import shutil
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# ヘルパ関数
# ----------------------------------------------------------------------


def _load_params(path: str):
    spec = importlib.util.spec_from_file_location("params", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[arg-type]
    return mod


def _serializable_params(mod) -> Dict[str, Any]:
    keep = (str, int, float, bool, list, dict, tuple,
            type(None), np.ndarray, np.generic)
    out: Dict[str, Any] = {}
    for k in dir(mod):
        if k.startswith("__"):
            continue
        v = getattr(mod, k)
        if isinstance(v, keep):
            try:
                out[k] = _convert(v)
            except Exception:
                pass
    return out

def _convert(obj):
        if isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}
        elif isinstance(obj, np.ndarray):
            return [_convert(x) for x in obj.tolist()]
        elif isinstance(obj, (list, tuple)):
            return [_convert(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        elif isinstance(obj, (np.generic, float, int)):
            return obj.item() if hasattr(obj, "item") else obj
        else:
            return obj


def _deserialize_pol(seq):
    out = np.zeros((len(seq), ), dtype=complex)
    for i, d in enumerate(seq):
        if isinstance(d, (int, float)):
            out[i] = d
        elif isinstance(d, dict):
            out[i] = complex(d["real"], d["imag"])
        else:
            raise ValueError(f"Invalid polarization format: {d}")
    return out


def _make_root(desc: str) -> str:
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    root = os.path.join("results", f"{now}_{desc}")
    os.makedirs(root, exist_ok=True)
    return root


# ----------------------------------------------------------------------
# コア計算：単一ケース
# ----------------------------------------------------------------------


def _run_one(params: Dict[str, Any]) -> np.ndarray:
    """
    1 パラメータセット実行し、population(t) を return
    """
    from rovibrational_excitation.core.basis import LinMolBasis
    from rovibrational_excitation.core.hamiltonian import generate_H0_LinMol
    from rovibrational_excitation.core.states import StateVector, DensityMatrix
    from rovibrational_excitation.core.electric_field import ElectricField, gaussian_fwhm
    from rovibrational_excitation.core.propagator import schrodinger_propagation
    from linmol_dipole import LinMolDipoleMatrix

    pol = _deserialize_pol(params.get("polarization", np.array([1, 0])))

    # ---- Electric field --------------------------------------------
    time_Efield = np.arange(params["t_start"], params["t_end"] + params["dt"], params["dt"])

    Efield = ElectricField(tlist=time_Efield)
    Efield.add_Efield_disp(
        envelope_func=params.get("envelope_func", gaussian_fwhm),
        duration=params["duration"],
        t_center=params["t_center"],
        carrier_freq=params["carrier_freq"],
        amplitude=params["amplitude"],
        polarization=pol,
        gdd=params.get("gdd", 0.0),
        tod=params.get("tod", 0.0),
    )
    if params.get("add_sinusoidal_mod", False):
        Efield.add_sinusoidal_mod(
            center_freq=params["carrier_freq"],
            amplitude_ratio=params["amplitude_sin_mod"],
            carrier_freq=params["carrier_freq_sin_mod"],
            phase_rad=params.get("phase_rad_sin_mod", 0.0),
            type_mod=params.get("type_sin_mod", "phase"),
        )
    # ---- Basis & initial state -------------------------------------
    basis = LinMolBasis(params["V_max"], params["J_max"], use_M=params.get("use_M", True))

    sv = StateVector(basis)
    initial_states = params.get("initial_states", [0])
    for i_s in initial_states:
        state = basis.get_state(i_s)
        sv.set_state(state, 1)
    sv.normalize()

    dm = DensityMatrix(basis)
    dm.set_pure_state(sv)

    # ---- Hamiltonian & dipole matrices -----------------------------
    H0 = generate_H0_LinMol(
        basis,
        omega_rad_phz=params["omega_rad_phz"],
        delta_omega_rad_phz=params.get("delta_omega_rad_phz", 0),
        B_rad_phz=params.get("B_rad_phz", 0),
        alpha_rad_phz=params.get("alpha_rad_phz", 0),
    )

    dip = LinMolDipoleMatrix(
        basis,
        mu0=params["mu0_Cm"],
        potential_type=params.get("potential_type", "harmonic"),
        backend=params.get("backend", "numpy"),
        dense=params.get("dense", True),
    )

    # ---- Propagation -----------------------------------------------
    return_traj = params.get("return_traj", True)
    return_time_psi = params.get("return_time_psi", True)
    result = schrodinger_propagation(
        H0,
        Efield,
        dip,
        sv.data,
        axes=params.get("axes", "xy"),                # e.g. "xy"
        return_traj=return_traj,
        return_time_psi=return_time_psi,
        sample_stride=params.get("sample_stride", 1),
        backend=params.get("backend", "numpy"),
    )
    # ---- Save -------------------------------------------------------
    outdir = params["outdir"]
    if return_time_psi:
        time_psi = result[0]
        psi_t = result[1]
        np.save(os.path.join(outdir, "time_psi.npy"), time_psi)
    else:
        psi_t = result
    if return_traj:
        np.save(os.path.join(outdir, "psi_trajectory.npy"), psi_t)
    pop_t = np.abs(psi_t) ** 2
    np.save(os.path.join(outdir, "pop_trajectory.npy"), pop_t)
    np.save(os.path.join(outdir, "time_Efield.npy"), time_Efield)
    np.save(os.path.join(outdir, "Efield_vector.npy"), Efield.Efield)

    with open(os.path.join(outdir, "parameters.json"), "w") as f:
        json.dump(params, f, indent=2)
    return pop_t


# ----------------------------------------------------------------------
# パラメータ空間の展開
# ----------------------------------------------------------------------


def _params_iter_not_iter(params: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    params_iter = {}
    params_not_iter = {}
    for k, v in params.items():
        if hasattr(v, "__iter__") and not isinstance(v, str):
            if len(v) > 0:
                params_iter[k] = v
            else:
                params_not_iter[k] = v
        else:
            params_not_iter[k] = v
    return params_iter, params_not_iter

def _case_paths(root: str, params) -> Tuple[List[Tuple], List[str]]:
    keys = params.keys()
    values = params.values()
    cases = list(itertools.product(*values))
    paths = []
    cases_dict = []
    for case in cases:
        relpath = ""
        for c, k in zip(case, keys):
            if isinstance(c, (int, float)):
                relpath = os.path.join(relpath, f"{k}_{c:.2e}")
            else:
                relpath = os.path.join(relpath, f"{k}_{c}")
        out = os.path.join(root, relpath)
        os.makedirs(out, exist_ok=True)
        paths.append(out)
        cases_dict.append(dict(zip(keys, case)))
    return cases_dict, paths



# ----------------------------------------------------------------------
# 逐次 & 並列ラッパ
# ----------------------------------------------------------------------


def run_all(param_path: str, parallel: bool = False):
    params = _load_params(param_path)
    p_dict = _serializable_params(params)

    root = _make_root(params.description)
    shutil.copy(param_path, os.path.join(root, "params.py"))
    params_iter, params_not_iter = _params_iter_not_iter(p_dict)
    cases_dict, paths = _case_paths(root, params_iter)
    for di, path in zip(cases_dict, paths):
        di.update(**params_not_iter, outdir=path)
    if parallel:
        with Pool(cpu_count()) as pool:
            results = pool.starmap(_run_one, cases_dict)
    else:
        results = [_run_one(case) for case in cases_dict]

    # ---- summary CSV ------------------------------------------------
    rows = []
    for case, pop in zip(cases_dict, results):
        for i, p in enumerate(pop[-1]):
            print(p)
            case.update(**{f"pop_{i}": float(p)})
        rows.append(case)
    pd.DataFrame(rows).to_csv(os.path.join(root, "summary.csv"), index=False)


# ----------------------------------------------------------------------
# CLI エントリポイント
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("paramfile", help="パラメータ .py ファイル")
    ap.add_argument("-P", "--parallel", action="store_true", help="並列実行 (multiprocessing)")
    args = ap.parse_args()

    t0 = time.perf_counter()
    run_all(args.paramfile, parallel=args.parallel)
    print(f"Finished in {(time.perf_counter()-t0):.1f}s")
