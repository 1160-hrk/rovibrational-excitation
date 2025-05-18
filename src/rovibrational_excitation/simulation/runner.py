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

def _serialize_pol(pol: np.ndarray):
    return [{"real": c.real, "imag": c.imag} for c in pol]


def _deserialize_pol(seq):
    return np.array([complex(d["real"], d["imag"]) for d in seq])


def _make_root(desc: str) -> str:
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    root = os.path.join("results", f"{now}_{desc}")
    os.makedirs(root, exist_ok=True)
    return root


# ----------------------------------------------------------------------
# コア計算：単一ケース
# ----------------------------------------------------------------------


def _run_one(
    duration: float,
    pol_ser: List[Dict[str, float]],
    delay: float,
    outdir: str,
    p: Dict[str, Any],
) -> np.ndarray:
    """
    1 パラメータセット実行し、population(t) を return
    """
    from rovibrational_excitation.core.basis import LinMolBasis
    from rovibrational_excitation.core.hamiltonian import generate_H0_LinMol
    from rovibrational_excitation.core.states import StateVector, DensityMatrix
    from rovibrational_excitation.core.electric_field import ElectricField, gaussian_fwhm
    from rovibrational_excitation.core.propagator import schrodinger_propagation
    from linmol_dipole import LinMolDipoleMatrix

    pol = _deserialize_pol(pol_ser)

    # ---- Electric field --------------------------------------------
    time_Efield = np.arange(p["t_start"], p["t_end"] + p["dt"], p["dt"])

    Efield = ElectricField(tlist=time_Efield)
    Efield.add_Efield_disp(
        envelope_func=gaussian_fwhm,
        duration=duration,
        t_center=delay,
        carrier_freq=p["carrier_freq"],
        amplitude=p["amplitude"],
        polarization=pol,
        gdd=p["gdd"],
        tod=p["tod"],
    )
    # ---- Basis & initial state -------------------------------------
    basis = LinMolBasis(p["V_max"], p["J_max"])

    sv = StateVector(basis)
    sv.set_state((0, 0, 0), 1)
    sv.normalize()

    dm = DensityMatrix(basis)
    dm.set_pure_state(sv)

    # ---- Hamiltonian & dipole matrices -----------------------------
    H0 = generate_H0_LinMol(
        basis,
        omega_rad_phz=p["omega"],
        delta_omega_rad_phz=p["delta_omega"],
        B_rad_phz=p["B"],
        alpha_rad_phz=p["alpha"],
    )

    dip = LinMolDipoleMatrix(
        basis,
        mu0=p["mu0_Cm"],
        potential_type=p["potential_type"],
        backend=p["backend"],
        dense=p["dense"],
    )

    # ---- Propagation -----------------------------------------------
    time_psi, psi_t = schrodinger_propagation(
        H0,
        Efield,
        dip,
        sv.data,
        axes=p["axes"],                # e.g. "xy"
        return_traj=True,
        sample_stride=p["sample_stride"],
        backend=p["backend"],
    )
    pop_t = np.abs(psi_t) ** 2
    # ---- Save -------------------------------------------------------
    np.save(os.path.join(outdir, "time_psi.npy"), time_psi)
    np.save(os.path.join(outdir, "psi_trajectory.npy"), psi_t)
    np.save(os.path.join(outdir, "pop_trajectory.npy"), pop_t)
    np.save(os.path.join(outdir, "time_Efield.npy"), time_Efield)
    np.save(os.path.join(outdir, "Efield_vector.npy"), Efield.Efield)

    with open(os.path.join(outdir, "parameters.json"), "w") as f:
        json.dump(p, f, indent=2)

    return pop_t


# ----------------------------------------------------------------------
# パラメータ空間の展開
# ----------------------------------------------------------------------


def _case_paths(root: str, gw, pols, dlys):
    cases = list(itertools.product(gw, pols, dlys))
    paths = []
    for g, p, d in cases:
        rel = f"gw_{g}/pol_{p}/dly_{d}"
        out = os.path.join(root, rel)
        os.makedirs(out, exist_ok=True)
        paths.append(out)
    return cases, paths


# ----------------------------------------------------------------------
# 逐次 & 並列ラッパ
# ----------------------------------------------------------------------


def run_all(param_path: str, parallel: bool = False):
    params = _load_params(param_path)
    p_dict = _serializable_params(params)

    root = _make_root(params.description)
    shutil.copy(param_path, os.path.join(root, "params.py"))

    cases, outdirs = _case_paths(
        root, params.durations, params.polarizations, params.delays
    )
    inputs = [
        (
            gw,
            _serialize_pol(pol),
            dly,
            out,
            p_dict,
        )
        for (gw, pol, dly), out in zip(cases, outdirs)
    ]

    if parallel:
        with Pool(cpu_count()) as pool:
            results = pool.starmap(_run_one, inputs)
    else:
        results = [_run_one(*inp) for inp in inputs]

    # ---- summary CSV ------------------------------------------------
    rows = []
    for (gw, pol, dly), pop in zip(cases, results):
        rows.append(
            dict(
                gauss_width=gw,
                polarization=str(pol),
                delay=dly,
                final_population_sum=float(np.sum(pop[-1])),
            )
        )
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
