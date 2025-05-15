"""
simulation/runner.py  ―  修正版
-------------------------------
* 新しい propagator API に合わせて
  - `liouville_propagation` / `schrodinger_propagation`
    の引数を **dipole_matrix オブジェクト** に変更
  - `tlist`・`mu_x`・`mu_y` を渡さない
* dipole行列は `LinMolDipoleMatrix` で一括生成
* 電場クラスの API 変更に追従（`Efield.vector` → `Efield.Efield` などはそのまま）
* その他：型ヒントを追加し，コメントを整理
"""

from __future__ import annotations
import os
import shutil
import itertools
from datetime import datetime
import importlib.util
import json
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# ヘルパ
# ----------------------------------------------------------------------

def load_params(params_path: str):
    spec = importlib.util.spec_from_file_location("params", params_path)
    params = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(params)
    return params


def extract_serializable_params(params) -> Dict[str, Any]:
    """
    JSON に書ける or numpy で cast できる型だけ抽出
    """
    allowed = (str, int, float, bool, list, dict, tuple,
               type(None), np.ndarray, np.generic,
               np.float64, np.int64, np.complex128)
    out = {}
    for k in dir(params):
        if k.startswith("__"):
            continue
        v = getattr(params, k)
        if isinstance(v, allowed):
            if hasattr(v, "__iter__") and not isinstance(v, (str, bytes)):
                v = np.array(v)
            out[k] = v
    return out


def serialize_polarization(pol: np.ndarray) -> List[Dict[str, float]]:
    return [{"real": z.real, "imag": z.imag} for z in pol]


def deserialize_polarization(seq: Iterable[Dict[str, float]]) -> np.ndarray:
    return np.array([complex(p["real"], p["imag"]) for p in seq])


def make_result_root(description: str = "Sim") -> str:
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    root = os.path.join("results", f"{now}_{description}")
    os.makedirs(root, exist_ok=True)
    return root


# ----------------------------------------------------------------------
# 単一条件の実行
# ----------------------------------------------------------------------

def run_simulation_one_case(
    gauss_width: float,
    pol_serialized: Iterable[Dict[str, float]],
    delay: float,
    outdir: str,
    p: Dict[str, Any],
):
    """
    1 組み合わせだけ走らせて結果を保存
    """
    from core.basis import LinMolBasis
    from core.states import StateVector, DensityMatrix
    from core.hamiltonian import generate_free_hamiltonian
    from core.dipole_matrix import LinMolDipoleMatrix
    from core.electric_field import ElectricField
    from core.propagator import liouville_propagation  # 新 API

    print(f"Running: width={gauss_width}, delay={delay}")

    # --- 電場 ----------------------------------------------------------
    pol = deserialize_polarization(pol_serialized)
    tlist = np.linspace(p["t_start"] + delay, p["t_end"] + delay, p["t_points"])
    envelope = lambda t: np.exp(-((t - delay) / gauss_width) ** 2)

    Efield = ElectricField(
        tlist=tlist,
        envelope_func=envelope,
        carrier_freq=p["carrier_freq"],
        amplitude=p["amplitude"],
        polarization=pol,
        gdd=p["gdd"],
        tod=p["tod"],
    )

    # --- 基底 & 状態 ---------------------------------------------------
    basis = LinMolBasis(p["V_max"], p["J_max"])

    sv = StateVector(basis)
    sv.set_state(0, 0, 0)
    sv.normalize()

    dm = DensityMatrix(basis)
    dm.set_pure_state(sv)

    # --- ハミルトニアン & 双極子 ---------------------------------------
    H0 = generate_free_hamiltonian(
        basis,
        h=p["h"],
        omega=p["omega"],
        delta_omega=p["delta_omega"],
        B=p["B"],
        alpha=p["alpha"],
    )

    dipole_matrix = LinMolDipoleMatrix(basis, mu0=p["dipole_constant"])

    # --- 伝搬 -----------------------------------------------------------
    rho_t = liouville_propagation(
        H0,
        Efield,
        dm.rho,
        dipole_matrix,
        return_traj=True,
        sample_stride=p.get("sample_stride", 1),
    )

    # --- 保存 -----------------------------------------------------------
    np.save(os.path.join(outdir, "rho_t.npy"), rho_t)
    np.save(os.path.join(outdir, "tlist.npy"), tlist)
    np.save(os.path.join(outdir, "population.npy"), np.real(np.diagonal(rho_t, axis1=1, axis2=2)))
    np.save(os.path.join(outdir, "Efield_vector.npy"), Efield.Efield)

    with open(os.path.join(outdir, "parameters.json"), "w") as f:
        json.dump(p, f, indent=2, default=str)

    return np.real(np.diagonal(rho_t, axis1=1, axis2=2))


# ----------------------------------------------------------------------
# バッチ実行（逐次 & 並列）
# ----------------------------------------------------------------------

def _make_case_paths(
    result_root: str,
    gauss_widths, polarizations, delays
) -> List[Tuple]:
    cases = list(itertools.product(gauss_widths, polarizations, delays))
    paths = []
    for gw, pol, d in cases:
        rel = f"gw_{gw}/pol_{pol}/dly_{d}"
        out = os.path.join(result_root, rel)
        os.makedirs(out, exist_ok=True)
        paths.append((gw, serialize_polarization(pol), d, out))
    return cases, paths


def run_all(params_path: str):
    params = load_params(params_path)
    result_root = make_result_root(params.description)
    shutil.copy(params_path, os.path.join(result_root, "params.py"))

    cases, paths = _make_case_paths(
        result_root, params.gauss_widths, params.polarizations, params.delays
    )
    p_dict = extract_serializable_params(params)

    summary = []
    for (gw, pol, dly), (gw_, pol_s, dly_, outdir) in zip(cases, paths):
        pop = run_simulation_one_case(gw_, pol_s, dly_, outdir, p_dict)
        summary.append({
            "gauss_width": gw,
            "polarization": str(pol),
            "delay": dly,
            "final_population_sum": float(np.sum(pop[-1])),
        })

    pd.DataFrame(summary).to_csv(os.path.join(result_root, "summary.csv"), index=False)


def run_all_parallel(params_path: str):
    params = load_params(params_path)
    result_root = make_result_root(params.description)
    shutil.copy(params_path, os.path.join(result_root, "params.py"))

    cases, paths = _make_case_paths(
        result_root, params.gauss_widths, params.polarizations, params.delays
    )
    p_dict = extract_serializable_params(params)

    inputs = [(gw, pol_s, dly, outdir, p_dict) for (_, pol_s, dly, outdir), (gw, _, dly) in zip(paths, cases)]

    with Pool(processes=cpu_count()) as pool:
        results = pool.starmap(run_simulation_one_case, inputs)

    summary = []
    for (gw, pol, dly), pop in zip(cases, results):
        summary.append({
            "gauss_width": gw,
            "polarization": str(pol),
            "delay": dly,
            "final_population_sum": float(np.sum(pop[-1])),
        })

    pd.DataFrame(summary).to_csv(os.path.join(result_root, "summary.csv"), index=False)
