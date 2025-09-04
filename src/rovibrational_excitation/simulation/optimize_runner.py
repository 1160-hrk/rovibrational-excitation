#!/usr/bin/env python
"""
High-level optimization runner API (package-internal).

This module provides a thin orchestration layer to:
  1) load a YAML config (or accept a dict)
  2) construct basis/H0 and dipole
  3) execute a selected optimization algorithm via ALGO_REGISTRY
  4) optionally plot and save figures

It mirrors the examples/runners/optimization_runner.py behavior,
but lives under src so it is available after pip install.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Iterable

import yaml

from rovibrational_excitation.core.basis import (
    LinMolBasis,
    VibLadderBasis,
    SymTopBasis,
    TwoLevelBasis,
)
from rovibrational_excitation.optimization import ALGO_REGISTRY
from rovibrational_excitation.dipole.factory import create_dipole_matrix


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _build_basis(system_cfg: dict):
    t = str(system_cfg["type"]).lower()
    p = dict(system_cfg.get("params", {}))
    if t == "linmol":
        return LinMolBasis(
            V_max=int(p["V_max"]),
            J_max=int(p["J_max"]),
            use_M=bool(p.get("use_M", True)),
            omega=p.get("omega_cm"),
            delta_omega=p.get("delta_omega_cm", 0.0),
            B=p.get("B_cm"),
            alpha=p.get("alpha_cm", 0.0),
            input_units=p.get("input_units", "cm^-1"),
            output_units=p.get("output_units", "rad/fs"),
        )
    if t == "viblad":
        return VibLadderBasis(
            V_max=int(p["V_max"]),
            omega=p.get("omega_cm"),
            delta_omega=p.get("delta_omega_cm", 0.0),
            input_units=p.get("input_units", "cm^-1"),
            output_units=p.get("output_units", "rad/fs"),
        )
    if t == "symtop":
        return SymTopBasis(
            V_max=int(p["V_max"]),
            J_max=int(p["J_max"]),
            omega=p.get("omega_cm"),
            B=p.get("B_cm"),
            C=p.get("C_cm"),
            alpha=p.get("alpha_cm", 0.0),
            delta_omega=p.get("delta_omega_cm", 0.0),
            input_units=p.get("input_units", "cm^-1"),
            output_units=p.get("output_units", "rad/fs"),
        )
    if t == "twolevel":
        return TwoLevelBasis(
            energy_gap=p.get("energy_gap_cm"),
            input_units=p.get("input_units", "cm^-1"),
            output_units=p.get("output_units", "rad/fs"),
        )
    raise ValueError(f"Unknown system.type: {t}")


def _build_dipole(basis, system_cfg: dict):
    p = dict(system_cfg.get("params", {}))
    mu0 = float(p.get("mu0", 1.0e-30))
    unit_dipole = str(p.get("unit_dipole", "C*m"))
    if unit_dipole not in ("C*m", "D", "ea0"):
        unit_dipole = "C*m"
    potential_type = str(p.get("potential_type", "harmonic"))
    if potential_type not in ("harmonic", "morse"):
        potential_type = "harmonic"
    return create_dipole_matrix(
        basis,
        mu0=mu0,
        potential_type=potential_type,
        backend="numpy",
        dense=False,
        units=unit_dipole,  # type: ignore[arg-type]
        units_input=unit_dipole,  # type: ignore[arg-type]
    )


def _normalize_state_for_basis(basis, state):
    if getattr(basis, "use_M", False):
        if len(state) == 2:
            return (int(state[0]), int(state[1]), 0)
        return tuple(int(x) for x in state)
    if len(state) == 3:
        return (int(state[0]), int(state[1]))
    return tuple(int(x) for x in state)


def _format_system_label(system_cfg: dict) -> str:
    t = str(system_cfg.get("type", "")).lower()
    p = dict(system_cfg.get("params", {}))
    if t == "linmol":
        v = p.get("V_max"); j = p.get("J_max"); use_m = p.get("use_M", True)
        return f"linmol_V{v}_J{j}_M{int(bool(use_m))}"
    if t == "viblad":
        v = p.get("V_max"); return f"viblad_V{v}"
    if t == "symtop":
        v = p.get("V_max"); j = p.get("J_max"); return f"symtop_V{v}_J{j}"
    if t == "twolevel":
        gap = p.get("energy_gap_cm"); return f"twolevel_gap{gap}cm" if gap is not None else "twolevel"
    return t or "system"


def _apply_overrides(cfg: dict, overrides: Iterable[str] | None) -> dict:
    if not overrides:
        return cfg
    new_cfg = cfg
    for kv in overrides:
        k, v = kv.split("=", 1)
        d = new_cfg
        ks = k.split(".")
        for kk in ks[:-1]:
            d = d.setdefault(kk, {})
        d[ks[-1]] = yaml.safe_load(v)
    return new_cfg


def run_from_config(
    config: str | Path | dict,
    algorithm: str | None = None,
    overrides: Iterable[str] | None = None,
    *,
    out_dir: str | None = None,
    do_plot: bool = True,
    **kwargs
) -> dict:
    """
    Execute optimization from a YAML (or dict) config.

    Returns a dict containing results and metadata:
      {
        "basis": ..., "H0": ..., "dipole": ...,
        "result": {efield, time, psi_traj, tlist, field_data, target_idx, metrics},
        "out_dir": "/abs/path/to/results",
      }
    """
    # 1) Load config
    if isinstance(config, (str, Path)):
        cfg = _load_yaml(str(config))
    elif isinstance(config, dict):
        cfg = dict(config)
    else:
        raise TypeError("config must be filepath or dict")

    cfg = _apply_overrides(cfg, overrides)

    # 2) Build basis and H0
    basis = _build_basis(cfg["system"])
    H0 = basis.generate_H0()

    # 3) Dipole
    dipole = _build_dipole(basis, cfg["system"])  # SI-managed dipole

    # 4) States
    initial = _normalize_state_for_basis(basis, tuple(cfg["states"]["initial"]))
    target = _normalize_state_for_basis(basis, tuple(cfg["states"]["target"]))
    states = {"initial": initial, "target": target}

    # 5) Algorithm/time
    selected = cfg["algorithm"]["selected"]
    if algorithm is not None and str(algorithm).strip():
        selected = str(algorithm).strip()
    params = dict(cfg.get("algorithms", {}).get(selected, {}))
    time_cfg = dict(cfg["time"])  # {total_fs, dt_fs, sample_stride}

    runner = ALGO_REGISTRY.get(selected)
    if runner is None:
        raise ValueError(f"Unknown algorithm: {selected}")

    # 6) Output directory
    ts = time.strftime("%Y%m%d_%H%M%S")
    system_label = _format_system_label(cfg.get("system", {}))
    safe_name = "".join(c if c.isalnum() or c in "_.-" else "_" for c in selected)
    if out_dir is None:
        out_root = Path.cwd() / "results"
    else:
        out_root = Path(out_dir)
    out_path = out_root / f"{ts}_{safe_name}_{system_label}"
    os.makedirs(out_path, exist_ok=True)

    # 7) Execute
    t0 = time.time()
    # kwargsから追加パラメータを取得してparamsにマージ
    if kwargs:
        params.update(kwargs)
    result = runner(basis=basis, hamiltonian=H0, dipole=dipole, states=states, time_cfg=time_cfg, params=params)
    elapsed = time.time() - t0
    print(f"{selected} elapsed: {elapsed:.2f} s")

    # 8) Optional plotting
    try:
        if do_plot:
            from rovibrational_excitation.plots.plot_all import plot_all  # lazy import
            efield_obj = result.get("efield")
            time_full = result.get("time")
            psi_traj = result.get("psi_traj")
            tlist = result.get("tlist") or time_full
            field_data = result.get("field_data")
            target_idx = result.get("target_idx", -1)
            if efield_obj is not None and psi_traj is not None and field_data is not None and tlist is not None:
                omega_center_cm = cfg["system"].get("params", {}).get("omega_cm")
                plot_cfg = cfg.get("plot", {})
                plot_all(
                    basis=basis,
                    optimizer_like=type("O", (), {"tlist": tlist, "target_idx": target_idx if target_idx is not None else -1})(),
                    efield=efield_obj,
                    psi_traj=psi_traj,
                    field_data=field_data,
                    sample_stride=int(cfg["time"].get("sample_stride", 1)),
                    omega_center_cm=omega_center_cm,
                    figures_dir=str(out_path),
                    filename_prefix=safe_name,
                    do_spectrum=bool(plot_cfg.get("spectrum", False)),
                    do_spectrogram=bool(plot_cfg.get("spectrogram", False)),
                )
    except Exception as e:
        print(f"Plotting failed: {e}")

    return {
        "cfg": cfg,
        "basis": basis,
        "H0": H0,
        "dipole": dipole,
        "result": result,
        "out_dir": str(out_path),
        "elapsed_sec": elapsed,
    }


