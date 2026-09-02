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
from collections.abc import Iterable
from pathlib import Path

import yaml

from rovibrational_excitation.optimization import ALGO_REGISTRY
from rovibrational_excitation.optimization.model import (
    OptimizationModelConfigurationError,
    build_optimization_model,
    validate_optimization_state,
)


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        raise ValueError("optimization YAML root must be a mapping")
    return loaded


def _build_states(basis, states_cfg: object) -> dict[str, tuple[int, ...]]:
    if not isinstance(states_cfg, dict):
        raise OptimizationModelConfigurationError("states must be a mapping")
    unknown = sorted(set(states_cfg) - {"initial", "target"})
    if unknown:
        raise OptimizationModelConfigurationError(
            "Unknown optimization state keys: " + ", ".join(unknown)
        )
    missing = sorted({"initial", "target"} - set(states_cfg))
    if missing:
        raise OptimizationModelConfigurationError(
            "Missing required optimization state keys: " + ", ".join(missing)
        )
    return {
        "initial": validate_optimization_state(
            basis,
            states_cfg["initial"],
            label="initial",
        ),
        "target": validate_optimization_state(
            basis,
            states_cfg["target"],
            label="target",
        ),
    }


def _format_system_label(system_cfg: dict) -> str:
    t = str(system_cfg.get("type", "")).lower()
    p = dict(system_cfg.get("params", {}))
    if t == "linmol":
        v = p.get("V_max")
        j = p.get("J_max")
        representation = p.get("representation", "unknown")
        return f"linmol_V{v}_J{j}_{representation}"
    if t == "vibladder":
        v = p.get("V_max")
        return f"viblad_V{v}"
    if t == "symtop":
        v = p.get("V_max")
        j = p.get("J_max")
        return f"symtop_V{v}_J{j}"
    if t == "twolevel":
        gap = p.get("energy_gap")
        units = p.get("energy_gap_units")
        return f"twolevel_gap{gap}{units}" if gap is not None else "twolevel"
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
    **kwargs,
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

    # 2) Build the same basis and operators as the production model layer.
    model = build_optimization_model(cfg["system"])
    basis = model.basis
    H0 = model.hamiltonian
    dipole = model.dipole

    # 3) Validate exact optimization quantum-number tuples.
    states = _build_states(basis, cfg["states"])

    # 4) Algorithm/time
    selected = cfg["algorithm"]["selected"]
    if algorithm is not None and str(algorithm).strip():
        selected = str(algorithm).strip()
    params = dict(cfg.get("algorithms", {}).get(selected, {}))
    time_cfg = dict(cfg["time"])

    runner = ALGO_REGISTRY.get(selected)
    if runner is None:
        raise ValueError(f"Unknown algorithm: {selected}")

    # 5) Output directory
    ts = time.strftime("%Y%m%d_%H%M%S")
    system_label = _format_system_label(cfg.get("system", {}))
    safe_name = "".join(c if c.isalnum() or c in "_.-" else "_" for c in selected)
    if out_dir is None:
        out_root = Path.cwd() / "results"
    else:
        out_root = Path(out_dir)
    out_path = out_root / f"{ts}_{safe_name}_{system_label}"
    os.makedirs(out_path, exist_ok=True)

    # 6) Execute
    t0 = time.time()
    # kwargsから追加パラメータを取得してparamsにマージ
    if kwargs:
        params.update(kwargs)
    result = runner(
        basis=basis,
        hamiltonian=H0,
        dipole=dipole,
        states=states,
        time_cfg=time_cfg,
        params=params,
    )
    elapsed = time.time() - t0
    print(f"{selected} elapsed: {elapsed:.2f} s")

    # 7) Optional plotting
    try:
        if do_plot:
            from rovibrational_excitation.visualization.plot_all import (
                plot_all,  # lazy import
            )

            efield_obj = result.get("efield")
            time_full = result.get("time")
            psi_traj = result.get("psi_traj")
            tlist = result.get("tlist")
            if tlist is None:
                tlist = time_full
            field_data = result.get("field_data")
            target_idx = result.get("target_idx", -1)
            if (
                efield_obj is not None
                and psi_traj is not None
                and field_data is not None
                and tlist is not None
            ):
                system_params = cfg["system"].get("params", {})
                omega_center_cm = None
                if system_params.get("vibrational_frequency_units") == "cm^-1":
                    omega_center_cm = system_params.get("vibrational_frequency")
                plot_cfg = cfg.get("plot", {})
                plot_stride_key = (
                    "sample_stride" if selected == "local" else "output_stride"
                )
                plot_all(
                    basis=basis,
                    optimizer_like=type(
                        "O",
                        (),
                        {
                            "tlist": tlist,
                            "target_idx": target_idx if target_idx is not None else -1,
                        },
                    )(),
                    efield=efield_obj,
                    psi_traj=psi_traj,
                    field_data=field_data,
                    sample_stride=int(cfg["time"].get(plot_stride_key, 1)),
                    trajectory_times_fs=time_full,
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
