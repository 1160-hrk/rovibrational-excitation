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
from copy import deepcopy
from pathlib import Path

import yaml

from rovibrational_excitation.optimization import ALGO_REGISTRY
from rovibrational_excitation.optimization.config import validate_optimization_config
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
    out_dir: str | Path | None = None,
    do_plot: bool | None = None,
) -> dict:
    """
    Execute optimization from a YAML (or dict) config.

    Returns a dict containing results and metadata:
      {
        "basis": ..., "H0": ..., "dipole": ...,
        "result": algorithm-specific arrays plus time, psi_traj, target_idx, metrics,
        "out_dir": "/abs/path/to/results",
      }
    """
    # 1) Load config
    if isinstance(config, (str, Path)):
        cfg = _load_yaml(str(config))
    elif isinstance(config, dict):
        cfg = deepcopy(config)
    else:
        raise TypeError("config must be filepath or dict")
    if do_plot is not None and not isinstance(do_plot, bool):
        raise TypeError("do_plot must be a bool or None")

    cfg = _apply_overrides(cfg, overrides)
    validated = validate_optimization_config(
        cfg,
        algorithm_override=algorithm,
    )
    plot_enabled = validated.plot_enabled if do_plot is None else do_plot
    if validated.algorithm == "krotov" and plot_enabled:
        raise ValueError(
            "standard Krotov plotting is unavailable for interval controls; "
            "disable plotting or use an explicitly supported plot adapter"
        )

    # 2) Build the same basis and operators as the production model layer.
    model = build_optimization_model(cfg["system"])
    basis = model.basis
    H0 = model.hamiltonian
    dipole = model.dipole

    # 3) Validate exact optimization quantum-number tuples.
    states = _build_states(basis, cfg["states"])

    # 4) Algorithm/time
    selected = validated.algorithm
    params = validated.algorithm_params
    time_cfg = validated.time

    runner = ALGO_REGISTRY.get(selected)
    if runner is None:
        raise ValueError(f"Unknown algorithm: {selected}")

    # 5) Output directory
    ts = time.strftime("%Y%m%d_%H%M%S")
    system_label = _format_system_label(cfg.get("system", {}))
    safe_name = "".join(c if c.isalnum() or c in "_.-" else "_" for c in selected)
    if out_dir is None:
        out_root = Path(validated.output_dir)
    else:
        out_root = Path(out_dir)
    out_path = out_root / f"{ts}_{safe_name}_{system_label}"
    os.makedirs(out_path, exist_ok=True)

    # 6) Execute
    t0 = time.time()
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

    # 7) Optional plotting. An explicitly requested plot must either succeed or raise.
    if plot_enabled:
        from rovibrational_excitation.visualization.plot_all import (
            plot_all,  # lazy import
        )

        efield_obj = result.electric_field
        time_full = result.trajectory_times_fs
        psi_traj = result.trajectory
        tlist = result.control_times_fs
        field_data = result.controls_v_per_m
        target_idx = result.target_index
        missing_plot_data = [
            name
            for name, value in (
                ("efield", efield_obj),
                ("psi_traj", psi_traj),
                ("field_data", field_data),
                ("tlist", tlist),
            )
            if value is None
        ]
        if missing_plot_data:
            raise ValueError(
                "optimization result is missing data required for plotting: "
                + ", ".join(missing_plot_data)
            )
        system_params = cfg["system"]["params"]
        omega_center_cm = None
        if system_params.get("vibrational_frequency_units") == "cm^-1":
            omega_center_cm = system_params.get("vibrational_frequency")
        plot_stride_key = "sample_stride" if selected == "local" else "output_stride"
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
            do_spectrum=validated.plot_spectrum,
            do_spectrogram=validated.plot_spectrogram,
        )

    return {
        "cfg": cfg,
        "basis": basis,
        "H0": H0,
        "dipole": dipole,
        "result": result,
        "out_dir": str(out_path),
        "elapsed_sec": elapsed,
    }
