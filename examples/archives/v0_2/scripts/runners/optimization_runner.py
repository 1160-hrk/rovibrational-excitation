#!/usr/bin/env python
import os
import sys
import json
import time
from pathlib import Path
import re

import numpy as np
import yaml
from typing import Any, Iterable

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rovibrational_excitation.core.basis import LinMolBasis, VibLadderBasis, SymTopBasis, TwoLevelBasis
from rovibrational_excitation.optimization import ALGO_REGISTRY
from rovibrational_excitation.dipole import create_dipole_matrix
from utils.plotting import plot_all


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def discover_config_dir() -> Path:
    # project_root/examples/runners -> parents[2]
    return Path(__file__).resolve().parents[2] / "configs"


def list_config_candidates(conf_dir: Path) -> list[Path]:
    if not conf_dir.exists():
        return []
    # 再帰で .yaml/.yml を探索（深すぎる階層は避ける）
    files = list(conf_dir.rglob("*.yaml")) + list(conf_dir.rglob("*.yml"))
    # 表示順: ルート直下優先、その後サブディレクトリ、名前順
    files.sort(key=lambda p: (len(p.relative_to(conf_dir).parts) > 1, str(p).lower()))
    return files


def prompt_select_config(candidates: list[Path], conf_dir: Path) -> Path | None:
    # 対話は使用しない運用に変更（後方互換のため残置）。
    default_path = conf_dir / "config.yaml"
    if default_path.exists():
        return default_path
    return candidates[0] if candidates else None


def resolve_config_path(config: str | Path | None, conf_dir: Path) -> Path | None:
    """
    ユーザー指定の config があれば最優先で解決。指定がなければ None。
    - 絶対/相対パスが存在すればそれを使用
    - ファイル名（拡張子なし/あり）の場合は configs/ 配下を探索
    """
    if config is None:
        return None
    p = Path(config)
    if p.exists():
        return p
    # 拡張子候補
    candidates: list[Path] = []
    if p.suffix in (".yaml", ".yml"):
        candidates.append(conf_dir / p.name)
    else:
        candidates.append(conf_dir / f"{p.name}.yaml")
        candidates.append(conf_dir / f"{p.name}.yml")
    for c in candidates:
        if c.exists():
            return c
    return None


def build_basis(system_cfg):
    t = system_cfg["type"].lower()
    p = dict(system_cfg.get("params", {}))
    if t == "linmol":
        return LinMolBasis(
            V_max=int(p["V_max"]), J_max=int(p["J_max"]), use_M=bool(p.get("use_M", True)),
            omega=p.get("omega_cm"), delta_omega=p.get("delta_omega_cm", 0.0),
            B=p.get("B_cm"), alpha=p.get("alpha_cm", 0.0),
            input_units=p.get("input_units", "cm^-1"), output_units=p.get("output_units", "rad/fs"),
        )
    if t == "viblad":
        return VibLadderBasis(
            V_max=int(p["V_max"]),
            omega=p.get("omega_cm"), delta_omega=p.get("delta_omega_cm", 0.0),
            input_units=p.get("input_units", "cm^-1"), output_units=p.get("output_units", "rad/fs"),
        )
    if t == "symtop":
        return SymTopBasis(
            V_max=int(p["V_max"]), J_max=int(p["J_max"]),
            omega=p.get("omega_cm"), B=p.get("B_cm"), C=p.get("C_cm"),
            alpha=p.get("alpha_cm", 0.0), delta_omega=p.get("delta_omega_cm", 0.0),
            input_units=p.get("input_units", "cm^-1"), output_units=p.get("output_units", "rad/fs"),
        )
    if t == "twolevel":
        return TwoLevelBasis(
            energy_gap=p.get("energy_gap_cm"),
            input_units=p.get("input_units", "cm^-1"), output_units=p.get("output_units", "rad/fs"),
        )
    raise ValueError(f"未知のsystem.type: {t}")


def build_dipole(basis, system_type: str, system_params: dict):
    mu0 = float(system_params.get("mu0", 1.0e-30))
    unit_dipole = str(system_params.get("unit_dipole", "C*m"))
    if unit_dipole not in ("C*m", "D", "ea0"):
        unit_dipole = "C*m"
    potential_type = str(system_params.get("potential_type", "harmonic"))
    if potential_type not in ("harmonic", "morse"):
        potential_type = "harmonic"
    # create_dipole_matrix は基底型から自動判別（linmol / viblad / twolevel / symtop）
    return create_dipole_matrix(
        basis,
        mu0=mu0,
        potential_type=potential_type,  # viblad でも 'harmonic' 既定で可
        backend="numpy",
        dense=False,
        units=unit_dipole,  # type: ignore[arg-type]
        units_input=unit_dipole,  # type: ignore[arg-type]
    )


def normalize_state_for_basis(basis, state):
    if getattr(basis, "use_M", False):
        if len(state) == 2:
            return (int(state[0]), int(state[1]), 0)
        return tuple(int(x) for x in state)
    if len(state) == 3:
        return (int(state[0]), int(state[1]))
    return tuple(int(x) for x in state)


def format_system_label(system_cfg: dict) -> str:
    t = str(system_cfg.get("type", "")).lower()
    p = dict(system_cfg.get("params", {}))
    if t == "linmol":
        v = p.get("V_max")
        j = p.get("J_max")
        use_m = p.get("use_M", True)
        return f"linmol_V{v}_J{j}_M{int(bool(use_m))}"
    if t == "viblad":
        v = p.get("V_max")
        return f"viblad_V{v}"
    if t == "symtop":
        v = p.get("V_max")
        j = p.get("J_max")
        return f"symtop_V{v}_J{j}"
    if t == "twolevel":
        gap = p.get("energy_gap_cm")
        if gap is None:
            return "twolevel"
        return f"twolevel_gap{gap}cm"
    return t or "system"


def sanitize_for_path(text: str) -> str:
    # ファイル名として安全な文字に制限
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))


def main(config: str | None = None, algorithm: str | None = None, overrides: Iterable[str] | None = None):
    """
    メイン実行関数。

    初学者向け: 引数解析を使わずに main("configs/config.yaml") のように直接指定可。
    例: from examples.runners.local_optimization_runner import main; main("configs/config.yaml")
    """
    conf_dir = discover_config_dir()

    # 1) 引数 config があれば最優先で解決（argsやsys.argvは使わない）
    resolved = resolve_config_path(config, conf_dir)
    ov_list = list(overrides) if overrides is not None else []

    # 2) 未指定なら既定を選択（configs/config.yaml → 最初の候補 → エラー）
    if resolved is None:
        default_path = conf_dir / "config.yaml"
        if default_path.exists():
            resolved = default_path
        else:
            candidates = list_config_candidates(conf_dir)
            if candidates:
                resolved = candidates[0]
            else:
                raise FileNotFoundError(
                    f"設定ファイルが見つかりません: {default_path} または {conf_dir} 配下の *.yml/*.yaml"
                )

    config_path = resolved
    if not config_path.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")

    cfg = load_yaml(str(config_path))

    # override key.path=value を適用
    for kv in ov_list:
        k, v = kv.split("=", 1)
        d = cfg
        ks = k.split(".")
        for kk in ks[:-1]:
            d = d.setdefault(kk, {})
        d[ks[-1]] = yaml.safe_load(v)

    # 基底・H0
    basis = build_basis(cfg["system"])
    H0 = basis.generate_H0()

    # 双極子（linmolのみ）
    sys_type = cfg["system"]["type"].lower()
    dipole = build_dipole(basis, sys_type, cfg["system"].get("params", {}))

    # 状態
    initial = normalize_state_for_basis(basis, tuple(cfg["states"]["initial"]))
    target = normalize_state_for_basis(basis, tuple(cfg["states"]["target"]))
    states = {"initial": initial, "target": target}

    # 実行
    selected = cfg["algorithm"]["selected"]
    if algorithm is not None and str(algorithm).strip():
        selected = str(algorithm).strip()
    params = dict(cfg.get("algorithms", {}).get(selected, {}))
    time_cfg = dict(cfg["time"])

    runner = ALGO_REGISTRY.get(selected)
    if runner is None:
        raise ValueError(f"未知のアルゴリズム: {selected}")

    # 保存ディレクトリ（このファイルの1つ上の階層の results/ 以下に作成）
    ts = time.strftime("%Y%m%d_%H%M%S")
    system_label = format_system_label(cfg.get("system", {}))
    selected_safe = sanitize_for_path(selected)
    results_root = Path(__file__).resolve().parents[1] / "results"
    out_dir_path = results_root / f"{ts}_{selected_safe}_{system_label}"
    os.makedirs(out_dir_path, exist_ok=True)
    out_dir = str(out_dir_path)

    start = time.time()
    result = runner(basis=basis, hamiltonian=H0, dipole=dipole, states=states, time_cfg=time_cfg, params=params)
    elapsed = time.time() - start
    print(f"{selected} 実行時間: {elapsed:.2f} s")

    # 設定ダンプ
    with open(os.path.join(out_dir, f"run_config_{ts}.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    meta = {"timestamp": ts, "algorithm": selected, "elapsed_sec": elapsed}
    with open(os.path.join(out_dir, f"run_meta_{ts}.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # 可視化
    try:
        efield_obj = result.get("efield")
        time_full = result.get("time")
        psi_traj = result.get("psi_traj")
        tlist = result.get("tlist")
        if tlist is None:
            tlist = time_full
        field_data = result.get("field_data")
        target_idx = result.get("target_idx", -1)
        if efield_obj is not None and psi_traj is not None and field_data is not None and tlist is not None:
            try:
                print(f"[debug] tlist_len={len(tlist)}, psi_traj_shape={getattr(psi_traj, 'shape', None)}, field_data_shape={getattr(field_data, 'shape', None)}")
            except Exception:
                pass
            omega_center_cm = cfg["system"].get("params", {}).get("omega_cm")
            plot_cfg = cfg.get("plot", {})
            plot_all(
                basis=basis,
                optimizer_like=type("O", (), {"tlist": tlist, "target_idx": target_idx if target_idx >= 0 else None})(),
                efield=efield_obj,
                psi_traj=psi_traj,
                field_data=field_data,
                sample_stride=int(cfg["time"].get("sample_stride", 1)),
                omega_center_cm=omega_center_cm,
                figures_dir=out_dir,
                do_spectrum=bool(plot_cfg.get("spectrum", True)),
                do_spectrogram=bool(plot_cfg.get("spectrogram", True)),
                filename_prefix=selected_safe,
            )
    except Exception as e:
        print(f"可視化でエラー: {e}")

    # メトリクス表示
    metrics = result.get("metrics", {})
    if metrics:
        print("metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    # シンプル実行例:
    #  - python runner.py configs/config.yaml
    #  - python runner.py  (対話選択フォールバック)
    #  - main("configs/config.yaml") として他スクリプトから呼び出し
    # main("configs/config_v3j2_g3.75e19.yaml")
    # main("configs/config_v3j2_g1e20.yaml")
    # main("configs/config_v3j2m_g3.75e19.yaml")
    main("configs/config_v3j2m_krotov.yaml")


