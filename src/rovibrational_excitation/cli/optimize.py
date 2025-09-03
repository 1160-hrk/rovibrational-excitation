#!/usr/bin/env python
"""
Command line interface for running optimization.

Usage examples:
  rve-optimize --config configs/config.yaml
  rve-optimize --config my.yaml --algorithm krotov --override time.total_fs=300000 \
               --override algorithms.krotov.max_iter=100 --out ./results --no-plot
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from rovibrational_excitation.simulation.optimize_runner import run_from_config


def main():
    ap = argparse.ArgumentParser(description="Run rovibrational optimization")
    ap.add_argument("--config", required=True, help="YAML config path")
    ap.add_argument("--algorithm", default=None, help="Override selected algorithm")
    ap.add_argument("--override", action="append", default=None, help="key.path=value, repeatable")
    ap.add_argument("--out", default=None, help="Output directory (default: ./results)")
    ap.add_argument("--no-plot", action="store_true", help="Disable plotting")
    args = ap.parse_args()

    cfg_path = args.config
    if not os.path.exists(cfg_path):
        ap.error(f"Config not found: {cfg_path}")

    res = run_from_config(
        cfg_path,
        algorithm=args.algorithm,
        overrides=args.override,
        out_dir=args.out,
        do_plot=(not args.no_plot),
    )

    out_dir = res.get("out_dir")
    print(f"Saved outputs under: {out_dir}")


if __name__ == "__main__":
    main()


