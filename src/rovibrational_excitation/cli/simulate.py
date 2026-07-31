"""Command-line interface for batch simulations."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence

from rovibrational_excitation.simulation.runner import (
    resume_run,
    run_all_with_checkpoint,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run rovibrational simulation batch")
    parser.add_argument("paramfile", nargs="?", help="Python parameter file")
    parser.add_argument("-j", "--nproc", type=int, help="number of worker processes")
    parser.add_argument("--no-save", action="store_true", help="do not write files")
    parser.add_argument("--dry-run", action="store_true", help="list cases only")
    parser.add_argument(
        "--resume",
        metavar="RESULTS_DIR",
        help="resume from a checkpoint in the specified results directory",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="save a checkpoint every N cases (default: 10)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.resume:
        started = time.perf_counter()
        try:
            resume_run(
                args.resume,
                nproc=args.nproc,
                checkpoint_interval=args.checkpoint_interval,
            )
        except Exception as exc:
            parser.exit(1, f"resume failed: {exc}\n")
        elapsed = time.perf_counter() - started
        print(f"Resumed and finished in {elapsed:.1f} s")
        return 0

    if not args.paramfile:
        parser.error("paramfile is required when --resume is not used")

    started = time.perf_counter()
    run_all_with_checkpoint(
        args.paramfile,
        nproc=args.nproc,
        save=not args.no_save,
        dry_run=args.dry_run,
        checkpoint_interval=args.checkpoint_interval,
    )
    print(f"Finished in {time.perf_counter() - started:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
