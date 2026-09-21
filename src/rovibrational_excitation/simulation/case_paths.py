"""Materialize expanded simulation cases and their existing result paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .sweep import expand_cases, label


def materialize_sweep_cases(
    base: dict[str, Any],
    *,
    root: Path | None,
    save: bool,
) -> list[dict[str, Any]]:
    """Preserve sweep order, labels, and eager directory creation."""
    cases: list[dict[str, Any]] = []
    for case, sweep_keys in expand_cases(base):
        case["save"] = save
        if save and root is not None:
            rel = Path(*[f"{key}_{label(case[key])}" for key in sweep_keys])
            outdir = root / rel
            outdir.mkdir(parents=True, exist_ok=True)
            case["outdir"] = str(outdir)
        cases.append(case)
    return cases
