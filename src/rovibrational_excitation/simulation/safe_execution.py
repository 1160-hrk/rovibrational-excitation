"""Retry and failure reporting for one generated simulation case."""

from __future__ import annotations

import json
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from ..io import json_safe


class CaseRunOutcome(NamedTuple):
    """Tuple-compatible success or failure from one isolated batch case."""

    result: np.ndarray | None
    error: str | None


def run_case_safely(
    params: dict[str, Any],
    *,
    execute: Callable[[dict[str, Any]], np.ndarray],
    max_retries: int = 2,
) -> CaseRunOutcome:
    """Execute one case with the established OSError-only retry policy."""
    for attempt in range(max_retries + 1):
        try:
            result = execute(params)
            return CaseRunOutcome(result, None)

        except Exception as exc:
            error_msg = f"Attempt {attempt + 1}/{max_retries + 1} failed: {str(exc)}"
            if isinstance(exc, OSError) and attempt < max_retries:
                print(f"⚠ {error_msg} (再試行中...)")
                time.sleep(2**attempt)
            else:
                full_error = f"{error_msg}\nTraceback:\n{traceback.format_exc()}"
                print(f"✗ ケース失敗: {full_error}")

                if params.get("save", True) and "outdir" in params:
                    outdir = Path(params["outdir"])
                    outdir.mkdir(parents=True, exist_ok=True)
                    with open(outdir / "error.txt", "w", encoding="utf-8") as file:
                        file.write(full_error)
                        file.write(
                            f"\nParameters:\n{json.dumps(json_safe(params), indent=2)}"
                        )

                return CaseRunOutcome(None, full_error)

    return CaseRunOutcome(None, "Unknown error")


__all__ = ["CaseRunOutcome", "run_case_safely"]
