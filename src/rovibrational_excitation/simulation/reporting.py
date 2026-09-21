"""Normal batch completion reporting and in-memory CSV summaries."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def report_normal_batch(
    *,
    total_cases: int,
    completed_cases: list[dict[str, Any]],
    failed_cases: list[dict[str, Any]],
    outcomes: list[tuple[dict[str, Any], Any, str | None]],
    save: bool,
    root: Path | None,
) -> None:
    """Preserve the existing normal-run preview and returned-result summary."""
    print(
        f"✅ 実行完了: {len(completed_cases)}/{total_cases} 成功, {len(failed_cases)} 失敗"
    )

    if failed_cases:
        print(f"⚠ 失敗ケース: {len(failed_cases)} 件")
        for i, failed_case in enumerate(failed_cases[:5]):  # 最初の5件のみ表示
            error_preview = failed_case.get("error", "Unknown error")[:100]
            print(f"  {i + 1}. {error_preview}...")
        if len(failed_cases) > 5:
            print(f"  ... (他 {len(failed_cases) - 5} 件)")

    if save and root is not None:
        rows: list[dict[str, Any]] = []
        for case, result, error in outcomes:
            row = {k: v for k, v in case.items() if k not in ["outdir", "save"]}
            if result is not None:
                vals = result
                if isinstance(vals, np.ndarray):
                    if vals.ndim == 0:
                        vals = np.array([float(vals)])
                    elif vals.ndim == 1:
                        pass
                    else:
                        vals = vals[-1]
                else:
                    vals = [vals]
                row.update({f"pop_{i}": float(p) for i, p in enumerate(vals)})
                row["status"] = "success"
            else:
                row["status"] = "failed"
                row["error"] = error
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(root / "summary.csv", index=False)

        # 成功ケースのみのサマリー
        success_df = df[df["status"] == "success"]
        if not success_df.empty:
            success_df.to_csv(root / "summary_success.csv", index=False)


def report_resumed_batch(
    *,
    results_dir: Path,
    all_cases: list[dict[str, Any]],
    completed_cases: list[dict[str, Any]],
    failed_cases: list[dict[str, Any]],
    update_summary: Callable[[Path, list[dict[str, Any]]], None],
) -> None:
    """Report newly completed cases, then rebuild the file-backed summary."""
    print(f"✅ 再開完了: {len(completed_cases)} 新規完了, {len(failed_cases)} 失敗")
    update_summary(results_dir, all_cases)
