"""Result-directory and summary-file persistence."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def make_results_root(description: str) -> Path:
    """Create the timestamped root directory used by batch runs."""
    root = Path("results") / f"{datetime.now():%Y%m%d_%H%M%S}_{description}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def update_summary(results_dir: Path, all_cases: list[dict[str, Any]]) -> None:
    """Update CSV summaries from result files without rerunning simulations."""
    try:
        rows = []
        for case in all_cases:
            row = {
                key: value
                for key, value in case.items()
                if key not in ["outdir", "save"]
            }
            result_file = Path(case["outdir"]) / "result.npz"

            if result_file.exists():
                try:
                    data = np.load(result_file)
                    if "pop" in data:
                        pop_final = data["pop"][-1]
                        row.update(
                            {
                                f"pop_{index}": float(population)
                                for index, population in enumerate(pop_final)
                            }
                        )
                    row["status"] = "success"
                except Exception:
                    row["status"] = "corrupted"
            else:
                row["status"] = "failed"
            rows.append(row)

        dataframe = pd.DataFrame(rows)
        dataframe.to_csv(results_dir / "summary.csv", index=False)
        successful = dataframe[dataframe["status"] == "success"]
        if not successful.empty:
            successful.to_csv(results_dir / "summary_success.csv", index=False)
        print(f"📊 サマリー更新完了: {len(successful)}/{len(dataframe)} 成功")
    except Exception as exc:
        print(f"⚠ サマリー更新失敗: {exc}")
