"""Result-directory and summary-file persistence."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .result_schema import (
    CURRENT_RESULT_NAME,
    MANIFEST_NAME,
    ResultFormatError,
    load_simulation_result,
)


def make_results_root(description: str) -> Path:
    """Create the timestamped root directory used by batch runs."""
    root = Path("results") / f"{datetime.now():%Y%m%d_%H%M%S}_{description}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def update_summary(results_dir: Path, all_cases: list[dict[str, Any]]) -> None:
    """Rebuild resumed-run CSV summaries only from validated disk results."""
    rows = []
    for case in all_cases:
        row = {
            key: value for key, value in case.items() if key not in ["outdir", "save"]
        }
        result_dir = Path(case["outdir"])
        result_file = result_dir / "result.npz"

        if (
            result_file.exists()
            or (result_dir / MANIFEST_NAME).exists()
            or (result_dir / CURRENT_RESULT_NAME).exists()
            or (result_dir / CURRENT_RESULT_NAME).is_symlink()
        ):
            saved = load_simulation_result(result_dir)
            population = saved.arrays["pop"]
            if population.ndim != 2 or population.shape[0] == 0:
                raise ResultFormatError(
                    f"result population must have a nonempty time axis: {result_file}"
                )
            pop_final = population[-1]
            row.update(
                {f"pop_{index}": float(value) for index, value in enumerate(pop_final)}
            )
            row["status"] = "success"
        else:
            row["status"] = "failed"
        rows.append(row)

    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(results_dir / "summary.csv", index=False)
    successful = dataframe[dataframe["status"] == "success"]
    if not successful.empty:
        successful.to_csv(results_dir / "summary_success.csv", index=False)
    print(f"📊 サマリー更新完了: {len(successful)}/{len(dataframe)} 成功")
