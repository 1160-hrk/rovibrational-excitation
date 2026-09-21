"""Batch execution and checkpoint cadence shared by run and resume."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, NamedTuple, TypeAlias

from ..io import CheckpointManager
from .safe_execution import CaseRunOutcome

Case: TypeAlias = dict[str, Any]
CaseOutcome: TypeAlias = tuple[Case, Any, str | None]


class BatchRun(NamedTuple):
    """The existing per-case projections needed by batch callers."""

    completed_cases: list[Case]
    failed_cases: list[Case]
    results: list[Any]
    outcomes: list[CaseOutcome]


def execute_case_batches(
    cases: list[Case],
    *,
    all_cases: list[Case],
    checkpoint_manager: CheckpointManager | None,
    checkpoint_interval: int,
    nproc: int,
    start_time: float,
    execute_case: Callable[[Case], CaseRunOutcome],
    progress: Callable[..., Iterable[Any]],
    pool_factory: Callable[[int], Any],
    progress_label: str,
    initial_failed_cases: list[Case] | None = None,
) -> BatchRun:
    """Execute the inherited fixed-size batches and checkpoint on the same cadence."""
    completed_cases: list[Case] = []
    failed_cases: list[Case] = (
        list(initial_failed_cases) if initial_failed_cases is not None else []
    )
    results: list[Any] = []
    outcomes: list[CaseOutcome] = []

    for i in range(0, len(cases), checkpoint_interval):
        batch = cases[i : i + checkpoint_interval]
        batch_num = i // checkpoint_interval + 1
        total_batches = (len(cases) + checkpoint_interval - 1) // checkpoint_interval

        print(
            f"🔄 バッチ {batch_num}/{total_batches} を実行中... ({len(batch)} ケース)"
        )

        if nproc > 1:
            with pool_factory(nproc) as pool:
                batch_results = list(
                    progress(
                        pool.imap(execute_case, batch),
                        total=len(batch),
                        desc=f"{progress_label} {batch_num}",
                    )
                )
        else:
            batch_results = [
                execute_case(case)
                for case in progress(batch, desc=f"{progress_label} {batch_num}")
            ]

        for case, (result, error) in zip(batch, batch_results):
            outcomes.append((case, result, error))
            if error is None:
                completed_cases.append(case)
                results.append(result)
            else:
                failed_case = case.copy()
                failed_case["error"] = error
                failed_cases.append(failed_case)

        if batch_num % 2 == 0 or batch_num == total_batches:
            all_completed: list[Case] = []
            if checkpoint_manager is not None:
                existing_checkpoint = checkpoint_manager.load_checkpoint()
                if existing_checkpoint:
                    completed_hashes = set(
                        existing_checkpoint.get("completed_case_hashes", [])
                    )
                    for case in all_cases:
                        case_hash = checkpoint_manager._case_hash(case)
                        if case_hash in completed_hashes:
                            all_completed.append(case)
            all_completed.extend(completed_cases)

            if checkpoint_manager is not None:
                checkpoint_manager.save_checkpoint(
                    all_completed, failed_cases, len(all_cases), start_time
                )

    return BatchRun(completed_cases, failed_cases, results, outcomes)
