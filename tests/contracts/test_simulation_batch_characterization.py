"""Characterize simulation retry, failure, and checkpoint cadence."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd

from rovibrational_excitation.io import CheckpointManager, json_safe
from rovibrational_excitation.simulation.runner import (
    _run_one_safe,
    resume_run,
    run_all_with_checkpoint,
)


def test_safe_case_retries_only_oserror_with_exponential_backoff():
    expected = np.array([[0.75, 0.25]])
    with (
        patch(
            "rovibrational_excitation.simulation.runner._run_one",
            side_effect=[OSError("first"), OSError("second"), expected],
        ) as run_one,
        patch("rovibrational_excitation.simulation.runner.time.sleep") as sleep,
    ):
        result, error = _run_one_safe({"save": False}, max_retries=2)

    np.testing.assert_array_equal(result, expected)
    assert error is None
    assert run_one.call_count == 3
    assert sleep.call_args_list == [call(1), call(2)]


def test_safe_case_non_oserror_fails_once_and_writes_returned_traceback(tmp_path):
    params = {
        "save": True,
        "outdir": str(tmp_path),
        "polarization": 1.0 + 2.0j,
    }
    with (
        patch(
            "rovibrational_excitation.simulation.runner._run_one",
            side_effect=ValueError("broken case"),
        ) as run_one,
        patch("rovibrational_excitation.simulation.runner.time.sleep") as sleep,
    ):
        result, error = _run_one_safe(params, max_retries=2)

    assert result is None
    assert error is not None
    assert error.startswith("Attempt 1/3 failed: broken case\nTraceback:\n")
    assert "ValueError: broken case" in error
    run_one.assert_called_once_with(params)
    sleep.assert_not_called()
    assert (tmp_path / "error.txt").read_text() == (
        error + f"\nParameters:\n{json.dumps(json_safe(params), indent=2)}"
    )


def test_batch_checkpoint_is_saved_after_second_and_final_batch(tmp_path):
    manager = Mock()
    manager.load_checkpoint.return_value = None
    populations = [np.array([[float(index), 0.0]]) for index in range(5)]

    with (
        patch(
            "rovibrational_excitation.simulation.runner._make_root",
            return_value=Path(tmp_path),
        ),
        patch(
            "rovibrational_excitation.simulation.runner.CheckpointManager",
            return_value=manager,
        ),
        patch(
            "rovibrational_excitation.simulation.runner._run_one_safe",
            side_effect=[(population, None) for population in populations],
        ),
        patch(
            "rovibrational_excitation.simulation.runner._tqdm",
            side_effect=lambda values, **kwargs: values,
        ),
    ):
        results = run_all_with_checkpoint(
            {
                "description": "checkpoint_cadence",
                "amplitude": [1.0, 2.0, 3.0, 4.0, 5.0],
            },
            save=True,
            checkpoint_interval=2,
        )

    assert len(results) == 5
    assert manager.save_checkpoint.call_count == 2
    first, final = manager.save_checkpoint.call_args_list
    assert len(first.args[0]) == 4
    assert first.args[1] == []
    assert first.args[2] == 5
    assert len(final.args[0]) == 5
    assert final.args[1] == []
    assert final.args[2] == 5
    assert first.args[3] == final.args[3]


def test_normal_summary_uses_returned_population_not_saved_result(tmp_path):
    def run_case(case):
        np.savez(
            Path(case["outdir"]) / "result.npz",
            pop=np.array([[0.9, 0.1]]),
        )
        return np.array([[0.25, 0.75]]), None

    with (
        patch(
            "rovibrational_excitation.simulation.runner._make_root",
            return_value=Path(tmp_path),
        ),
        patch(
            "rovibrational_excitation.simulation.runner._run_one_safe",
            side_effect=run_case,
        ),
        patch(
            "rovibrational_excitation.simulation.runner._tqdm",
            side_effect=lambda values, **kwargs: values,
        ),
    ):
        results = run_all_with_checkpoint(
            {"description": "normal_summary", "amplitude": [1.0]},
            save=True,
        )

    np.testing.assert_array_equal(results[0], [[0.25, 0.75]])
    summary = pd.read_csv(tmp_path / "summary.csv")
    assert summary["status"].tolist() == ["success"]
    assert summary["pop_0"].tolist() == [0.25]
    assert summary["pop_1"].tolist() == [0.75]
    assert (tmp_path / "summary_success.csv").exists()


def test_resume_rebuilds_cases_skips_completed_and_uses_saved_results(tmp_path):
    (tmp_path / "params.py").write_text(
        "description = 'resume_summary'\namplitude = [1.0, 2.0, 3.0]\n"
    )
    old_dir = tmp_path / "amplitude_1"
    old_dir.mkdir()
    np.savez(old_dir / "result.npz", pop=np.array([[0.2, 0.8]]))
    manager = CheckpointManager(tmp_path)
    manager.save_checkpoint(
        [
            {
                "description": "resume_summary",
                "amplitude": 1.0,
                "save": True,
                "outdir": str(old_dir),
            }
        ],
        [],
        3,
        0.0,
    )
    executed = []

    def run_case(case):
        executed.append(case)
        if case["amplitude"] == 2.0:
            np.savez(
                Path(case["outdir"]) / "result.npz",
                pop=np.array([[0.3, 0.7]]),
            )
            return np.array([[0.4, 0.6]]), None
        return None, "new failure"

    with (
        patch(
            "rovibrational_excitation.simulation.runner._run_one_safe",
            side_effect=run_case,
        ),
        patch(
            "rovibrational_excitation.simulation.runner._tqdm",
            side_effect=lambda values, **kwargs: values,
        ),
    ):
        results = resume_run(tmp_path, checkpoint_interval=1)

    assert [case["amplitude"] for case in executed] == [2.0, 3.0]
    assert [case["save"] for case in executed] == [True, True]
    assert [case["outdir"] for case in executed] == [
        str(tmp_path / "amplitude_2"),
        str(tmp_path / "amplitude_3"),
    ]
    np.testing.assert_array_equal(results[0], [[0.4, 0.6]])
    checkpoint = manager.load_checkpoint()
    assert checkpoint is not None
    assert checkpoint["completed_cases"] == 2
    assert checkpoint["failed_cases"] == 1
    assert checkpoint["total_cases"] == 3
    assert checkpoint["failed_case_data"][0]["amplitude"] == 3.0
    assert checkpoint["failed_case_data"][0]["error"] == "new failure"
    summary = pd.read_csv(tmp_path / "summary.csv")
    assert summary["amplitude"].tolist() == [1.0, 2.0, 3.0]
    assert summary["status"].tolist() == ["success", "success", "failed"]
    assert summary.loc[1, "pop_0"] == 0.3
    assert summary.loc[1, "pop_1"] == 0.7
    assert pd.read_csv(tmp_path / "summary_success.csv")["amplitude"].tolist() == [
        1.0,
        2.0,
    ]


def test_parallel_batches_keep_one_pool_per_batch_and_order():
    pool = Mock()
    pool.__enter__ = Mock(return_value=pool)
    pool.__exit__ = Mock(return_value=False)
    pool.imap.side_effect = lambda execute, batch: map(execute, batch)
    populations = [
        np.array([[0.1, 0.9]]),
        np.array([[0.2, 0.8]]),
        np.array([[0.3, 0.7]]),
    ]
    progress_desc = []

    def progress(values, **kwargs):
        progress_desc.append(kwargs["desc"])
        return values

    with (
        patch(
            "rovibrational_excitation.simulation.runner.Pool", return_value=pool
        ) as pool_factory,
        patch("rovibrational_excitation.simulation.runner.cpu_count", return_value=4),
        patch(
            "rovibrational_excitation.simulation.runner._run_one_safe",
            side_effect=[(population, None) for population in populations],
        ),
        patch(
            "rovibrational_excitation.simulation.runner._tqdm",
            side_effect=progress,
        ),
    ):
        results = run_all_with_checkpoint(
            {"amplitude": [1.0, 2.0, 3.0]},
            nproc=2,
            save=False,
            checkpoint_interval=2,
        )

    assert pool_factory.call_args_list == [call(2), call(2)]
    assert pool.imap.call_count == 2
    assert [len(args.args[1]) for args in pool.imap.call_args_list] == [2, 1]
    assert progress_desc == ["Batch 1", "Batch 2"]
    for result, expected in zip(results, populations):
        np.testing.assert_array_equal(result, expected)


def test_normal_case_paths_preserve_sweep_order_labels_and_directories(tmp_path):
    executed = []

    def run_case(case):
        executed.append(case.copy())
        return np.array([[1.0, 0.0]]), None

    with (
        patch(
            "rovibrational_excitation.simulation.runner._make_root",
            return_value=Path(tmp_path),
        ),
        patch(
            "rovibrational_excitation.simulation.runner._run_one_safe",
            side_effect=run_case,
        ),
        patch(
            "rovibrational_excitation.simulation.runner._tqdm",
            side_effect=lambda values, **kwargs: values,
        ),
    ):
        run_all_with_checkpoint(
            {
                "description": "path_contract",
                "amplitude": [1.0, 2.0],
                "phase_sweep": [0.1, 0.2],
            },
            save=True,
            checkpoint_interval=4,
        )

    assert [(case["amplitude"], case["phase"]) for case in executed] == [
        (1.0, 0.1),
        (1.0, 0.2),
        (2.0, 0.1),
        (2.0, 0.2),
    ]
    assert [case["outdir"] for case in executed] == [
        str(tmp_path / "amplitude_1" / "phase_0.1"),
        str(tmp_path / "amplitude_1" / "phase_0.2"),
        str(tmp_path / "amplitude_2" / "phase_0.1"),
        str(tmp_path / "amplitude_2" / "phase_0.2"),
    ]
    assert all(case["save"] is True for case in executed)
    assert all(Path(case["outdir"]).is_dir() for case in executed)


def test_dry_run_still_creates_saved_case_directories_without_executing(tmp_path):
    with (
        patch(
            "rovibrational_excitation.simulation.runner._make_root",
            return_value=Path(tmp_path),
        ),
        patch("rovibrational_excitation.simulation.runner._run_one_safe") as execute,
        patch(
            "rovibrational_excitation.simulation.runner.CheckpointManager"
        ) as manager,
    ):
        result = run_all_with_checkpoint(
            {"amplitude": [1.0, 2.0]},
            save=True,
            dry_run=True,
        )

    assert result == []
    assert (tmp_path / "amplitude_1").is_dir()
    assert (tmp_path / "amplitude_2").is_dir()
    execute.assert_not_called()
    manager.assert_not_called()
    assert not (tmp_path / "summary.csv").exists()


def test_normal_summary_preserves_scalar_vector_and_last_time_row(tmp_path):
    outcomes = [
        (np.array(0.2), None),
        (np.array([0.3, 0.7]), None),
        (np.array([[0.9, 0.1], [0.4, 0.6]]), None),
        (None, "failed fourth"),
    ]
    with (
        patch(
            "rovibrational_excitation.simulation.runner._make_root",
            return_value=Path(tmp_path),
        ),
        patch(
            "rovibrational_excitation.simulation.runner._run_one_safe",
            side_effect=outcomes,
        ),
        patch(
            "rovibrational_excitation.simulation.runner._tqdm",
            side_effect=lambda values, **kwargs: values,
        ),
    ):
        run_all_with_checkpoint(
            {"case_number": [0, 1, 2, 3]},
            save=True,
            checkpoint_interval=4,
        )

    summary = pd.read_csv(tmp_path / "summary.csv")
    assert summary["case_number"].tolist() == [0, 1, 2, 3]
    assert summary["status"].tolist() == [
        "success",
        "success",
        "success",
        "failed",
    ]
    assert summary["pop_0"].iloc[:3].tolist() == [0.2, 0.3, 0.4]
    assert np.isnan(summary.loc[0, "pop_1"])
    assert summary["pop_1"].iloc[1:3].tolist() == [0.7, 0.6]
    assert summary.loc[3, "error"] == "failed fourth"
    assert pd.read_csv(tmp_path / "summary_success.csv")["case_number"].tolist() == [
        0,
        1,
        2,
    ]


def test_normal_report_limits_failure_previews_and_omits_empty_success_csv(tmp_path):
    with (
        patch(
            "rovibrational_excitation.simulation.runner._make_root",
            return_value=Path(tmp_path),
        ),
        patch(
            "rovibrational_excitation.simulation.runner._run_one_safe",
            side_effect=[(None, f"issue-{index}") for index in range(7)],
        ),
        patch(
            "rovibrational_excitation.simulation.runner._tqdm",
            side_effect=lambda values, **kwargs: values,
        ),
        patch("builtins.print") as printed,
    ):
        results = run_all_with_checkpoint(
            {"case_number": list(range(7))},
            save=True,
            checkpoint_interval=7,
        )

    assert results == []
    messages = [args.args[0] for args in printed.call_args_list if args.args]
    assert "✅ 実行完了: 0/7 成功, 7 失敗" in messages
    assert "⚠ 失敗ケース: 7 件" in messages
    assert [message for message in messages if message.startswith("  ")][:5] == [
        f"  {index + 1}. issue-{index}..." for index in range(5)
    ]
    assert "  ... (他 2 件)" in messages
    assert not any("issue-5" in message or "issue-6" in message for message in messages)
    assert pd.read_csv(tmp_path / "summary.csv")["status"].tolist() == ["failed"] * 7
    assert not (tmp_path / "summary_success.csv").exists()
