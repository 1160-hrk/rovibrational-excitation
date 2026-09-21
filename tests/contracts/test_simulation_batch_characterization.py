"""Characterize simulation retry, failure, and checkpoint cadence."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock, call, patch

import numpy as np

from rovibrational_excitation.io import json_safe
from rovibrational_excitation.simulation.runner import (
    _run_one_safe,
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
