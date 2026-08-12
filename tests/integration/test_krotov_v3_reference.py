"""Short end-to-end guard for the stored four-level Krotov reference."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from rovibrational_excitation.simulation.optimize_runner import run_from_config

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "reference_krotov_viblad_v3.yaml"


@pytest.mark.slow
def test_krotov_v0_to_v3_improves_the_unoptimized_pulse(tmp_path: Path) -> None:
    with CONFIG.open(encoding="utf-8") as stream:
        base = yaml.safe_load(stream)

    unoptimized = deepcopy(base)
    unoptimized["algorithms"]["krotov"]["max_iter"] = 0
    optimized = deepcopy(base)
    optimized["algorithms"]["krotov"]["max_iter"] = 10

    initial_result = run_from_config(
        config=unoptimized, out_dir=tmp_path, do_plot=False
    )["result"]
    optimized_result = run_from_config(
        config=optimized, out_dir=tmp_path, do_plot=False
    )["result"]

    initial_fidelity = initial_result["metrics"]["fidelity"]
    optimized_fidelity = optimized_result["metrics"]["fidelity"]
    assert initial_fidelity == pytest.approx(0.053696933060449)
    assert optimized_fidelity == pytest.approx(0.591542443817852)
    assert optimized_fidelity > initial_fidelity
