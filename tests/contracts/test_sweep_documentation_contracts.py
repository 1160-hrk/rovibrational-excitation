"""Contracts for the public sweep guide and dry-run description."""

from __future__ import annotations

import re
from pathlib import Path

from rovibrational_excitation.cli.simulate import build_parser
from rovibrational_excitation.simulation.sweep import expand_cases

ROOT = Path(__file__).resolve().parents[2]
GUIDE = ROOT / "docs" / "SWEEP_SPECIFICATION.md"
LOCAL_LINK = re.compile(r"\[[^]]+\]\((?P<target>[^)]+)\)")


def test_sweep_guide_has_no_private_api_or_removed_parameter_example():
    text = GUIDE.read_text()

    assert "_expand_cases" not in text
    assert "_load_params_file" not in text
    assert "mu0_Cm" not in text
    assert "後方互換" not in text
    assert "--dry-run --no-save" in text


def test_sweep_guide_records_singleton_order_and_fixed_list_contracts():
    text = GUIDE.read_text()
    expanded = list(
        expand_cases(
            {
                "amplitude": [1.0, 2.0],
                "phase_sweep": [0.1, 0.2],
                "V_max": [3],
                "polarization": [1.0, 0.0],
                "initial_states": [0, 1],
            }
        )
    )

    assert [(case["amplitude"], case["phase"]) for case, _ in expanded] == [
        (1.0, 0.1),
        (1.0, 0.2),
        (2.0, 0.1),
        (2.0, 0.2),
    ]
    assert all(case["V_max"] == 3 for case, _ in expanded)
    assert all(keys == ["amplitude", "phase", "V_max"] for _, keys in expanded)
    assert all(case["polarization"] == [1.0, 0.0] for case, _ in expanded)
    assert all(case["initial_states"] == [0, 1] for case, _ in expanded)
    assert "1要素" in text
    assert "insertion order" in text
    assert "Cartesian product" in text


def test_dry_run_help_describes_count_without_claiming_case_listing():
    help_text = build_parser().format_help()

    assert "report expanded case count without execution" in help_text
    assert "list cases only" not in help_text


def test_sweep_guide_local_links_resolve():
    missing: list[str] = []
    for match in LOCAL_LINK.finditer(GUIDE.read_text()):
        target = match.group("target").split("#", 1)[0]
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        if not (GUIDE.parent / target).resolve().exists():
            missing.append(target)

    assert missing == []
