"""Contracts for the public documentation index."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "docs" / "README.md"
LOCAL_LINK = re.compile(r"\[[^]]+\]\((?P<target>[^)]+)\)")


def test_documentation_index_has_no_removed_or_unverified_recommendations():
    text = INDEX.read_text()
    forbidden = (
        'backend="cupy"',
        "dense=False",
        "mu0_Cm",
        "params_CO2_AntiSymm.py",
        "params_example_new_sweep.py",
        "params_example_checkpoint.py",
        "python -m rovibrational_excitation.simulation.runner",
    )

    for value in forbidden:
        assert value not in text
    assert "実 CUDA 数値受入れ" in text
    assert "移行監査中" not in text
    assert "| 現行の数値契約 |" in text
    assert "| 現行の単位契約 |" in text


def test_documentation_index_local_links_resolve():
    missing: list[str] = []
    for match in LOCAL_LINK.finditer(INDEX.read_text()):
        target = match.group("target").split("#", 1)[0]
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        if not (INDEX.parent / target).resolve().exists():
            missing.append(target)

    assert missing == []
