"""Contracts for the normal-simulation parameter reference and template."""

from __future__ import annotations

import re
from pathlib import Path

from rovibrational_excitation.models.validation import _MODEL_REQUIRED
from rovibrational_excitation.simulation.validation import _GENERATED_REQUIRED

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "docs" / "PARAMETER_REFERENCE.md"
TEMPLATE = ROOT / "examples" / "params_template.py"
LOCAL_LINK = re.compile(r"\[[^]]+\]\((?P<target>[^)]+)\)")
REMOVED_KEYS = {
    "amplitude_sin_mod",
    "carrier_freq_sin_mod",
    "phase_rad_sin_mod",
    "type_mod_sin_mod",
    "mu0_Cm",
    "pulse_duration",
    "use_M",
}


def test_parameter_reference_covers_authoritative_required_keys():
    text = REFERENCE.read_text()

    for key in _GENERATED_REQUIRED:
        assert f"`{key}`" in text
    for required in _MODEL_REQUIRED.values():
        for key in required:
            assert f"`{key}`" in text


def test_parameter_reference_rejects_removed_names_and_old_cli():
    combined = REFERENCE.read_text() + TEMPLATE.read_text()

    for key in REMOVED_KEYS:
        assert key not in combined
    assert "python -m rovibrational_excitation.simulation.runner" not in combined
    assert "python -m rovibrational_excitation.cli.simulate" in combined
    assert "実 CUDA は未検証" in combined


def test_parameter_reference_links_resolve():
    missing: list[str] = []
    for match in LOCAL_LINK.finditer(REFERENCE.read_text()):
        target = match.group("target").split("#", 1)[0]
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        if not (REFERENCE.parent / target).resolve().exists():
            missing.append(target)

    assert missing == []
