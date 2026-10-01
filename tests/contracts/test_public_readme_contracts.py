"""Executable and truthful contracts for the two public repository READMEs."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
README_PATHS = (ROOT / "README.md", ROOT / "README_JP.md")
SMOKE_BLOCK = re.compile(
    r"```python\n# README_SMOKE\n(?P<code>.*?)```",
    flags=re.DOTALL,
)
LOCAL_LINK = re.compile(r"\[[^]]+\]\((?P<target>[^)]+)\)")
REMOVED_PUBLIC_CALLS = (
    "rve.LinMolBasis",
    "rve.LinMolDipoleMatrix",
    "rve.generate_H0",
    "rve.schrodinger_propagation",
    "rovibrational_excitation.dipole",
)


@pytest.mark.parametrize("readme", README_PATHS, ids=lambda path: path.name)
def test_public_readme_uses_current_api_and_ci_evidence(readme: Path):
    text = readme.read_text()

    assert ".github/workflows/ci.yml" in text
    assert "actions/workflows/tests.yml" not in text
    assert "63%" not in text
    for removed in REMOVED_PUBLIC_CALLS:
        assert removed not in text


@pytest.mark.parametrize("readme", README_PATHS, ids=lambda path: path.name)
def test_public_readme_has_one_executable_generated_field_quickstart(readme: Path):
    matches = list(SMOKE_BLOCK.finditer(readme.read_text()))
    assert len(matches) == 1

    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    completed = subprocess.run(
        [sys.executable, "-c", matches[0].group("code")],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    assert "final populations:" in completed.stdout


@pytest.mark.parametrize("readme", README_PATHS, ids=lambda path: path.name)
def test_public_readme_local_links_resolve(readme: Path):
    missing: list[str] = []
    for match in LOCAL_LINK.finditer(readme.read_text()):
        target = match.group("target").split("#", 1)[0]
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        if not (readme.parent / target).resolve().exists():
            missing.append(target)

    assert missing == []


def test_public_readmes_record_real_cuda_acceptance_without_speed_claim():
    english = README_PATHS[0].read_text()
    japanese = README_PATHS[1].read_text()

    assert "Real-CUDA execution is numerically accepted" in english
    assert "実 CUDA での数値実行は受入れ済み" in japanese
    assert "does not establish a general GPU speed advantage" in english
    assert "一般的なGPU速度優位を意味しません" in japanese
