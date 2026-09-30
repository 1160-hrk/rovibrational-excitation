"""Mechanical contracts for repository documentation and YAML content."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

import yaml

ROOT = Path(__file__).resolve().parents[2]
IGNORED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "build",
    "dist",
    "htmlcov",
    "node_modules",
    "results",
    "venv",
}
FENCE = re.compile(r"^\s{0,3}(?P<fence>`{3,}|~{3,})")
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]*\]\((?P<target>[^)]+)\)")


def _repository_files(*patterns: str) -> list[Path]:
    return sorted(
        path
        for pattern in patterns
        for path in ROOT.rglob(pattern)
        if path.is_file()
        and not any(part in IGNORED_PARTS for part in path.relative_to(ROOT).parts)
    )


def _rendered_markdown_lines(path: Path) -> list[tuple[int, str]]:
    rendered: list[tuple[int, str]] = []
    open_fence: tuple[str, int] | None = None
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        match = FENCE.match(line)
        if match is not None:
            marker = match.group("fence")
            if open_fence is None:
                open_fence = (marker[0], len(marker))
            elif marker[0] == open_fence[0] and len(marker) >= open_fence[1]:
                open_fence = None
            continue
        if open_fence is None:
            rendered.append((line_number, line))
    if open_fence is not None:
        raise AssertionError(f"{path.relative_to(ROOT)} has an unclosed code fence")
    return rendered


def _local_target(raw_target: str) -> str | None:
    target = raw_target.strip()
    if target.startswith("<") and ">" in target:
        target = target[1 : target.index(">")]
    else:
        target = target.split(maxsplit=1)[0]
    if not target or target.startswith(("#", "http://", "https://", "mailto:")):
        return None
    return unquote(target.split("#", maxsplit=1)[0]) or None


def test_all_markdown_code_fences_are_closed():
    markdown_files = _repository_files("*.md")
    assert markdown_files
    for path in markdown_files:
        _rendered_markdown_lines(path)


def test_all_rendered_markdown_local_links_resolve():
    missing: list[str] = []
    for path in _repository_files("*.md"):
        for line_number, line in _rendered_markdown_lines(path):
            for match in MARKDOWN_LINK.finditer(line):
                target = _local_target(match.group("target"))
                if target is None:
                    continue
                candidate = (path.parent / target).resolve()
                if not candidate.is_relative_to(ROOT) or not candidate.exists():
                    missing.append(
                        f"{path.relative_to(ROOT)}:{line_number} -> {target}"
                    )
    assert missing == []


def test_all_repository_yaml_files_parse():
    yaml_files = _repository_files("*.yml", "*.yaml")
    assert yaml_files
    for path in yaml_files:
        yaml.load(path.read_text(), Loader=yaml.BaseLoader)
