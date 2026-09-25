#!/usr/bin/env python3
"""Prepare a final package version without committing, tagging, or publishing.

The script is deliberately local and reversible. The dry-run mode validates the
version transition without writing. The apply mode requires a clean worktree,
updates only pyproject.toml, runs every local release gate, and restores the
original file if a gate fails. Creating and pushing the reviewed commit/tag is
always a separate explicit user action.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
_ACTIVE_SCOPE = ("src", "tests", "examples", "benchmarks", "scripts")
_CURRENT_VERSION = re.compile(
    r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
    r"(?P<dev>\.dev(?:0|[1-9]\d*))?$"
)
_FINAL_VERSION = re.compile(
    r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)$"
)


def read_current_version(path: Path = PYPROJECT) -> str:
    """Read the package version from the authoritative project metadata."""
    with path.open("rb") as stream:
        data = tomllib.load(stream)
    value = data["project"]["version"]
    if not isinstance(value, str):
        raise ValueError("project.version must be a string")
    return value


def _core(match: re.Match[str]) -> tuple[int, int, int]:
    return (
        int(match.group("major")),
        int(match.group("minor")),
        int(match.group("patch")),
    )


def validate_release_transition(
    current_version: str,
    target_version: str,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Validate a supported development/final to final release transition."""
    current_match = _CURRENT_VERSION.fullmatch(current_version)
    if current_match is None:
        raise ValueError(
            "current project version must be X.Y.Z or X.Y.Z.devN before release"
        )
    target_match = _FINAL_VERSION.fullmatch(target_version)
    if target_match is None:
        raise ValueError("release target must be a final X.Y.Z version")

    current_core = _core(current_match)
    target_core = _core(target_match)
    current_is_development = current_match.group("dev") is not None
    if target_core < current_core or (
        target_core == current_core and not current_is_development
    ):
        raise ValueError("release target must be newer than the current final version")
    return current_core, target_core


def _replace_project_version(
    source: str,
    *,
    current_version: str,
    target_version: str,
) -> str:
    old = f'version = "{current_version}"'
    new = f'version = "{target_version}"'
    if source.count(old) != 1:
        raise ValueError(
            "pyproject.toml must contain exactly one current version entry"
        )
    return source.replace(old, new, 1)


def _run(command: Sequence[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _require_clean_worktree() -> None:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stdout.strip():
        raise RuntimeError(
            "release preparation requires a clean worktree; review or commit changes first"
        )


def _run_release_gates(target_version: str) -> None:
    python = sys.executable
    scope = list(_ACTIVE_SCOPE)
    commands = [
        [python, "-m", "ruff", "check", "--no-fix", *scope],
        [python, "-m", "ruff", "format", "--check", *scope],
        [python, "-m", "mypy"],
        [python, "-m", "pytest", "-q"],
        [python, str(ROOT / "scripts" / "smoke_examples.py")],
        [python, "-m", "build"],
    ]
    for command in commands:
        _run(command)

    distributions = sorted((ROOT / "dist").glob(f"*{target_version}*"))
    if not distributions:
        raise RuntimeError(
            f"build did not produce a distribution for version {target_version}"
        )
    _run([python, "-m", "twine", "check", *(str(path) for path in distributions)])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate or prepare a final rovibrational-excitation release"
    )
    parser.add_argument("version", help="final release version in X.Y.Z form")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the transition without writing",
    )
    mode.add_argument(
        "--apply",
        action="store_true",
        help="update pyproject.toml and run all local release gates",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        current = read_current_version()
        validate_release_transition(current, args.version)
        print(f"release transition: {current} -> {args.version}")

        if args.dry_run:
            print(
                "dry run: no files changed; no commit, tag, push, or publish performed"
            )
            return 0

        _require_clean_worktree()
        original = PYPROJECT.read_text(encoding="utf-8")
        updated = _replace_project_version(
            original,
            current_version=current,
            target_version=args.version,
        )
        PYPROJECT.write_text(updated, encoding="utf-8")
        try:
            _run_release_gates(args.version)
        except BaseException:
            PYPROJECT.write_text(original, encoding="utf-8")
            print(
                "release gates failed; restored the original pyproject.toml",
                file=sys.stderr,
            )
            raise

        print("release preparation passed")
        print(
            "Review and commit pyproject.toml, then create/push the matching tag explicitly."
        )
        print("The tag workflow repeats CPU gates and requires a real-GPU runner.")
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"release preparation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
