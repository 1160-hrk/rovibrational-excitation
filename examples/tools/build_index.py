#!/usr/bin/env python3
"""Build or verify the catalog for supported top-level examples only."""

from __future__ import annotations

import argparse
import ast
import difflib
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = PROJECT_ROOT / "examples"
README_PATH = EXAMPLES_DIR / "README.md"


def iter_examples() -> tuple[Path, ...]:
    """Return only supported top-level example modules."""
    return tuple(sorted(EXAMPLES_DIR.glob("example_*.py")))


def _summary(path: Path) -> str:
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    docstring = ast.get_docstring(module, clean=True)
    if not docstring:
        raise ValueError(f"supported example has no module docstring: {path}")
    return docstring.splitlines()[0]


def render_readme() -> str:
    entries = [
        f"- [{path.name}]({path.name}): {_summary(path)}" for path in iter_examples()
    ]
    lines = [
        "# Examples",
        "",
        "Only the top-level example files documented below are supported. They use",
        "the current typed simulation boundary, are linted, and run as smoke tests",
        "in CI. This catalog is generated only from top-level example_*.py files;",
        "helpers, generated outputs, and archives are never scanned.",
        "",
        "## Quick start",
        "",
        "Run the smallest generated-field example:",
        "",
        "~~~bash",
        "python examples/launcher.py --run quickstart --quick",
        "~~~",
        "",
        "List all supported examples:",
        "",
        "~~~bash",
        "python examples/launcher.py --list",
        "~~~",
        "",
        "## Supported examples",
        "",
        *entries,
        "",
        "All supported examples use save=False, complete in seconds, check population",
        "normalization, and avoid plotting or persistent output.",
        "",
        "The parameter-file template is also executed without saving by the smoke",
        "suite, but it is not an additional example module:",
        "",
        "~~~bash",
        "python -m rovibrational_excitation.cli.simulate examples/params_template.py --no-save",
        "~~~",
        "",
        "Run the same smoke suite used by CI:",
        "",
        "~~~bash",
        "python scripts/smoke_examples.py",
        "~~~",
        "",
        "## Historical material",
        "",
        "examples/archives contains migration evidence and scripts written against",
        "older APIs. In particular, archives/v0_2_scripts contains the former",
        "top-level examples and their dedicated helpers. Some require removed",
        "calling conventions, undefined experiment-specific constants, or external",
        "compiled extensions.",
        "",
        "Archived files are intentionally:",
        "",
        "- not listed or executed by examples/launcher.py;",
        "- excluded from Ruff and CI smoke tests;",
        "- not scanned by this catalog builder;",
        "- not repaired by guessed physical or optimization parameters.",
        "",
        "Use archived code only as historical reference. Before restoring one as a",
        "supported example, migrate it to the current public API, add a bounded",
        "quick mode, and include it in scripts/smoke_examples.py.",
        "",
        "Verify that this generated catalog is current with:",
        "",
        "~~~bash",
        "python examples/tools/build_index.py --check",
        "~~~",
        "",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the supported-example catalog")
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if examples/README.md differs from the generated catalog",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    expected = render_readme()
    current = README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else ""
    if args.check:
        if current == expected:
            print("examples/README.md is current")
            return 0
        print(
            "".join(
                difflib.unified_diff(
                    current.splitlines(keepends=True),
                    expected.splitlines(keepends=True),
                    fromfile=str(README_PATH),
                    tofile="generated examples catalog",
                )
            ),
            end="",
        )
        return 1

    README_PATH.write_text(expected, encoding="utf-8")
    print(f"Wrote {README_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
