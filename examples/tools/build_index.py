#!/usr/bin/env python
import os
import sys
from pathlib import Path
import textwrap


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = PROJECT_ROOT / "examples"
README_PATH = EXAMPLES_DIR / "README.md"


def extract_docstring(path: Path) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        # very light-weight extraction
        if '"""' in src:
            s = src.split('"""', 2)
            if len(s) >= 3:
                return s[1].strip().splitlines()[0]
        if "'''" in src:
            s = src.split("'''", 2)
            if len(s) >= 3:
                return s[1].strip().splitlines()[0]
    except Exception:
        pass
    return ""


def extract_tags(path: Path) -> list[str]:
    tags: list[str] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for _ in range(40):
                line = f.readline()
                if not line:
                    break
                if "tags:" in line.lower():
                    _, rhs = line.split(":", 1)
                    tags = [t.strip().lower() for t in rhs.split(',') if t.strip()]
                    break
    except Exception:
        pass
    return tags


def classify_example(p: Path) -> str:
    name = p.name.lower()
    if "twolevel" in name or "vibrational_2d" in name:
        return "Two-level / Vibrational sweeps"
    if "nondimensional" in name:
        return "Propagation"
    if "rovibrational_excitation" in name:
        return "Propagation"
    if "krotov" in name or "grape" in name or "optimization" in name:
        return "Optimization"
    if "absorbance" in name or "transient" in name or "spectroscopy" in str(p).lower():
        return "Spectroscopy"
    if "archives" in str(p).lower():
        return "Archives"
    return "Quickstart"


def scan_examples() -> dict[str, list[tuple[str, str, list[str]]]]:
    groups: dict[str, list[tuple[str, str, list[str]]]] = {}
    for py in EXAMPLES_DIR.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        if "runners" in py.parts:
            continue
        category = classify_example(py)
        title = extract_docstring(py)
        tags = extract_tags(py)
        rel = py.relative_to(PROJECT_ROOT).as_posix()
        groups.setdefault(category, []).append((rel, title, tags))
    # stable order
    for k in groups:
        groups[k].sort(key=lambda x: x[0])
    return groups


def build_readme(groups: dict[str, list[tuple[str, str, list[str]]]]) -> str:
    header = textwrap.dedent(
        """
        # Examples Catalog

        This directory provides runnable examples for how to use the code under `src/`.

        Goals:
        - Beginner-friendly: run quickly and see outputs in minutes
        - Consistent outputs: results are saved under `results/<YYYYMMDD_HHMMSS>_<algorithm>_<system>` when applicable
        - Interactive-friendly: most scripts have `# %%` cells for Interactive Window/Jupyter

        ## Quick Start

        Run a minimal demo (fast):

        ```bash
        python examples/launcher.py --run quickstart --quick
        ```

        Open an interactive Jupyter session:

        ```bash
        bash scripts/start_jupyter.sh
        # Then open examples/notebooks/01_quickstart.ipynb
        ```

        Use Interactive Window (VS Code/Cursor): open the example file and run by cells (`# %%`).

        ## Quick Mode

        Many examples support a fast "quick" mode to reduce runtime by shrinking grids and durations.
        Enable it via:
        - CLI: `--quick`
        - Environment variable: `EXAMPLES_QUICK=1`

        Quick mode is ideal for first runs and smoke tests.

        ## Categories
        """
    ).strip()

    lines = [header, ""]
    order = [
        "Quickstart",
        "Propagation",
        "Optimization",
        "Two-level / Vibrational sweeps",
        "Spectroscopy",
        "Archives",
    ]
    for cat in order:
        if cat not in groups:
            continue
        lines.append(f"### {cat}")
        for rel, title, tags in groups[cat]:
            tag_str = f" [tags: {', '.join(tags)}]" if tags else ""
            title_str = f" - {title}" if title else ""
            lines.append(f"- `{rel}`{title_str}{tag_str}")
        lines.append("")

    tail = textwrap.dedent(
        """
        ## Usage via Launcher

        List available examples:

        ```bash
        python examples/launcher.py --list
        ```

        Filter by name or tag:

        ```bash
        python examples/launcher.py --list --filter tag=beginner
        python examples/launcher.py --list --filter name=twolevel
        ```

        Run an example (fast mode):

        ```bash
        python examples/launcher.py --run example_twolevel_2d_map --quick
        ```

        ## Output Conventions

        All runners and optimization examples save into a timestamped directory under `results/` at the project root. Files include:
        - run_config_*.yaml: The resolved configuration
        - run_meta_*.json: Metadata (timestamp, algorithm, elapsed time)
        - figures and data: Plots and arrays generated by the example
        """
    ).strip()
    lines.append(tail)
    return "\n\n".join(lines) + "\n"


def main() -> int:
    groups = scan_examples()
    md = build_readme(groups)
    os.makedirs(EXAMPLES_DIR, exist_ok=True)
    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Wrote {README_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


