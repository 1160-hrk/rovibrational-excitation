#!/usr/bin/env python
import argparse
import os
import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT


def iter_examples():
    for py in (EXAMPLES_DIR).rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        if py.name == "launcher.py":
            continue
        if "tools" in py.parts:
            continue
        yield py


def list_examples(filter_expr: str | None = None):
    items = []
    for p in iter_examples():
        rel = p.relative_to(PROJECT_ROOT).as_posix()
        if filter_expr:
            key, _, val = filter_expr.partition("=")
            key = key.strip().lower()
            val = val.strip().lower()
            text = rel.lower()
            if key in ("name", "path"):
                if val not in text:
                    continue
            elif key == "tag":
                # naive: look for 'tags:' near the top
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        head = f.read(5120).lower()
                    if f"tags:" not in head or val not in head:
                        continue
                except Exception:
                    continue
            else:
                if val not in text:
                    continue
        items.append(rel)
    items.sort()
    return items


def run_example(name: str, quick: bool = False):
    # resolve path
    target = None
    for p in iter_examples():
        rel = p.relative_to(PROJECT_ROOT).as_posix()
        if rel.endswith(name) or rel.endswith(name + ".py") or Path(rel).name == name:
            target = p
            break
        if Path(rel).stem == name:
            target = p
            break
    if target is None:
        raise SystemExit(f"Example not found: {name}")

    # quick mode via env
    if quick:
        os.environ["EXAMPLES_QUICK"] = "1"

    # Execute as a script in its own globals
    runpy.run_path(str(target), run_name="__main__")


def main():
    parser = argparse.ArgumentParser(description="Examples launcher")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="List examples")
    group.add_argument("--run", type=str, help="Run an example by name or path")
    parser.add_argument("--filter", type=str, default=None, help="Filter: name=<kw>|tag=<kw>")
    parser.add_argument("--quick", action="store_true", help="Run in quick mode")
    args = parser.parse_args()

    if args.list:
        items = list_examples(args.filter)
        if not items:
            print("No examples found.")
            return 0
        print("Available examples:")
        for it in items:
            print(f"  {it}")
        return 0

    if args.run:
        run_example(args.run, quick=args.quick)
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())


