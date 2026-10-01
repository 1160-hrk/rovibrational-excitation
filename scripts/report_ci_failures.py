#!/usr/bin/env python3
"""Emit escaped GitHub Check annotations for CI command failures."""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path

_MAX_DETAIL_CHARS = 4000


def _escape_message(value: str) -> str:
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _escape_property(value: str) -> str:
    return _escape_message(value).replace(":", "%3A").replace(",", "%2C")


def _annotation(title: str, detail: str) -> str:
    return (
        f"::error title={_escape_property(title)}::"
        f"{_escape_message(detail[:_MAX_DETAIL_CHARS])}"
    )


def failure_annotations(path: Path) -> Iterator[str]:
    """Yield GitHub error annotations for JUnit failures and errors."""
    root = ET.parse(path).getroot()
    for case in root.iter("testcase"):
        issue = case.find("failure")
        if issue is None:
            issue = case.find("error")
        if issue is None:
            continue

        name = case.get("name", "unknown test")
        class_name = case.get("classname")
        title = f"{class_name}::{name}" if class_name else name
        detail = (issue.text or issue.get("message") or issue.tag).strip()
        yield _annotation(title, detail)


def _missing_annotation(kind: str, path: Path) -> str:
    return _annotation(
        f"{kind} diagnostics unavailable",
        f"Diagnostic file was not created: {path}",
    )


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) == 2 and arguments[0] == "junit":
        path = Path(arguments[1])
        if not path.is_file():
            print(_missing_annotation("JUnit", path))
            return 0
        try:
            for annotation in failure_annotations(path):
                print(annotation)
        except (ET.ParseError, OSError) as exc:
            print(_annotation("JUnit diagnostics unavailable", str(exc)))
        return 0

    if len(arguments) == 3 and arguments[0] == "text":
        title = arguments[1]
        path = Path(arguments[2])
        if not path.is_file():
            print(_missing_annotation(title, path))
            return 0
        try:
            detail = path.read_text(errors="replace").strip()
        except OSError as exc:
            print(_annotation(f"{title} diagnostics unavailable", str(exc)))
            return 0
        print(_annotation(title, detail or "Command failed without output."))
        return 0

    print(
        "usage: report_ci_failures.py junit JUNIT_XML | text TITLE OUTPUT_FILE",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
