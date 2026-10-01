#!/usr/bin/env python3
"""Emit GitHub Check annotations for failures stored in JUnit XML."""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path


def _escape_message(value: str) -> str:
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _escape_property(value: str) -> str:
    return _escape_message(value).replace(":", "%3A").replace(",", "%2C")


def failure_annotations(path: Path) -> Iterator[str]:
    """Yield escaped GitHub error annotations for JUnit failures and errors."""
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
        yield f"::error title={_escape_property(title)}::{_escape_message(detail[:4000])}"


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        print("usage: report_junit_failures.py JUNIT_XML", file=sys.stderr)
        return 2

    path = Path(arguments[0])
    if not path.is_file():
        print(
            "::error title=JUnit diagnostics unavailable::"
            f"JUnit XML was not created%3A {_escape_message(str(path))}"
        )
        return 0

    try:
        for annotation in failure_annotations(path):
            print(annotation)
    except (ET.ParseError, OSError) as exc:
        print(
            "::error title=JUnit diagnostics unavailable::"
            f"Could not read JUnit XML%3A {_escape_message(str(exc))}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
