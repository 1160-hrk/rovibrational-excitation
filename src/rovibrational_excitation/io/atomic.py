"""Atomic replacement of one NPZ or JSON file in its destination directory.

This protects each filename from partial writes. It does not make a group of
result files transactional; the manifest is still published last.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


def _temporary_path(path: Path, *, suffix: str) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=suffix,
        dir=path.parent,
    )
    os.close(descriptor)
    return Path(name)


def atomic_write_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    """Write a compressed NPZ and replace the destination only when complete."""
    path = Path(path)
    temporary = _temporary_path(path, suffix=".npz")
    try:
        np.savez_compressed(temporary, **arrays)
        with temporary.open("rb") as file:
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any, *, sort_keys: bool = False) -> None:
    """Serialize JSON completely before replacing the destination file."""
    path = Path(path)
    encoded = json.dumps(value, indent=2, sort_keys=sort_keys).encode("utf-8")
    temporary = _temporary_path(path, suffix=".tmp")
    try:
        with temporary.open("wb") as file:
            file.write(encoded)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = ["atomic_write_json", "atomic_write_npz"]
