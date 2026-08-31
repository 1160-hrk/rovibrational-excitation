"""Loading and unit processing for simulation parameter files."""

from __future__ import annotations

import importlib.util
from types import ModuleType
from typing import Any


def load_params_file(path: str) -> dict[str, Any]:
    """Execute a Python parameter file without changing values or unit labels."""
    spec = importlib.util.spec_from_file_location("params", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    params = {
        name: value
        for name in dir(module)
        if not name.startswith("__")
        and not isinstance(value := getattr(module, name), ModuleType)
    }

    print(f"📊 Loading parameters from {path}")
    print("📋 Values and explicit unit labels loaded unchanged.")
    return params
