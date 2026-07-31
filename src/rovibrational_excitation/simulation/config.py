"""Loading and unit processing for simulation parameter files."""

from __future__ import annotations

import importlib.util
from typing import Any

from rovibrational_excitation.core.units.parameter_processor import parameter_processor


def load_params_file(path: str) -> dict[str, Any]:
    """Execute a Python parameter file and convert values to internal units."""
    spec = importlib.util.spec_from_file_location("params", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    params = {
        name: getattr(module, name) for name in dir(module) if not name.startswith("__")
    }

    print(f"📊 Loading parameters from {path}")
    converted = process_params(params)
    if params != converted:
        print("📋 Unit processing completed.")
    else:
        print("📋 No unit processing needed.")
    return converted


def process_params(params: dict[str, Any]) -> dict[str, Any]:
    """Convert a parameter mapping to the established internal units."""
    return parameter_processor.auto_convert_parameters(params.copy())
