"""Serialization helpers for simulation configuration and metadata."""

from __future__ import annotations

import types
from typing import Any

import numpy as np


def json_safe(obj: Any) -> Any:
    """Recursively convert complex and NumPy values to JSON-safe values."""
    if isinstance(obj, complex):
        return {"__complex__": True, "r": obj.real, "i": obj.imag}
    if callable(obj):
        return (
            f"{getattr(obj, '__module__', 'builtins')}."
            f"{getattr(obj, '__qualname__', str(obj))}"
        )
    if isinstance(obj, types.ModuleType):
        return obj.__name__
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return [json_safe(value) for value in obj.tolist()]
    if isinstance(obj, list | tuple):
        return [json_safe(value) for value in obj]
    if isinstance(obj, dict):
        return {key: json_safe(value) for key, value in obj.items()}
    return obj


def deserialize_polarization(value: Any) -> np.ndarray:
    """Convert serialized scalar or vector polarization to a complex vector."""

    def to_complex(component: Any) -> complex:
        if isinstance(component, dict):
            real = component.get("r", component.get("real", 0))
            imaginary = component.get("i", component.get("imag", 0))
            return complex(real, imaginary)
        if isinstance(component, float | int | complex):
            return complex(component)
        raise TypeError(f"Invalid polarization component: {type(component)}")

    if isinstance(value, int | float | complex | dict):
        return np.array([to_complex(value), 0], dtype=complex)
    if hasattr(value, "__iter__") and not isinstance(value, str | bytes):
        components = list(value)
        if len(components) == 1:
            return np.array([to_complex(components[0]), 0], dtype=complex)
        if len(components) == 2:
            return np.array([to_complex(item) for item in components], dtype=complex)
        raise ValueError(
            f"Polarization must have 1 or 2 elements, got {len(components)}"
        )
    return np.array([1.0, 0.0], dtype=complex)
