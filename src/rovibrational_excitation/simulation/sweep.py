"""Parameter-sweep expansion without simulation side effects."""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from typing import Any

FIXED_VALUE_KEYS = {"polarization", "initial_states", "envelope_func"}


def expand_cases(base: dict[str, Any]) -> Iterator[tuple[dict[str, Any], list[str]]]:
    """Expand iterable parameters into a Cartesian product of simulation cases."""
    sweep_keys: list[str] = []
    static: dict[str, Any] = {}
    sweep_keys_mapping: dict[str, str] = {}

    for key, value in base.items():
        if isinstance(value, str | bytes):
            static[key] = value
            continue
        if key.endswith("_sweep"):
            if hasattr(value, "__iter__"):
                try:
                    if len(value) > 0:
                        base_key = key[:-6]
                        sweep_keys.append(base_key)
                        sweep_keys_mapping[base_key] = key
                        continue
                except TypeError:
                    pass
            raise ValueError(
                f"Parameter '{key}' has '_sweep' suffix but is not iterable"
            )
        if key in FIXED_VALUE_KEYS:
            static[key] = value
            continue
        if hasattr(value, "__iter__"):
            try:
                if len(value) >= 1:
                    sweep_keys.append(key)
                    continue
            except TypeError:
                pass
        static[key] = value

    if not sweep_keys:
        yield static, []
        return

    iterables = [base[sweep_keys_mapping.get(key, key)] for key in sweep_keys]
    for combination in itertools.product(*iterables):
        case = static.copy()
        case.update(dict(zip(sweep_keys, combination)))
        yield case, sweep_keys


def label(value: Any) -> str:
    """Format a parameter value for use in a result-directory name."""
    if isinstance(value, int | float):
        return f"{value:g}"
    return str(value).replace(" ", "").replace("\n", "")
