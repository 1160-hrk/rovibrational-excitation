"""Persist one normal-simulation result using the established v0.2 schema."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..fields import CartesianField, ScalarField
from ..io import json_safe
from .m_average import MAveragePropagationResult


def _field_samples_for_storage(sampled_field: Any) -> np.ndarray:
    """Return canonical sampled values without changing their meaning."""
    if isinstance(sampled_field, ScalarField):
        return sampled_field.samples_v_per_m
    if isinstance(sampled_field, CartesianField):
        return sampled_field.components_v_per_m
    return np.asarray(sampled_field.Efield)


def persist_m_average_result(
    *,
    outdir: Path,
    params: Mapping[str, Any],
    field_times_fs: np.ndarray,
    sampled_field: Any,
    result: MAveragePropagationResult,
) -> None:
    """Write the exact established D-017 M-average result payload."""
    save_data: dict[str, Any] = {
        "t_E": field_times_fs,
        "pop": result.population,
        "E": _field_samples_for_storage(sampled_field),
        "t_p": result.time_fs,
        "representation": np.array("m_incoherent_average"),
        "abs_m": np.array([block.abs_m for block in result.blocks]),
        "m_multiplicity": np.array([block.multiplicity for block in result.blocks]),
        "m_weight": np.array([block.weight for block in result.blocks]),
    }
    for block, wavefunction in zip(result.blocks, result.block_wavefunctions):
        save_data[f"psi_abs_m_{block.abs_m}"] = wavefunction
    np.savez_compressed(outdir / "result.npz", **save_data)
    with open(outdir / "parameters.json", "w") as file:
        json.dump(json_safe(params), file, indent=2)


def persist_wavefunction_result(
    *,
    outdir: Path,
    params: Mapping[str, Any],
    field_times_fs: np.ndarray,
    sampled_field: Any,
    times_fs: np.ndarray,
    state: np.ndarray,
    population: np.ndarray,
    regime_info: Any | None,
) -> None:
    """Write the exact established normal pure-state result payload."""
    save_data: dict[str, Any] = {
        "t_E": field_times_fs,
        "psi": state,
        "pop": population,
        "E": _field_samples_for_storage(sampled_field),
        "t_p": times_fs,
    }
    if regime_info is not None:
        save_data["regime_info"] = regime_info
    np.savez_compressed(outdir / "result.npz", **save_data)
    with open(outdir / "parameters.json", "w") as file:
        json.dump(json_safe(params), file, indent=2)
    if regime_info is not None:
        with open(outdir / "regime_analysis.json", "w") as file:
            json.dump(json_safe(regime_info), file, indent=2)


__all__ = ["persist_m_average_result", "persist_wavefunction_result"]
