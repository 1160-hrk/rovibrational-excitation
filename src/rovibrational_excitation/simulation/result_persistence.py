"""Persist one normal-simulation result and its versioned disk manifest."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..fields import CartesianField, ScalarField
from ..io import json_safe
from ..io.atomic import atomic_write_json, atomic_write_npz
from ..io.result_schema import (
    create_result_generation,
    publish_result_generation,
    write_result_manifest,
)
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
    """Write unchanged D-017 M-average arrays with a versioned manifest."""
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
    result_dir = create_result_generation(outdir)
    atomic_write_npz(result_dir / "result.npz", save_data)
    atomic_write_json(result_dir / "parameters.json", json_safe(params))
    write_result_manifest(
        result_dir,
        representation="m_incoherent_average",
        arrays=save_data,
        has_regime_info=False,
    )
    publish_result_generation(outdir, result_dir)


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
    """Write unchanged numeric pure-state arrays with a versioned manifest."""
    save_data: dict[str, Any] = {
        "t_E": field_times_fs,
        "psi": state,
        "pop": population,
        "E": _field_samples_for_storage(sampled_field),
        "t_p": times_fs,
    }
    result_dir = create_result_generation(outdir)
    atomic_write_npz(result_dir / "result.npz", save_data)
    atomic_write_json(result_dir / "parameters.json", json_safe(params))
    if regime_info is not None:
        atomic_write_json(result_dir / "regime_analysis.json", json_safe(regime_info))
    write_result_manifest(
        result_dir,
        representation="wavefunction",
        arrays=save_data,
        has_regime_info=regime_info is not None,
    )
    publish_result_generation(outdir, result_dir)


__all__ = ["persist_m_average_result", "persist_wavefunction_result"]
