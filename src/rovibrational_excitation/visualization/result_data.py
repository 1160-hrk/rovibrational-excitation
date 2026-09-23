"""Validated plotting projections from one stored simulation result."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any

from numpy.typing import NDArray

from rovibrational_excitation.io.result_schema import (
    ResultFormatError,
    load_simulation_result,
)

ResultPath = str | PathLike[str]


def load_field_plot_data(result_dir: ResultPath) -> tuple[NDArray[Any], NDArray[Any]]:
    """Load one validated field time axis and scalar/Cartesian field array."""
    result = load_simulation_result(Path(result_dir))
    times_fs = result.arrays["t_E"]
    field_v_per_m = result.arrays["E"]
    if times_fs.ndim != 1:
        raise ResultFormatError("stored field time axis t_E must be one-dimensional")
    if field_v_per_m.ndim not in {1, 2}:
        raise ResultFormatError(
            "stored electric field E must be one- or two-dimensional"
        )
    if field_v_per_m.shape[0] != times_fs.size:
        raise ResultFormatError("stored electric field E length must match t_E")
    return times_fs, field_v_per_m


def load_cartesian_field_plot_data(
    result_dir: ResultPath,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Load a validated two-component Cartesian electric field."""
    times_fs, field_v_per_m = load_field_plot_data(result_dir)
    if field_v_per_m.ndim != 2 or field_v_per_m.shape[1] != 2:
        raise ResultFormatError(
            "electric-field vector plot requires two Cartesian components"
        )
    return times_fs, field_v_per_m


def load_population_plot_data(
    result_dir: ResultPath,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Load one validated population time axis and state-population matrix."""
    result = load_simulation_result(Path(result_dir))
    times_fs = result.arrays["t_p"]
    population = result.arrays["pop"]
    if times_fs.ndim != 1:
        raise ResultFormatError(
            "stored population time axis t_p must be one-dimensional"
        )
    if population.ndim != 2:
        raise ResultFormatError("stored population pop must be two-dimensional")
    if population.shape[0] != times_fs.size:
        raise ResultFormatError("stored population pop length must match t_p")
    return times_fs, population


__all__ = [
    "ResultPath",
    "load_cartesian_field_plot_data",
    "load_field_plot_data",
    "load_population_plot_data",
]
