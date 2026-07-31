"""Dispatch from a configured basis type to its construction function."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .linmol import build_linmol
from .twolevel import build_twolevel
from .vibladder import build_vibladder


@dataclass(frozen=True)
class CouplingSpec:
    """Describe how a model couples to the electric field."""

    mode: Literal["cartesian", "scalar"]
    axis: Literal["x", "y", "z"] | None = None
    default_axes: str = "xy"


@dataclass(frozen=True)
class ModelComponents:
    basis: Any
    state: Any
    hamiltonian: Any
    dipole: Any
    coupling: CouplingSpec


def build_model(params: dict[str, Any]) -> ModelComponents:
    """Build a configured model using the same dispatch as the existing runner."""
    basis_type = params.get("basis_type", "linmol").lower()
    builders = {
        "linmol": (build_linmol, CouplingSpec("cartesian")),
        "twolevel": (build_twolevel, CouplingSpec("scalar", axis="x")),
        "vibladder": (build_vibladder, CouplingSpec("scalar", axis="z")),
    }
    try:
        builder, coupling = builders[basis_type]
    except KeyError:
        raise ValueError(f"Unknown basis_type: {basis_type}") from None
    parts = builder(params)
    return ModelComponents(*parts, coupling=coupling)
