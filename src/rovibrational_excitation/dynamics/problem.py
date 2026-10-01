"""Immutable model, coupling, and propagation problem contracts."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Literal, Protocol

import numpy as np

from ..core.model import Axis, CouplingMode, CouplingSpec, SystemModel
from ..core.states import DensityState, IncoherentEnsemble, PureState
from ..core.time import TimeGrid
from .capabilities import StatePath

_FIELDS_MODULE = import_module("rovibrational_excitation.fields")
_ELECTRIC_FIELD_TYPES: tuple[type[Any], ...] = (
    _FIELDS_MODULE.ElectricField,
    _FIELDS_MODULE.ScalarField,
    _FIELDS_MODULE.CartesianField,
)
_SCALAR_FIELD_TYPE: type[Any] = _FIELDS_MODULE.ScalarField
_CARTESIAN_FIELD_TYPE: type[Any] = _FIELDS_MODULE.CartesianField


class _ElectricFieldLike(Protocol):
    """Minimum field interface owned by a propagation problem."""

    tlist: np.ndarray[Any, Any]


PropagationState = PureState | IncoherentEnsemble | DensityState


@dataclass(frozen=True, slots=True, eq=False)
class PropagationProblem:
    """Complete physical input for one propagation calculation."""

    model: SystemModel
    field: _ElectricFieldLike
    time_grid: TimeGrid
    initial_state: PropagationState

    def __post_init__(self) -> None:
        if not isinstance(self.model, SystemModel):
            raise TypeError("model must be a SystemModel")
        if not isinstance(self.field, _ELECTRIC_FIELD_TYPES):
            raise TypeError(
                "field must be an ElectricField, ScalarField, or CartesianField"
            )
        if not isinstance(self.time_grid, TimeGrid):
            raise TypeError("time_grid must be a TimeGrid")
        if not isinstance(
            self.initial_state,
            (PureState, IncoherentEnsemble, DensityState),
        ):
            raise TypeError(
                "initial_state must be a PureState, IncoherentEnsemble, or DensityState"
            )
        if not np.array_equal(self.field.tlist, self.time_grid.field_times_fs):
            raise ValueError("field time grid must exactly equal time_grid")
        if (
            isinstance(self.field, _SCALAR_FIELD_TYPE)
            and self.model.coupling.mode is not CouplingMode.SCALAR
        ):
            raise ValueError("Cartesian coupling requires a CartesianField")
        if (
            isinstance(self.field, _CARTESIAN_FIELD_TYPE)
            and self.model.coupling.mode is not CouplingMode.CARTESIAN
        ):
            raise ValueError("scalar coupling requires a ScalarField")
        if self.initial_state.dimension != self.model.dimension:
            raise ValueError(
                "initial-state dimension must equal model dimension; "
                f"got {self.initial_state.dimension} and {self.model.dimension}"
            )

    @property
    def coupling(self) -> CouplingSpec:
        """Return the model-owned coupling selection."""
        return self.model.coupling

    @property
    def coupling_mode(self) -> Literal["cartesian", "scalar"]:
        """Project the typed coupling mode to the temporary solver literal."""
        return self.coupling.mode.value

    @property
    def coupling_kwargs(self) -> dict[str, str]:
        """Project the model-owned axes to the temporary solver fields."""
        return self.coupling.propagation_kwargs()

    @property
    def state_path(self) -> StatePath:
        """Return the explicit numerical state path."""
        if isinstance(self.initial_state, PureState):
            return StatePath.PURE
        if isinstance(self.initial_state, IncoherentEnsemble):
            return StatePath.INCOHERENT_ENSEMBLE
        return StatePath.DENSITY


__all__ = [
    "Axis",
    "CouplingMode",
    "CouplingSpec",
    "PropagationProblem",
    "PropagationState",
    "SystemModel",
]
