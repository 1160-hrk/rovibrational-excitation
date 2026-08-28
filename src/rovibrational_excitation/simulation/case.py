"""Immutable, fully sampled normal-simulation input."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

from rovibrational_excitation.core.time import TimeGrid
from rovibrational_excitation.dynamics.options import PropagationOptions
from rovibrational_excitation.fields import SampledField
from rovibrational_excitation.models.validation import (
    LinMolRepresentation,
    ModelParameters,
    model_parameters_from_mapping,
)


@dataclass(frozen=True, slots=True)
class SimulationCase:
    """Validated values consumed by one normal-simulation execution.

    Generated and externally supplied fields both enter execution through this
    boundary after their samples and time grid have been fixed. The original
    mapping is not retained, so mutable user data cannot alter a case after
    construction.
    """

    basis_type: str
    model_parameters: ModelParameters
    representation: LinMolRepresentation | None
    initial_states: tuple[Any, ...]
    field: SampledField
    time_grid: TimeGrid
    options: PropagationOptions
    axes: str | None
    split_interaction: Literal["cartesian", "helicity_projected"]
    validate_units: bool
    verbose: bool

    @classmethod
    def from_validated_mapping(
        cls,
        params: Mapping[str, Any],
        *,
        field: SampledField,
        options: PropagationOptions,
    ) -> SimulationCase:
        """Freeze a mapping that already passed simulation validation."""
        basis_type = str(params["basis_type"]).lower()
        representation = (
            LinMolRepresentation(params["representation"])
            if basis_type == "linmol"
            else None
        )
        return cls(
            basis_type=basis_type,
            model_parameters=model_parameters_from_mapping(params),
            representation=representation,
            initial_states=tuple(params["initial_states"]),
            field=field,
            time_grid=field.time_grid,
            options=options,
            axes=params.get("axes"),
            split_interaction=cast(
                Literal["cartesian", "helicity_projected"],
                params.get("split_interaction", "cartesian"),
            ),
            validate_units=params.get("validate_units", True),
            verbose=params.get("verbose", False),
        )

    @property
    def uses_m_average(self) -> bool:
        return self.representation is LinMolRepresentation.M_INCOHERENT_AVERAGE

    @property
    def expects_cartesian_field(self) -> bool:
        return self.representation is LinMolRepresentation.M_RESOLVED


__all__ = ["SimulationCase"]
