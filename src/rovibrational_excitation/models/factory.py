"""Dispatch from a configured basis type to its construction function."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rovibrational_excitation.core.execution import ExecutionPolicy
from rovibrational_excitation.dynamics.problem import (
    Axis,
    CouplingSpec,
    SystemModel,
)

from .linmol import build_linmol_from_parameters
from .parameters import LinMolParameters, TwoLevelParameters, VibLadderParameters
from .twolevel import build_twolevel_from_parameters
from .validation import (
    LinMolRepresentation,
    ModelConfigurationError,
    ModelParameters,
    model_parameters_from_mapping,
    validate_linmol_representation,
    validate_model_parameters,
)
from .vibladder import build_vibladder_from_parameters


@dataclass(frozen=True)
class ModelComponents:
    name: str
    basis: Any
    state: Any
    hamiltonian: Any
    dipole: Any
    coupling: CouplingSpec

    def to_system_model(self) -> SystemModel:
        """Project the temporary builder result to the typed model boundary."""
        return SystemModel(
            name=self.name,
            basis=self.basis,
            hamiltonian=self.hamiltonian,
            dipole=self.dipole,
            coupling=self.coupling,
            metadata={},
        )


def build_model(
    params: dict[str, Any], *, execution_policy: ExecutionPolicy
) -> ModelComponents:
    """Build a configured model using the same dispatch as the existing runner."""
    basis_type = validate_model_parameters(params)
    if basis_type == "linmol":
        representation = validate_linmol_representation(params)
        if representation is not LinMolRepresentation.M_RESOLVED:
            raise ValueError(
                "representation=m_incoherent_average is a multi-block workflow "
                "and cannot be built as one pure-state model; use the simulation runner"
            )
        if "axes" not in params:
            raise ModelConfigurationError(
                "Missing required LinMol m_resolved parameter: axes"
            )
    else:
        representation = None
    return build_model_from_parameters(
        model_parameters_from_mapping(params),
        initial_states=params["initial_states"],
        representation=representation,
        axes=params.get("axes"),
        execution_policy=execution_policy,
    )


def build_model_from_parameters(
    model_parameters: ModelParameters,
    *,
    initial_states: Any,
    representation: LinMolRepresentation | None,
    axes: str | None,
    execution_policy: ExecutionPolicy,
) -> ModelComponents:
    """Build a model from one frozen parameter schema."""
    if isinstance(model_parameters, LinMolParameters):
        if representation is not LinMolRepresentation.M_RESOLVED:
            raise ValueError(
                "representation=m_incoherent_average is a multi-block workflow "
                "and cannot be built as one pure-state model; use the simulation runner"
            )
        if axes is None:
            raise ModelConfigurationError(
                "Missing required LinMol m_resolved parameter: axes"
            )
        try:
            coupling = CouplingSpec.cartesian(axes)
        except (TypeError, ValueError) as exc:
            raise ModelConfigurationError(str(exc)) from exc
        basis, state, hamiltonian, dipole = build_linmol_from_parameters(
            model_parameters,
            initial_states,
            execution_policy=execution_policy,
        )
        basis_type = "linmol"
    elif isinstance(model_parameters, TwoLevelParameters):
        coupling = CouplingSpec.scalar(Axis.X)
        basis, state, hamiltonian, dipole = build_twolevel_from_parameters(
            model_parameters,
            initial_states,
            execution_policy=execution_policy,
        )
        basis_type = "twolevel"
    elif isinstance(model_parameters, VibLadderParameters):
        coupling = CouplingSpec.scalar(Axis.Z)
        basis, state, hamiltonian, dipole = build_vibladder_from_parameters(
            model_parameters,
            initial_states,
            execution_policy=execution_policy,
        )
        basis_type = "vibladder"
    else:
        raise TypeError("unsupported model parameter schema")

    return ModelComponents(
        name=basis_type,
        basis=basis,
        state=state,
        hamiltonian=hamiltonian,
        dipole=dipole,
        coupling=coupling,
    )
