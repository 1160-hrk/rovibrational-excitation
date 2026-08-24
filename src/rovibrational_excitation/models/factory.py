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

from .linmol import build_linmol
from .twolevel import build_twolevel
from .validation import (
    LinMolRepresentation,
    ModelConfigurationError,
    validate_linmol_representation,
    validate_model_parameters,
)
from .vibladder import build_vibladder


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
    builders = {
        "linmol": build_linmol,
        "twolevel": build_twolevel,
        "vibladder": build_vibladder,
    }
    try:
        builder = builders[basis_type]
    except KeyError:
        raise ValueError(f"Unknown basis_type: {basis_type}") from None
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
        try:
            coupling = CouplingSpec.cartesian(params["axes"])
        except (TypeError, ValueError) as exc:
            raise ModelConfigurationError(str(exc)) from exc
    elif basis_type == "twolevel":
        coupling = CouplingSpec.scalar(Axis.X)
    else:
        coupling = CouplingSpec.scalar(Axis.Z)

    basis, state, hamiltonian, dipole = builder(
        params,
        execution_policy=execution_policy,
    )
    return ModelComponents(
        name=basis_type,
        basis=basis,
        state=state,
        hamiltonian=hamiltonian,
        dipole=dipole,
        coupling=coupling,
    )
