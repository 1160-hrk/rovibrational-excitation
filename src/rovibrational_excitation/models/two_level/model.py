"""Two-level-system construction used by the batch runner."""

from __future__ import annotations

from typing import Any

from rovibrational_excitation.core.execution import ExecutionPolicy

from ..common import build_initial_state
from .basis import TwoLevelBasis
from .dipole import TwoLevelDipoleMatrix
from .parameters import TwoLevelParameters


def build_twolevel_from_parameters(
    model_params: TwoLevelParameters,
    initial_states: Any,
    *,
    execution_policy: ExecutionPolicy,
) -> tuple[Any, Any, Any, Any]:
    """Build from the frozen schema without re-reading a configuration mapping."""
    basis, hamiltonian, dipole = build_twolevel_operators_from_parameters(
        model_params,
        execution_policy=execution_policy,
    )
    state = build_initial_state(basis, initial_states)
    return basis, state, hamiltonian, dipole


def build_twolevel_operators_from_parameters(
    model_params: TwoLevelParameters,
    *,
    execution_policy: ExecutionPolicy,
) -> tuple[Any, Any, Any]:
    """Build basis and operators without imposing a workflow's state semantics."""
    basis = TwoLevelBasis(
        energy_gap=model_params.energy_gap,
        input_units=model_params.energy_gap_units,
        output_units="J",
    )
    hamiltonian = basis.generate_H0()
    dipole = TwoLevelDipoleMatrix(
        basis,
        mu0=model_params.dipole_c_m,
        backend=execution_policy.backend.value,
        dense=execution_policy.dense,
    )
    return basis, hamiltonian, dipole
