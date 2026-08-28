"""Two-level-system construction used by the batch runner."""

from __future__ import annotations

from typing import Any

from rovibrational_excitation.core.basis import TwoLevelBasis
from rovibrational_excitation.core.execution import ExecutionPolicy
from rovibrational_excitation.dipole.twolevel import TwoLevelDipoleMatrix

from .common import build_initial_state
from .parameters import TwoLevelParameters
from .validation import model_parameters_from_mapping


def build_twolevel(
    params: dict[str, Any], *, execution_policy: ExecutionPolicy
) -> tuple[Any, Any, Any, Any]:
    """Build the existing two-level simulation components."""
    model_params = model_parameters_from_mapping(params)
    if not isinstance(model_params, TwoLevelParameters):
        raise TypeError("twolevel builder requires TwoLevelParameters")
    return build_twolevel_from_parameters(
        model_params,
        params["initial_states"],
        execution_policy=execution_policy,
    )


def build_twolevel_from_parameters(
    model_params: TwoLevelParameters,
    initial_states: Any,
    *,
    execution_policy: ExecutionPolicy,
) -> tuple[Any, Any, Any, Any]:
    """Build from the frozen schema without re-reading a configuration mapping."""
    basis = TwoLevelBasis(
        energy_gap=model_params.energy_gap,
        input_units=model_params.energy_gap_units,
        output_units="J",
    )
    state = build_initial_state(basis, initial_states)
    hamiltonian = basis.generate_H0()
    dipole = TwoLevelDipoleMatrix(
        basis,
        mu0=model_params.dipole_c_m,
        backend=execution_policy.backend.value,
        dense=execution_policy.dense,
    )
    return basis, state, hamiltonian, dipole
