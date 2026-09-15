"""Vibrational-ladder construction used by the batch runner."""

from __future__ import annotations

from typing import Any

from rovibrational_excitation.core.execution import ExecutionPolicy

from ..common import build_initial_state
from ..parameters import VibLadderParameters
from ..validation import model_parameters_from_mapping
from .basis import VibLadderBasis
from .dipole import VibLadderDipoleMatrix


def build_vibladder(
    params: dict[str, Any], *, execution_policy: ExecutionPolicy
) -> tuple[Any, Any, Any, Any]:
    """Build the existing vibrational-ladder simulation components."""
    model_params = model_parameters_from_mapping(params)
    if not isinstance(model_params, VibLadderParameters):
        raise TypeError("vibladder builder requires VibLadderParameters")
    return build_vibladder_from_parameters(
        model_params,
        params["initial_states"],
        execution_policy=execution_policy,
    )


def build_vibladder_from_parameters(
    model_params: VibLadderParameters,
    initial_states: Any,
    *,
    execution_policy: ExecutionPolicy,
) -> tuple[Any, Any, Any, Any]:
    """Build from the frozen schema without re-reading a configuration mapping."""
    basis, hamiltonian, dipole = build_vibladder_operators_from_parameters(
        model_params,
        execution_policy=execution_policy,
    )
    state = build_initial_state(basis, initial_states)
    return basis, state, hamiltonian, dipole


def build_vibladder_operators_from_parameters(
    model_params: VibLadderParameters,
    *,
    execution_policy: ExecutionPolicy,
) -> tuple[Any, Any, Any]:
    """Build basis and operators without imposing a workflow's state semantics."""
    basis = VibLadderBasis(
        model_params.v_max,
        omega=model_params.vibrational_frequency.angular_rad_per_fs,
        delta_omega=model_params.anharmonic_shift.angular_rad_per_fs,
    )
    hamiltonian = basis.generate_H0()
    dipole = VibLadderDipoleMatrix(
        basis,
        mu0=model_params.dipole_c_m,
        potential_type=model_params.potential_type,
        backend=execution_policy.backend.value,
        dense=execution_policy.dense,
    )
    return basis, hamiltonian, dipole
