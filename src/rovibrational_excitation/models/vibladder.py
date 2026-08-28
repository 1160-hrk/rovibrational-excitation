"""Vibrational-ladder construction used by the batch runner."""

from __future__ import annotations

from typing import Any

from rovibrational_excitation.core.basis import VibLadderBasis
from rovibrational_excitation.core.execution import ExecutionPolicy
from rovibrational_excitation.dipole.viblad import VibLadderDipoleMatrix

from .common import build_initial_state
from .parameters import VibLadderParameters
from .validation import model_parameters_from_mapping


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
    basis = VibLadderBasis(
        model_params.v_max,
        omega=model_params.vibrational_frequency.angular_rad_per_fs,
        delta_omega=model_params.anharmonic_shift.angular_rad_per_fs,
    )
    state = build_initial_state(basis, initial_states)
    hamiltonian = basis.generate_H0()
    dipole = VibLadderDipoleMatrix(
        basis,
        mu0=model_params.dipole_c_m,
        potential_type=model_params.potential_type,
        backend=execution_policy.backend.value,
        dense=execution_policy.dense,
    )
    return basis, state, hamiltonian, dipole
