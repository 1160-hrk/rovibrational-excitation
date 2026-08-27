"""Linear-molecule construction used by the batch runner."""

from __future__ import annotations

from typing import Any

from rovibrational_excitation.core.basis import LinMolBasis
from rovibrational_excitation.core.execution import ExecutionPolicy
from rovibrational_excitation.dipole.linmol import LinMolDipoleMatrix

from .common import build_initial_state
from .parameters import LinMolParameters
from .validation import (
    LinMolRepresentation,
    model_parameters_from_mapping,
    validate_linmol_representation,
)


def build_linmol(
    params: dict[str, Any], *, execution_policy: ExecutionPolicy
) -> tuple[Any, Any, Any, Any]:
    """Build basis, initial state, Hamiltonian, and dipole without changing formulas."""
    representation = validate_linmol_representation(params)
    model_params = model_parameters_from_mapping(params)
    if not isinstance(model_params, LinMolParameters):
        raise TypeError("linmol builder requires LinMolParameters")
    if representation is not LinMolRepresentation.M_RESOLVED:
        raise ValueError(
            "representation=m_incoherent_average is a multi-block workflow "
            "and cannot be built as one pure-state model; use the simulation runner"
        )
    basis = LinMolBasis(
        model_params.v_max,
        model_params.j_max,
        use_M=True,
        omega=model_params.vibrational_frequency.angular_rad_per_fs,
        delta_omega=model_params.anharmonic_shift.angular_rad_per_fs,
        B=model_params.rotational_constant.angular_rad_per_fs,
        alpha=model_params.vibration_rotation_coupling.angular_rad_per_fs,
        output_units="J",
        input_units="rad/fs",
    )
    state = build_initial_state(basis, params["initial_states"])

    potential_type = model_params.potential_type

    hamiltonian = basis.generate_H0()
    dipole = LinMolDipoleMatrix(
        basis,
        mu0=model_params.dipole_c_m,
        potential_type=potential_type,
        backend=execution_policy.backend.value,
        dense=execution_policy.dense,
    )
    return basis, state, hamiltonian, dipole
