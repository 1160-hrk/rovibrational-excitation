"""Linear-molecule construction used by the batch runner."""

from __future__ import annotations

from typing import Any

from rovibrational_excitation.core.execution import ExecutionPolicy

from ..common import build_initial_state
from .basis import LinMolBasis
from .dipole import LinMolDipoleMatrix
from .parameters import LinMolParameters


def build_linmol_from_parameters(
    model_params: LinMolParameters,
    initial_states: Any,
    *,
    execution_policy: ExecutionPolicy,
) -> tuple[Any, Any, Any, Any]:
    """Build from the frozen schema without re-reading a configuration mapping."""
    basis, hamiltonian, dipole = build_linmol_operators_from_parameters(
        model_params,
        execution_policy=execution_policy,
    )
    state = build_initial_state(basis, initial_states)
    return basis, state, hamiltonian, dipole


def build_linmol_operators_from_parameters(
    model_params: LinMolParameters,
    *,
    execution_policy: ExecutionPolicy,
) -> tuple[Any, Any, Any]:
    """Build basis and operators without imposing a workflow's state semantics."""
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
    potential_type = model_params.potential_type

    hamiltonian = basis.generate_H0()
    dipole = LinMolDipoleMatrix(
        basis,
        mu0=model_params.dipole_c_m,
        potential_type=potential_type,
        backend=execution_policy.backend.value,
        dense=execution_policy.dense,
    )
    return basis, hamiltonian, dipole
