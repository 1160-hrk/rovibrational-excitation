"""Construction of the production rigid parallel-band SymTop model."""

from __future__ import annotations

from typing import Any

from rovibrational_excitation.core.execution import ArrayBackend, ExecutionPolicy
from rovibrational_excitation.models.common import build_initial_state
from rovibrational_excitation.models.parameters import SymmetricTopParameters

from .basis import SymmetricTopBasis
from .dipole import (
    SymmetricTopDipoleMatrix,
    validate_symmetric_top_vibrational_basis,
)


def build_symmetric_top_from_parameters(
    parameters: SymmetricTopParameters,
    initial_states: Any,
    *,
    execution_policy: ExecutionPolicy,
) -> tuple[Any, Any, Any, Any]:
    """Build one validated NumPy dense/CSR rigid symmetric-top model."""
    if execution_policy.backend is not ArrayBackend.NUMPY:
        raise ValueError("SymTop currently supports only backend='numpy'")
    validate_symmetric_top_vibrational_basis(parameters)
    basis = SymmetricTopBasis(parameters)
    state = build_initial_state(basis, initial_states)
    hamiltonian = basis.generate_H0()
    dipole = SymmetricTopDipoleMatrix(
        basis,
        mu0=parameters.dipole_c_m,
        potential_type=parameters.potential_type,
        backend="numpy",
        dense=execution_policy.dense,
    )
    return basis, state, hamiltonian, dipole


__all__ = ["build_symmetric_top_from_parameters"]
