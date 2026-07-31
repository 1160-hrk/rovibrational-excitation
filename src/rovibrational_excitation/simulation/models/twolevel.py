"""Two-level-system construction used by the batch runner."""

from __future__ import annotations

from typing import Any

from rovibrational_excitation.core.basis import TwoLevelBasis
from rovibrational_excitation.dipole.twolevel import TwoLevelDipoleMatrix

from .common import build_initial_state


def build_twolevel(params: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    """Build the existing two-level simulation components."""
    basis = TwoLevelBasis(
        energy_gap=params["energy_gap"],
        input_units=params.get("energy_gap_units", "rad/fs"),
        output_units="J",
    )
    state = build_initial_state(basis, params.get("initial_states", [0]))
    hamiltonian = basis.generate_H0()
    dipole = TwoLevelDipoleMatrix(
        basis,
        mu0=params["mu0_Cm"],
        backend=params.get("backend", "numpy"),
    )
    return basis, state, hamiltonian, dipole
