"""Vibrational-ladder construction used by the batch runner."""

from __future__ import annotations

from typing import Any

from rovibrational_excitation.core.basis import VibLadderBasis
from rovibrational_excitation.dipole.viblad import VibLadderDipoleMatrix

from .common import build_initial_state


def build_vibladder(params: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    """Build the existing vibrational-ladder simulation components."""
    basis = VibLadderBasis(
        params["V_max"],
        omega=params["omega_rad_phz"],
        delta_omega=params.get("delta_omega_rad_phz", 0.0),
    )
    state = build_initial_state(basis, params.get("initial_states", [0]))
    hamiltonian = basis.generate_H0()
    dipole = VibLadderDipoleMatrix(
        basis,
        mu0=params["mu0_Cm"],
        potential_type=params.get("potential_type", "harmonic"),
        backend=params.get("backend", "numpy"),
    )
    return basis, state, hamiltonian, dipole
