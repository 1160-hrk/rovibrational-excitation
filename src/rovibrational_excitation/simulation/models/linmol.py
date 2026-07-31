"""Linear-molecule construction used by the batch runner."""

from __future__ import annotations

from typing import Any

from rovibrational_excitation.core.basis import LinMolBasis
from rovibrational_excitation.dipole.linmol import LinMolDipoleMatrix

from .common import build_initial_state


def build_linmol(params: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    """Build basis, initial state, Hamiltonian, and dipole without changing formulas."""
    basis = LinMolBasis(
        params["V_max"],
        params["J_max"],
        use_M=params.get("use_M", True),
        omega=params["omega_rad_phz"],
        delta_omega=params.get("delta_omega_rad_phz", 0.0),
        B=params.get("B_rad_phz", 0.0),
        alpha=params.get("alpha_rad_phz", 0.0),
        output_units="J",
        input_units="rad/fs",
    )
    state = build_initial_state(basis, params.get("initial_states", [0]))

    delta_omega = params.get("delta_omega_rad_phz", 0.0)
    potential_type = params.get("potential_type", "harmonic")
    if potential_type == "morse" and delta_omega == 0.0:
        raise ValueError(
            "delta_omega_rad_phz must be non-zero when potential_type='morse'"
        )

    hamiltonian = basis.generate_H0()
    dipole = LinMolDipoleMatrix(
        basis,
        mu0=params["mu0_Cm"],
        potential_type=potential_type,
        backend=params.get("backend", "numpy"),
        dense=params.get("dense", True),
    )
    return basis, state, hamiltonian, dipole
