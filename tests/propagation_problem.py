"""Explicit propagation-problem builders used only by tests."""

from __future__ import annotations

from typing import Any

import numpy as np

from rovibrational_excitation.core.basis import TwoLevelBasis
from rovibrational_excitation.core.propagation import (
    Axis,
    CouplingSpec,
    PropagationProblem,
    SystemModel,
)
from rovibrational_excitation.core.states import (
    DensityState,
    IncoherentEnsemble,
    PureState,
)
from rovibrational_excitation.core.time import TimeGrid
from rovibrational_excitation.dipole import create_dipole_matrix
from rovibrational_excitation.fields import ElectricField

PropagationState = PureState | IncoherentEnsemble | DensityState


def propagation_problem(
    initial_state: PropagationState,
    *,
    hamiltonian: Any | None = None,
    field: ElectricField | None = None,
    dipole: Any | None = None,
    coupling: CouplingSpec | None = None,
) -> PropagationProblem:
    """Build a complete two-level problem without implicit production choices."""
    basis = getattr(dipole, "basis", None)
    if basis is None:
        basis = TwoLevelBasis(
            energy_gap=1.0,
            input_units="rad/fs",
            output_units="rad/fs",
        )
    if hamiltonian is None:
        hamiltonian = basis.generate_H0()
    if dipole is None:
        dipole = create_dipole_matrix(basis, mu0=1.0e-30)
    if coupling is None:
        coupling = CouplingSpec.cartesian("xy")

    if field is None:
        time_grid = TimeGrid.from_bounds(0.0, 0.2, 0.1)
        field = ElectricField.from_time_grid(time_grid)
    else:
        time_grid = TimeGrid(np.asarray(field.tlist))

    return PropagationProblem(
        model=SystemModel(
            name="test-twolevel",
            basis=basis,
            hamiltonian=hamiltonian,
            dipole=dipole,
            coupling=coupling,
            metadata={},
        ),
        field=field,
        time_grid=time_grid,
        initial_state=initial_state,
    )


def scalar_propagation_problem(
    initial_state: PropagationState,
    *,
    axis: Axis = Axis.Z,
    hamiltonian: Any | None = None,
    field: ElectricField | None = None,
    dipole: Any | None = None,
) -> PropagationProblem:
    """Build a complete scalar-coupling test problem."""
    return propagation_problem(
        initial_state,
        hamiltonian=hamiltonian,
        field=field,
        dipole=dipole,
        coupling=CouplingSpec.scalar(axis),
    )


__all__ = ["propagation_problem", "scalar_propagation_problem"]
