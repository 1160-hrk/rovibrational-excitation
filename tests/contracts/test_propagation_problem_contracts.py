"""Contracts for model-owned typed propagation problems."""

import inspect

import numpy as np
import pytest

from rovibrational_excitation.core.basis import TwoLevelBasis
from rovibrational_excitation.core.electric_field import ElectricField
from rovibrational_excitation.core.operators import Hamiltonian
from rovibrational_excitation.core.propagation import (
    Axis,
    CouplingMode,
    CouplingSpec,
    LiouvillePropagator,
    MixedStatePropagator,
    PropagationProblem,
    SchrodingerPropagator,
    SystemModel,
)
from rovibrational_excitation.core.states import PureState
from rovibrational_excitation.core.time import TimeGrid
from rovibrational_excitation.dipole import TwoLevelDipoleMatrix


def _model(*, hamiltonian=None, coupling=None):
    basis = TwoLevelBasis(energy_gap=1.0, input_units="rad/fs")
    return SystemModel(
        name="twolevel",
        basis=basis,
        hamiltonian=hamiltonian or basis.generate_H0(),
        dipole=TwoLevelDipoleMatrix(basis=basis, mu0=1.0e-30),
        coupling=coupling or CouplingSpec.scalar(Axis.X),
        metadata={"source": "contract"},
    )


def _problem(*, model=None, field=None, time_grid=None, initial_state=None):
    grid = time_grid or TimeGrid.from_bounds(0.0, 0.2, 0.1)
    return PropagationProblem(
        model=model or _model(),
        field=field or ElectricField.from_time_grid(grid),
        time_grid=grid,
        initial_state=initial_state
        or PureState(np.array([1.0, 0.0], dtype=np.complex128)),
    )


def test_coupling_spec_has_explicit_exclusive_axis_representation():
    scalar = CouplingSpec.scalar(Axis.Z)
    cartesian = CouplingSpec.cartesian("zx")

    assert scalar.mode is CouplingMode.SCALAR
    assert scalar.scalar_axis is Axis.Z
    assert scalar.cartesian_axes is None
    assert scalar.propagation_kwargs() == {"coupling_axis": "z"}
    assert cartesian.mode is CouplingMode.CARTESIAN
    assert cartesian.scalar_axis is None
    assert cartesian.cartesian_axes == (Axis.Z, Axis.X)
    assert cartesian.propagation_kwargs() == {"axes": "zx"}


def test_coupling_spec_has_no_ambiguous_defaults():
    with pytest.raises(TypeError):
        CouplingSpec()
    with pytest.raises(ValueError, match="scalar_axis"):
        CouplingSpec(
            mode=CouplingMode.SCALAR,
            scalar_axis=None,
            cartesian_axes=None,
        )
    with pytest.raises(ValueError, match="not applicable"):
        CouplingSpec(
            mode=CouplingMode.CARTESIAN,
            scalar_axis=Axis.X,
            cartesian_axes=(Axis.X, Axis.Y),
        )


def test_system_model_validates_dimension_without_building_new_operators():
    with pytest.raises(ValueError, match="Hamiltonian dimension"):
        _model(hamiltonian=Hamiltonian(np.eye(3), units="rad/fs"))


def test_system_model_metadata_is_immutable():
    model = _model()

    with pytest.raises(TypeError):
        model.metadata["source"] = "changed"


def test_problem_requires_exact_field_time_grid_identity():
    grid = TimeGrid.from_bounds(0.0, 0.2, 0.1)
    shifted_grid = TimeGrid.from_bounds(1.0, 1.2, 0.1)

    with pytest.raises(ValueError, match="field time grid"):
        _problem(
            time_grid=grid,
            field=ElectricField.from_time_grid(shifted_grid),
        )


def test_problem_rejects_state_dimension_mismatch():
    state = PureState(np.ones(3, dtype=np.complex128) / np.sqrt(3.0))

    with pytest.raises(ValueError, match="initial-state dimension"):
        _problem(initial_state=state)


def test_problem_keeps_exact_objects_and_reports_state_path():
    model = _model()
    grid = TimeGrid.from_bounds(0.0, 0.2, 0.1)
    field = ElectricField.from_time_grid(grid)
    state = PureState(np.array([1.0, 0.0], dtype=np.complex128))

    problem = _problem(
        model=model,
        field=field,
        time_grid=grid,
        initial_state=state,
    )

    assert problem.model is model
    assert problem.field is field
    assert problem.time_grid is grid
    assert problem.initial_state is state
    assert problem.coupling is model.coupling
    assert problem.coupling_kwargs == {"coupling_axis": "x"}
    assert problem.state_path.value == "pure"


@pytest.mark.parametrize(
    "solver_type",
    [SchrodingerPropagator, LiouvillePropagator, MixedStatePropagator],
)
def test_public_solver_boundary_accepts_one_problem_not_loose_components(solver_type):
    signature = inspect.signature(solver_type.propagate)

    assert list(signature.parameters)[:2] == ["self", "problem"]
    assert "hamiltonian" not in signature.parameters
    assert "efield" not in signature.parameters
    assert "dipole_matrix" not in signature.parameters
    assert "initial_state" not in signature.parameters
    assert "coupling_mode" not in signature.parameters
    assert "axes" not in signature.parameters
    assert "coupling_axis" not in signature.parameters
