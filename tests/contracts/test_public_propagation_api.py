"""Contractsdef for the explicit public propagation boundary."""

import inspect

import numpy as np
import pytest

from rovibrational_excitation.core.execution import (
    ArrayBackend,
    ExecutionPolicy,
    MatrixStorage,
)
from rovibrational_excitation.core.states import DensityState, PureState
from rovibrational_excitation.dynamics import (
    Axis,
    CouplingSpec,
    LiouvillePropagator,
    MixedStatePropagator,
    SchrodingerPropagator,
)
from rovibrational_excitation.dynamics.capabilities import (
    PropagationAlgorithm,
)
from rovibrational_excitation.dynamics.options import (
    PropagationOptions,
    RenormalizationPolicy,
    ScalingMode,
)
from tests.propagation_problem import (
    propagation_problem,
    scalar_propagation_problem,
)


def _options(
    *,
    algorithm=PropagationAlgorithm.RK4,
    storage=MatrixStorage.DENSE,
    trajectory=False,
    stride=3,
    scaling=ScalingMode.NONDIMENSIONAL,
    renormalization=RenormalizationPolicy.DISABLED,
):
    return PropagationOptions(
        algorithm=algorithm,
        execution=ExecutionPolicy(
            backend=ArrayBackend.NUMPY,
            storage=storage,
        ),
        return_trajectory=trajectory,
        sample_stride=stride,
        scaling=scaling,
        renormalization=renormalization,
    )


@pytest.mark.parametrize(
    "solver_type",
    [SchrodingerPropagator, LiouvillePropagator, MixedStatePropagator],
)
def test_public_propagate_has_no_unrestricted_keyword_arguments(solver_type):
    signature = inspect.signature(solver_type.propagate)

    assert all(
        parameter.kind is not inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    assert signature.parameters["problem"].default is inspect.Parameter.empty
    assert signature.parameters["options"].default is inspect.Parameter.empty
    assert "coupling_mode" not in signature.parameters
    assert "axes" not in signature.parameters
    assert "coupling_axis" not in signature.parameters


def test_schrodinger_public_boundary_projects_typed_options_without_change(monkeypatch):
    solver = SchrodingerPropagator(validate_units=False)
    options = _options()
    captured = {}

    def fake_array_propagation(*args, **kwargs):
        captured.update(kwargs)
        return np.array([0.2]), args[3], None

    monkeypatch.setattr(solver, "_propagate_array", fake_array_propagation)
    state = PureState(np.array([1.0, 0.0], dtype=np.complex128))

    problem = scalar_propagation_problem(state, axis=Axis.Z)
    result = solver.propagate(
        problem,
        options=options,
    )

    assert result.state is state.amplitudes
    assert captured == {
        "return_traj": False,
        "return_time_psi": True,
        "sample_stride": 1,
        "_return_context": True,
        "nondimensional": True,
        "coupling_mode": "scalar",
        "coupling_axis": "z",
        "verbose": False,
        "algorithm": "rk4",
        "sparse": False,
        "renorm": False,
        "direction": solver.propagate.__kwdefaults__["direction"],
    }


def test_problem_owns_the_only_coupling_configuration():
    state = PureState(np.array([1.0, 0.0]))
    problem = propagation_problem(
        state,
        coupling=CouplingSpec.cartesian("zx"),
    )

    assert problem.coupling_kwargs == {"axes": "zx"}


def test_liouville_public_boundary_rejects_incompatible_options_before_work():
    solver = LiouvillePropagator(validate_units=False)

    with pytest.raises(ValueError, match="algorithm"):
        solver.propagate(
            propagation_problem(DensityState(np.eye(2) / 2.0)),
            options=_options(algorithm=PropagationAlgorithm.SPLIT_OPERATOR),
        )
