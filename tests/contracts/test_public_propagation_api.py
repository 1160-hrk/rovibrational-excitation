"""Contracts for the explicit public propagation boundary."""

import inspect

import numpy as np
import pytest

from rovibrational_excitation.core.execution import (
    ArrayBackend,
    ExecutionPolicy,
    MatrixStorage,
)
from rovibrational_excitation.core.propagation import (
    LiouvillePropagator,
    MixedStatePropagator,
    SchrodingerPropagator,
)
from rovibrational_excitation.core.propagation.capabilities import (
    PropagationAlgorithm,
)
from rovibrational_excitation.core.propagation.options import (
    PropagationOptions,
    RenormalizationPolicy,
    ScalingMode,
)
from rovibrational_excitation.core.states import DensityState, PureState


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
    assert signature.parameters["options"].default is inspect.Parameter.empty
    assert signature.parameters["coupling_mode"].default is inspect.Parameter.empty


def test_schrodinger_public_boundary_projects_typed_options_without_change(monkeypatch):
    solver = SchrodingerPropagator(validate_units=False)
    options = _options()
    captured = {}

    def fake_array_propagation(*args, **kwargs):
        captured.update(kwargs)
        return args[3]

    monkeypatch.setattr(solver, "_propagate_array", fake_array_propagation)
    state = PureState(np.array([1.0, 0.0], dtype=np.complex128))

    result = solver.propagate(
        None,
        None,
        None,
        state,
        options=options,
        coupling_mode="scalar",
        coupling_axis="z",
        return_times=True,
    )

    assert result is state.amplitudes
    assert captured == {
        "return_traj": False,
        "return_time_psi": True,
        "sample_stride": 3,
        "nondimensional": True,
        "coupling_mode": "scalar",
        "coupling_axis": "z",
        "verbose": False,
        "algorithm": "rk4",
        "sparse": False,
        "renorm": False,
        "direction": solver.propagate.__kwdefaults__["direction"],
    }


@pytest.mark.parametrize(
    ("coupling_mode", "axes", "coupling_axis", "message"),
    [
        ("cartesian", None, None, "axes is required"),
        ("scalar", "xy", "z", "axes is not applicable"),
        ("scalar", None, None, "coupling_axis must be"),
    ],
)
def test_public_boundary_rejects_ambiguous_coupling(
    coupling_mode, axes, coupling_axis, message
):
    solver = SchrodingerPropagator(validate_units=False)

    with pytest.raises(ValueError, match=message):
        solver.propagate(
            None,
            None,
            None,
            PureState(np.array([1.0, 0.0])),
            options=_options(scaling=ScalingMode.DIMENSIONAL),
            coupling_mode=coupling_mode,
            axes=axes,
            coupling_axis=coupling_axis,
        )


def test_liouville_public_boundary_rejects_incompatible_options_before_work():
    solver = LiouvillePropagator(validate_units=False)

    with pytest.raises(ValueError, match="algorithm"):
        solver.propagate(
            None,
            None,
            None,
            DensityState(np.eye(2) / 2.0),
            options=_options(algorithm=PropagationAlgorithm.SPLIT_OPERATOR),
            coupling_mode="cartesian",
            axes="xy",
        )
