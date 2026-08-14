"""Explicit single-state dispatch contracts for propagation facades."""

import numpy as np
import pytest

from rovibrational_excitation.core.propagation import (
    LiouvillePropagator,
    SchrodingerPropagator,
)
from rovibrational_excitation.core.states import DensityState, PureState
from tests.propagation_options import propagation_options


def test_schrodinger_rejects_raw_array_before_unit_validation():
    solver = SchrodingerPropagator()

    with pytest.raises(TypeError, match="initial_state must be a PureState"):
        solver.propagate(
            None,
            None,
            None,
            np.array([1.0]),
            options=propagation_options(return_trajectory=False),
            coupling_mode="cartesian",
            axes="xy",
        )


def test_schrodinger_unwraps_pure_state_without_copy_or_repair(monkeypatch):
    state = PureState(np.array([1.0, 0.0], dtype=np.complex128))
    captured = {}
    solver = SchrodingerPropagator(validate_units=False)

    def fake_array_propagation(*args, **kwargs):
        captured["array"] = args[3]
        return args[3]

    monkeypatch.setattr(solver, "_propagate_array", fake_array_propagation)

    result = solver.propagate(
        None,
        None,
        None,
        state,
        options=propagation_options(return_trajectory=False),
        coupling_mode="cartesian",
        axes="xy",
    )

    assert captured["array"] is state.amplitudes
    assert result is state.amplitudes


def test_liouville_rejects_raw_array_before_unit_validation():
    solver = LiouvillePropagator()

    with pytest.raises(TypeError, match="initial_state must be a DensityState"):
        solver.propagate(
            None,
            None,
            None,
            np.eye(2) / 2.0,
            options=propagation_options(return_trajectory=False),
            coupling_mode="cartesian",
            axes="xy",
        )


def test_liouville_unwraps_density_state_without_copy_or_repair(monkeypatch):
    state = DensityState(np.diag([0.25, 0.75]))
    captured = {}
    solver = LiouvillePropagator(validate_units=False)

    def fake_array_propagation(*args, **kwargs):
        captured["array"] = args[3]
        return args[3]

    monkeypatch.setattr(solver, "_propagate_array", fake_array_propagation)

    result = solver.propagate(
        None,
        None,
        None,
        state,
        options=propagation_options(return_trajectory=False),
        coupling_mode="cartesian",
        axes="xy",
    )

    assert captured["array"] is state.matrix
    assert result is state.matrix
