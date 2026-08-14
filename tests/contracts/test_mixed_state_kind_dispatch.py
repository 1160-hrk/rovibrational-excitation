"""Explicitdef state-kind dispatch contracts for mixed-state propagation."""

import numpy as np
import pytest

from rovibrational_excitation.core.propagation import MixedStatePropagator
from rovibrational_excitation.core.propagation.liouville import LiouvillePropagator
from rovibrational_excitation.core.states import (
    DensityState,
    IncoherentEnsemble,
    PureState,
)
from tests.propagation_options import propagation_options
from tests.propagation_problem import propagation_problem


def test_mixed_state_rejects_pure_initial_state():
    solver = MixedStatePropagator()

    with pytest.raises(
        TypeError,
        match="initial_state must be an IncoherentEnsemble or DensityState",
    ):
        solver.propagate(
            propagation_problem(PureState(np.array([1.0, 0.0], dtype=np.complex128))),
            options=propagation_options(return_trajectory=False),
        )


def test_mixed_state_unwraps_typed_ensemble_without_changing_weights(monkeypatch):
    solver = MixedStatePropagator(validate_units=False)
    ensemble = IncoherentEnsemble(
        [
            2.0 * np.array([1.0, 0.0], dtype=np.complex128),
            np.array([0.0, 1.0], dtype=np.complex128),
        ]
    )

    monkeypatch.setattr(
        solver._schrodinger_prop,
        "_propagate_array",
        lambda *args, **kwargs: (
            np.array([0.2]),
            np.asarray(args[3], dtype=np.complex128),
            None,
        ),
    )

    density = solver.propagate(
        propagation_problem(ensemble),
        options=propagation_options(return_trajectory=False),
    )

    np.testing.assert_array_equal(density.state, np.diag([0.8, 0.2]))


def test_mixed_state_unwraps_density_without_repair(monkeypatch):
    matrix = np.array([[0.25, 0.1j], [-0.1j, 0.75]], dtype=np.complex128)
    density_state = DensityState(matrix)
    captured = {}

    def fake_propagate(self, *args, **kwargs):
        captured["matrix"] = args[3]
        return np.array([0.2]), args[3], None

    monkeypatch.setattr(LiouvillePropagator, "_propagate_array", fake_propagate)

    result = MixedStatePropagator(validate_units=False).propagate(
        propagation_problem(density_state),
        options=propagation_options(return_trajectory=False),
    )

    np.testing.assert_array_equal(captured["matrix"], matrix)
    np.testing.assert_array_equal(result.state, matrix)
