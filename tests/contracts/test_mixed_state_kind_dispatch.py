"""Explicit state-kind dispatch contracts for mixed-state propagation."""

import numpy as np
import pytest

from rovibrational_excitation.core.propagation import MixedStatePropagator
from rovibrational_excitation.core.propagation.liouville import LiouvillePropagator
from rovibrational_excitation.core.states import DensityState, IncoherentEnsemble


@pytest.mark.parametrize(
    "raw_state",
    [
        [np.array([1.0, 0.0], dtype=np.complex128)],
        np.eye(2, dtype=np.complex128) / 2.0,
    ],
)
def test_mixed_state_rejects_untyped_initial_state(raw_state):
    solver = MixedStatePropagator()

    with pytest.raises(
        TypeError,
        match="initial_state must be an IncoherentEnsemble or DensityState",
    ):
        solver.propagate(None, None, None, raw_state, return_traj=False)


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
        lambda *args, **kwargs: np.asarray(args[3], dtype=np.complex128),
    )

    density = solver.propagate(None, None, None, ensemble, return_traj=False)

    np.testing.assert_array_equal(density, np.diag([0.8, 0.2]))


def test_mixed_state_unwraps_density_without_repair(monkeypatch):
    matrix = np.array([[0.25, 0.1j], [-0.1j, 0.75]], dtype=np.complex128)
    density_state = DensityState(matrix)
    captured = {}

    def fake_propagate(self, *args, **kwargs):
        captured["matrix"] = args[3]
        return args[3]

    monkeypatch.setattr(LiouvillePropagator, "_propagate_array", fake_propagate)

    result = MixedStatePropagator(validate_units=False).propagate(
        None,
        None,
        None,
        density_state,
        return_traj=False,
    )

    np.testing.assert_array_equal(captured["matrix"], matrix)
    np.testing.assert_array_equal(result, matrix)
