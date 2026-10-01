"""Explicit, non-inferential quantum-state input contracts."""

from __future__ import annotations

import numpy as np
import pytest

from rovibrational_excitation.core.states import (
    DensityState,
    IncoherentEnsemble,
    PureState,
)


def test_pure_state_owns_a_read_only_normalized_copy() -> None:
    source = np.array([np.sqrt(0.25), 1j * np.sqrt(0.75)])
    state = PureState(source)
    source[:] = 0.0

    np.testing.assert_array_equal(
        state.amplitudes,
        np.array([np.sqrt(0.25), 1j * np.sqrt(0.75)]),
    )
    assert state.dimension == 2
    assert state.amplitudes.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        state.amplitudes[0] = 0.0


@pytest.mark.parametrize(
    ("amplitudes", "message"),
    [
        (np.array([]), "nonempty one-dimensional"),
        (np.eye(2), "nonempty one-dimensional"),
        (np.array([1.0, np.nan]), "finite"),
        (np.array([0.0, 0.0]), "norm one"),
        (np.array([2.0, 0.0]), "norm one"),
    ],
)
def test_pure_state_rejects_nonphysical_input(amplitudes, message) -> None:
    with pytest.raises(ValueError, match=message):
        PureState(amplitudes)


def test_incoherent_ensemble_preserves_norm_squared_weight_semantics() -> None:
    phase = np.exp(0.37j)
    ensemble = IncoherentEnsemble(
        [
            np.sqrt(0.25) * np.array([1.0, 0.0]),
            np.sqrt(0.75) * phase * np.array([0.0, 1.0]),
        ]
    )

    np.testing.assert_allclose(ensemble.weights, [0.25, 0.75], atol=1.0e-15)
    np.testing.assert_allclose(
        ensemble.density_matrix(), np.diag([0.25, 0.75]), atol=1.0e-15
    )
    assert ensemble.dimension == 2
    assert ensemble.weights.flags.writeable is False
    assert all(state.amplitudes.flags.writeable is False for state in ensemble.states)


def test_incoherent_ensemble_skips_zero_vectors_without_changing_other_weights() -> (
    None
):
    ensemble = IncoherentEnsemble([np.zeros(2), np.sqrt(2.0) * np.array([1.0, 0.0])])

    assert len(ensemble.states) == 1
    np.testing.assert_array_equal(ensemble.weights, [1.0])
    np.testing.assert_array_equal(ensemble.states[0].amplitudes, [1.0, 0.0])


@pytest.mark.parametrize(
    ("vectors", "message"),
    [
        ([], "must not be empty"),
        ([np.zeros(2)], "non-zero norm"),
        ([np.ones(2), np.ones(3)], "same dimension"),
        ([np.eye(2)], "one-dimensional"),
        ([np.array([1.0, np.inf])], "finite"),
        ([np.array([1.0e308, 0.0])], "weights must be finite"),
    ],
)
def test_incoherent_ensemble_rejects_ambiguous_input(vectors, message) -> None:
    with pytest.raises(ValueError, match=message):
        IncoherentEnsemble(vectors)


def test_density_state_requires_trace_one_and_never_repairs_input() -> None:
    source = np.array([[0.4, 0.1j], [-0.1j, 0.6]], dtype=np.complex128)
    state = DensityState(source)
    source[:] = 0.0

    np.testing.assert_array_equal(
        state.matrix,
        np.array([[0.4, 0.1j], [-0.1j, 0.6]], dtype=np.complex128),
    )
    assert state.dimension == 2
    assert state.matrix.flags.writeable is False

    with pytest.raises(ValueError, match="trace one"):
        DensityState(np.eye(2, dtype=np.complex128))

    with pytest.raises(ValueError, match="nonempty square"):
        DensityState(np.empty((0, 0), dtype=np.complex128))


def test_density_state_from_pure_state_is_explicit_and_exact() -> None:
    pure = PureState(np.array([1.0, 1.0j]) / np.sqrt(2.0))
    density = DensityState.from_pure_state(pure)

    np.testing.assert_array_equal(
        density.matrix,
        np.outer(pure.amplitudes, pure.amplitudes.conj()),
    )
