"""Contracts for one endpoint-complete typed propagation result."""

from __future__ import annotations

import inspect
from typing import get_type_hints

import numpy as np
import pytest

from rovibrational_excitation.core.propagation import (
    LiouvillePropagator,
    MixedStatePropagator,
    PropagationDirection,
    PropagationResult,
    SchrodingerPropagator,
)
from rovibrational_excitation.core.states import PureState
from rovibrational_excitation.core.time import TimeGrid
from rovibrational_excitation.fields import ElectricField
from tests.propagation_options import propagation_options
from tests.propagation_problem import propagation_problem


class _FakeDeviceArray:
    def __init__(self, host: np.ndarray):
        self._host = host
        self.shape = host.shape
        self.ndim = host.ndim

    @property
    def __cuda_array_interface__(self):
        return {
            "shape": self.shape,
            "typestr": self._host.dtype.str,
            "data": (1, False),
            "version": 3,
        }

    def get(self):
        return self._host.copy()


def test_propagation_result_has_no_defaults():
    with pytest.raises(TypeError):
        PropagationResult()


def test_result_copies_times_and_freezes_metadata_without_copying_state():
    times = np.array([0.0, 0.2])
    state = np.eye(2, dtype=np.complex128)
    metadata = {"model": {"name": "twolevel"}}

    result = PropagationResult(
        times_fs=times,
        state=state,
        state_kind="wavefunction",
        trajectory=True,
        backend="numpy",
        metadata=metadata,
    )

    times[0] = 99.0
    metadata["model"]["name"] = "changed"
    assert result.times_fs[0] == 0.0
    assert result.state is state
    assert result.metadata["model"]["name"] == "twolevel"
    with pytest.raises(TypeError):
        result.metadata["new"] = 1
    with pytest.raises(TypeError):
        result.metadata["model"]["name"] = "changed"


@pytest.mark.parametrize(
    ("state_kind", "trajectory", "state", "message"),
    [
        ("wavefunction", True, np.ones(2), "trajectory state"),
        ("wavefunction", False, np.ones((1, 2)), "final wavefunction"),
        ("density_matrix", True, np.eye(2), "trajectory state"),
        ("density_matrix", False, np.ones((2, 3)), "square"),
    ],
)
def test_result_rejects_state_shapes_that_disagree_with_kind_and_trajectory(
    state_kind, trajectory, state, message
):
    times = np.array([0.0, 0.2]) if trajectory else np.array([0.2])

    with pytest.raises(ValueError, match=message):
        PropagationResult(
            times_fs=times,
            state=state,
            state_kind=state_kind,
            trajectory=trajectory,
            backend="numpy",
            metadata={},
        )


def test_to_numpy_is_the_explicit_device_to_host_boundary():
    device_state = _FakeDeviceArray(np.array([1.0, 0.0], dtype=np.complex128))
    result = PropagationResult(
        times_fs=np.array([0.2]),
        state=device_state,
        state_kind="wavefunction",
        trajectory=False,
        backend="cupy",
        metadata={"execution_backend": "cupy"},
    )

    host = result.to_numpy()

    assert host.backend == "numpy"
    assert isinstance(host.state, np.ndarray)
    np.testing.assert_array_equal(host.state, [1.0, 0.0])
    assert host.metadata["execution_backend"] == "cupy"


@pytest.mark.parametrize(
    "solver_type",
    [SchrodingerPropagator, LiouvillePropagator, MixedStatePropagator],
)
def test_public_solver_signature_has_one_unconditional_result(solver_type):
    signature = inspect.signature(solver_type.propagate)

    assert "return_times" not in signature.parameters
    assert get_type_hints(solver_type.propagate)["return"] is PropagationResult


def test_public_wavefunction_result_applies_stride_and_appends_existing_endpoint(
    monkeypatch,
):
    grid = TimeGrid.from_bounds(0.0, 0.6, 0.1)
    field = ElectricField.from_time_grid(grid)
    problem = propagation_problem(
        PureState(np.array([1.0, 0.0], dtype=np.complex128)),
        field=field,
    )
    solver = SchrodingerPropagator(validate_units=False)
    full_time = np.array([0.0, 0.2, 0.4, 0.6])
    full_state = np.array(
        [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3]],
        dtype=np.complex128,
    )
    captured = {}

    def fake_array(*args, **kwargs):
        captured.update(kwargs)
        return full_time, full_state, None

    monkeypatch.setattr(solver, "_propagate_array", fake_array)

    result = solver.propagate(
        problem,
        options=propagation_options(return_trajectory=True, sample_stride=2),
    )

    assert captured["return_time_psi"] is True
    assert captured["sample_stride"] == 1
    assert captured["_return_context"] is True
    np.testing.assert_array_equal(result.times_fs, [0.0, 0.4, 0.6])
    np.testing.assert_array_equal(result.state, full_state[[0, 2, 3]])
    assert result.state_kind == "wavefunction"
    assert result.trajectory is True
    assert result.backend == "numpy"
    assert result.metadata["sample_stride"] == 2
    assert result.metadata["renormalization"] == "disabled"


def test_stride_one_preserves_the_complete_trajectory_array(monkeypatch):
    problem = propagation_problem(PureState(np.array([1.0, 0.0], dtype=np.complex128)))
    solver = SchrodingerPropagator(validate_units=False)
    full_time = np.array([0.0, 0.2])
    full_state = np.array([[1.0, 0.0], [0.8, 0.2]], dtype=np.complex128)
    monkeypatch.setattr(
        solver,
        "_propagate_array",
        lambda *args, **kwargs: (full_time, full_state, None),
    )

    result = solver.propagate(
        problem,
        options=propagation_options(return_trajectory=True, sample_stride=1),
    )

    assert result.state is full_state


def test_backward_result_reports_descending_exact_endpoint_times(monkeypatch):
    problem = propagation_problem(PureState(np.array([1.0, 0.0], dtype=np.complex128)))
    solver = SchrodingerPropagator(validate_units=False)
    full_state = np.array([[1.0, 0.0], [0.8, 0.2]], dtype=np.complex128)
    monkeypatch.setattr(
        solver,
        "_propagate_array",
        lambda *args, **kwargs: (np.array([0.2, 0.0]), full_state, None),
    )

    result = solver.propagate(
        problem,
        options=propagation_options(return_trajectory=True, sample_stride=1),
        direction=PropagationDirection.BACKWARD,
    )

    np.testing.assert_array_equal(result.times_fs, [0.2, 0.0])
    assert result.state is full_state


def test_public_final_result_always_contains_exact_endpoint(monkeypatch):
    grid = TimeGrid.from_bounds(0.0, 0.6, 0.1)
    field = ElectricField.from_time_grid(grid)
    problem = propagation_problem(
        PureState(np.array([1.0, 0.0], dtype=np.complex128)),
        field=field,
    )
    solver = SchrodingerPropagator(validate_units=False)
    final_state = np.array([0.25, 0.75], dtype=np.complex128)

    monkeypatch.setattr(
        solver,
        "_propagate_array",
        lambda *args, **kwargs: (np.array([0.6]), final_state, None),
    )

    result = solver.propagate(
        problem,
        options=propagation_options(return_trajectory=False, sample_stride=7),
    )

    np.testing.assert_array_equal(result.times_fs, [0.6])
    assert result.state is final_state
    assert result.trajectory is False
