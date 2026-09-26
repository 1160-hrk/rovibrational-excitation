"""Solver-boundary time contracts for GRAPE and Krotov."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import rovibrational_excitation.optimization.grape as grape_module
import rovibrational_excitation.optimization.legacy_batch_overlap as legacy_module
from rovibrational_excitation.dynamics import PropagationDirection


class _TwoStateBasis:
    basis = [(0,), (1,)]

    @staticmethod
    def size() -> int:
        return 2

    @staticmethod
    def get_index(state: tuple[int, ...]) -> int:
        return {(0,): 0, (1,): 1}[tuple(state)]


class _Hamiltonian:
    pass


class _ZeroDipole:
    @staticmethod
    def get_mu_x_SI() -> np.ndarray:
        return np.zeros((2, 2))

    @staticmethod
    def get_mu_y_SI() -> np.ndarray:
        return np.zeros((2, 2))

    @staticmethod
    def get_mu_z_SI() -> np.ndarray:
        return np.zeros((2, 2))


class _DirectionSpy:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def _propagate_array(self, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        assert isinstance(kwargs["initial_state"], np.ndarray)
        efield = kwargs["efield"]
        direction = kwargs.get("direction", PropagationDirection.FORWARD)
        self.calls.append(
            {
                "direction": direction,
                "sample_stride": kwargs["sample_stride"],
                "tlist": np.array(efield.tlist, copy=True),
                "field": np.array(efield.get_Efield(), copy=True),
            }
        )

        propagation_steps = (efield.tlist.size - 1) // 2
        initial = np.asarray(kwargs["initial_state"], dtype=np.complex128)
        trajectory = np.repeat(initial[np.newaxis, :], propagation_steps + 1, axis=0)
        if direction is PropagationDirection.BACKWARD:
            times = efield.tlist[-1] - np.arange(propagation_steps + 1) * 2 * efield.dt
        else:
            times = efield.tlist[0] + np.arange(propagation_steps + 1) * 2 * efield.dt
        return times, trajectory


def _generated_initial_field(**overrides: Any) -> dict[str, Any]:
    params: dict[str, Any] = {
        "control_axes": "xy",
        "initial_field_kind": "generated",
        "initial_duration": 0.2,
        "initial_duration_units": "fs",
        "initial_center": 0.4,
        "initial_center_units": "fs",
        "initial_carrier_frequency": 2300.0,
        "initial_carrier_frequency_units": "cm^-1",
        "initial_amplitude": 1.0e9,
        "initial_amplitude_units": "V/m",
        "initial_polarization": [1.0, 1.0],
    }
    params.update(overrides)
    return params


def test_grape_uses_full_internal_trajectory_and_thins_only_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = _DirectionSpy()
    monkeypatch.setattr(grape_module, "SchrodingerPropagator", lambda **_: spy)

    result = grape_module.run_grape_optimization(
        basis=_TwoStateBasis(),
        hamiltonian=_Hamiltonian(),
        dipole=_ZeroDipole(),
        states={"initial": (0,), "target": (1,)},
        time_cfg={"total_fs": 0.8, "field_dt_fs": 0.1, "output_stride": 3},
        params={
            "max_iter": 0,
            "control_axes": "xy",
            "initial_field_kind": "sampled",
            "initial_field_samples": np.zeros((9, 2)),
            "initial_field_units": "V/m",
        },
    )

    assert [call["sample_stride"] for call in spy.calls] == [1]
    assert [call["direction"] for call in spy.calls] == [PropagationDirection.FORWARD]
    np.testing.assert_array_equal(
        result["time"], np.arange(5, dtype=float)[[0, 3, 4]] * 0.2
    )
    assert result["psi_traj"].shape == (3, 2)


def test_krotov_uses_explicit_backward_direction_on_ascending_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = _DirectionSpy()
    monkeypatch.setattr(legacy_module, "SchrodingerPropagator", lambda **_: spy)

    result = legacy_module.run_legacy_batch_overlap_optimization(
        basis=_TwoStateBasis(),
        hamiltonian=_Hamiltonian(),
        dipole=_ZeroDipole(),
        states={"initial": (0,), "target": (1,)},
        time_cfg={"total_fs": 0.8, "field_dt_fs": 0.1, "output_stride": 3},
        params=_generated_initial_field(initial_amplitude=0.0, max_iter=1),
    )

    assert [call["sample_stride"] for call in spy.calls] == [1, 1, 1]
    assert [call["direction"] for call in spy.calls] == [
        PropagationDirection.FORWARD,
        PropagationDirection.BACKWARD,
        PropagationDirection.FORWARD,
    ]
    for call in spy.calls:
        assert np.all(np.diff(call["tlist"]) > 0.0)
    np.testing.assert_array_equal(spy.calls[1]["field"], spy.calls[0]["field"])
    np.testing.assert_array_equal(
        result["time"], np.arange(5, dtype=float)[[0, 3, 4]] * 0.2
    )
    assert result["psi_traj"].shape == (3, 2)


def test_krotov_generated_initial_field_preserves_frozen_legacy_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = _DirectionSpy()
    monkeypatch.setattr(legacy_module, "SchrodingerPropagator", lambda **_: spy)

    result = legacy_module.run_legacy_batch_overlap_optimization(
        basis=_TwoStateBasis(),
        hamiltonian=_Hamiltonian(),
        dipole=_ZeroDipole(),
        states={"initial": (0,), "target": (1,)},
        time_cfg={"total_fs": 0.8, "field_dt_fs": 0.1, "output_stride": 1},
        params=_generated_initial_field(max_iter=0),
    )

    expected = np.array(
        [
            [1.0627984523004956e4, 1.0627984523004956e4],
            [1.3694193539024193e6, 1.3694193539024193e6],
            [4.4028375516281895e7, 4.4028375516281895e7],
            [3.5322163832979065e8, 3.5322163832979065e8],
            [7.0710678118654740e8, 7.0710678118654740e8],
            [3.5322163832979065e8, 3.5322163832979065e8],
            [4.4028375516281806e7, 4.4028375516281806e7],
            [1.3694193539024722e6, 1.3694193539024722e6],
            [1.0627984523092236e4, 1.0627984523092236e4],
        ]
    )
    np.testing.assert_allclose(result["field_data"], expected, rtol=2e-15, atol=0.0)
    np.testing.assert_array_equal(spy.calls[0]["field"], result["field_data"])


def test_krotov_sampled_initial_field_is_consumed_without_resampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = _DirectionSpy()
    monkeypatch.setattr(legacy_module, "SchrodingerPropagator", lambda **_: spy)
    external = np.arange(18, dtype=float).reshape(9, 2)

    result = legacy_module.run_legacy_batch_overlap_optimization(
        basis=_TwoStateBasis(),
        hamiltonian=_Hamiltonian(),
        dipole=_ZeroDipole(),
        states={"initial": (0,), "target": (1,)},
        time_cfg={"total_fs": 0.8, "field_dt_fs": 0.1, "output_stride": 1},
        params={
            "control_axes": "xy",
            "initial_field_kind": "sampled",
            "initial_field_samples": external,
            "initial_field_units": "V/m",
            "max_iter": 0,
        },
    )

    np.testing.assert_array_equal(result["field_data"], external)
    np.testing.assert_array_equal(spy.calls[0]["field"], external)
