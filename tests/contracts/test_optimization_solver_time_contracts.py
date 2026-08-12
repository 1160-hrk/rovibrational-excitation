"""Solver-boundary time contracts for GRAPE and Krotov."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import rovibrational_excitation.optimization.grape as grape_module
import rovibrational_excitation.optimization.krotov as krotov_module
from rovibrational_excitation.core.propagation import PropagationDirection


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

    def propagate(self, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
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
        params={"max_iter": 0},
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
    monkeypatch.setattr(krotov_module, "SchrodingerPropagator", lambda **_: spy)

    result = krotov_module.run_krotov_optimization(
        basis=_TwoStateBasis(),
        hamiltonian=_Hamiltonian(),
        dipole=_ZeroDipole(),
        states={"initial": (0,), "target": (1,)},
        time_cfg={"total_fs": 0.8, "field_dt_fs": 0.1, "output_stride": 3},
        params={
            "duration_initial": 0.2,
            "amplitude_initial": 0.0,
            "max_iter": 1,
        },
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
