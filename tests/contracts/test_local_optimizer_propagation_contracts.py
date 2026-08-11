"""Propagation-boundary contracts for the legacy local time layout."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import rovibrational_excitation.optimization.local as local_module


class _OneStateBasis:
    basis = [(0,)]

    @staticmethod
    def size() -> int:
        return 1

    @staticmethod
    def get_index(state: tuple[int, ...]) -> int:
        assert tuple(state) == (0,)
        return 0


class _ZeroHamiltonian:
    @staticmethod
    def get_eigenvalues() -> np.ndarray:
        return np.zeros(1)


class _ZeroDipole:
    @staticmethod
    def get_mu_x_SI() -> np.ndarray:
        return np.zeros((1, 1))

    @staticmethod
    def get_mu_y_SI() -> np.ndarray:
        return np.zeros((1, 1))

    @staticmethod
    def get_mu_z_SI() -> np.ndarray:
        return np.zeros((1, 1))


class _PropagationSpy:
    def __init__(self) -> None:
        self.calls: list[dict[str, np.ndarray]] = []

    def propagate(self, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        efield = kwargs["efield"]
        tlist = np.array(efield.tlist, copy=True)
        field = np.array(efield.get_Efield(), copy=True)
        assert tlist.size % 2 == 1
        self.calls.append({"tlist": tlist, "field": field})

        stride = int(kwargs["sample_stride"])
        propagation_steps = (tlist.size - 1) // 2
        sampled_steps = propagation_steps // stride
        trajectory = np.repeat(
            np.asarray(kwargs["initial_state"])[np.newaxis, :],
            sampled_steps + 1,
            axis=0,
        )
        time = tlist[0] + np.arange(sampled_steps + 1) * 2.0 * efield.dt * stride
        return time, trajectory


@pytest.mark.parametrize(
    (
        "time_total",
        "expected_call_lengths",
        "expected_full_last_time",
        "expected_storage_length",
        "expected_storage_last_time",
    ),
    [
        (0.4, [5, 7], 0.6, 7, 0.6),
        (0.8, [5, 5, 9], 0.8, 10, 0.9),
    ],
)
def test_local_optimizer_passes_exact_legacy_odd_prefix_to_full_rk4(
    monkeypatch: pytest.MonkeyPatch,
    time_total: float,
    expected_call_lengths: list[int],
    expected_full_last_time: float,
    expected_storage_length: int,
    expected_storage_last_time: float,
) -> None:
    spy = _PropagationSpy()
    monkeypatch.setattr(local_module, "SchrodingerPropagator", lambda **_: spy)

    result = local_module.run_local_optimization(
        basis=_OneStateBasis(),
        hamiltonian=_ZeroHamiltonian(),
        dipole=_ZeroDipole(),
        states={"initial": (0,), "target": (0,)},
        time_cfg={"total_fs": time_total, "dt_fs": 0.1, "sample_stride": 1},
        params={"segment_size_steps": None, "segment_size_fs": 0.5},
    )

    assert [call["tlist"].size for call in spy.calls] == expected_call_lengths
    assert spy.calls[-1]["tlist"][-1] == pytest.approx(expected_full_last_time)
    assert result["tlist"].size == expected_storage_length
    assert result["tlist"][-1] == pytest.approx(expected_storage_last_time)


def test_local_optimizer_keeps_shared_boundary_on_previous_segment() -> None:
    spy = _PropagationSpy()
    original_propagator = local_module.SchrodingerPropagator
    local_module.SchrodingerPropagator = lambda **_: spy  # type: ignore[misc]
    try:
        result = local_module.run_local_optimization(
            basis=_OneStateBasis(),
            hamiltonian=_ZeroHamiltonian(),
            dipole=_ZeroDipole(),
            states={"initial": (0,), "target": (0,)},
            time_cfg={"total_fs": 0.8, "dt_fs": 0.1, "sample_stride": 1},
            params={"segment_size_steps": None, "segment_size_fs": 0.5},
        )
    finally:
        local_module.SchrodingerPropagator = original_propagator

    expected_component = np.array([0.0, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 0.0])
    np.testing.assert_array_equal(result["field_data"][:, 0], expected_component)
    np.testing.assert_array_equal(result["field_data"][:, 1], expected_component)
    np.testing.assert_array_equal(spy.calls[0]["field"][:, 0], expected_component[:5])
    np.testing.assert_array_equal(spy.calls[1]["field"][:, 0], expected_component[4:9])
