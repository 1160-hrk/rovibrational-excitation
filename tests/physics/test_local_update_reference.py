"""Independent direct reference for the legacy-grid Local control update."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest

from rovibrational_excitation.core.units.constants import CONSTANTS
from rovibrational_excitation.models.two_level import (
    TwoLevelBasis,
    TwoLevelDipoleMatrix,
)
from rovibrational_excitation.optimization import ControlLayout, OptimizationResult
from rovibrational_excitation.optimization.local import run_local_optimization


def _direct_rk4(
    h0: np.ndarray,
    dipoles: tuple[np.ndarray, np.ndarray],
    field: np.ndarray,
    initial: np.ndarray,
    propagation_dt_fs: float,
) -> np.ndarray:
    """Slow normalized RK4 written without a production propagation helper."""
    state = np.array(initial, dtype=np.complex128, copy=True)
    trajectory = [state.copy()]
    for step_index in range((field.shape[0] - 1) // 2):
        left = 2 * step_index

        def rhs(vector: np.ndarray, sample_index: int) -> np.ndarray:
            hamiltonian = (
                h0
                - field[sample_index, 0] * dipoles[0]
                - field[sample_index, 1] * dipoles[1]
            )
            return -1j * (hamiltonian @ vector)

        k1 = rhs(state, left)
        k2 = rhs(state + 0.5 * propagation_dt_fs * k1, left + 1)
        k3 = rhs(state + 0.5 * propagation_dt_fs * k2, left + 1)
        k4 = rhs(state + propagation_dt_fs * k3, left + 2)
        state = state + (propagation_dt_fs / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        state = state / np.linalg.norm(state)
        trajectory.append(state.copy())
    return np.asarray(trajectory)


def _direct_local_update(
    *,
    eval_mode: Literal["weights", "target"],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply both Local segments directly on the frozen legacy layout."""
    field_dt_fs = 0.1
    propagation_dt_fs = 2.0 * field_dt_fs
    tlist = np.arange(0.0, 1.0, field_dt_fs)
    segments = ((0, 4), (4, 8))

    h0 = np.diag([0.0, 0.47]).astype(np.complex128)
    coupling = 2.1e-29 / CONSTANTS.get_hbar_in_units("J·fs")
    mu_x = coupling * np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    mu_y = coupling * np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128)
    dipoles = (mu_x, mu_y)
    eigenvalues = np.array([0.0, 0.47])
    initial = np.array([1.0, 0.0], dtype=np.complex128)
    target = np.array([0.0, 1.0], dtype=np.complex128)
    weights = np.array([0.0, 1.0])

    gain = 2.0e19
    seed_amplitude = 2.0e8
    seed_segments_left = 1
    shape_floor = 1.0e-2
    lookahead_fraction = 0.5
    field_limit = 1.0e12

    field = np.zeros((tlist.size, 2), dtype=np.float64)
    state = initial.copy()
    for start, end in segments:
        midpoint = (start + end) // 2
        shape = float(np.sin(np.pi * tlist[midpoint] / tlist[-1]) ** 2)
        shape_for_seed = max(shape, shape_floor)

        lookahead_time = lookahead_fraction * (tlist[end - 1] - tlist[start])
        reference_state = state * np.exp(-1j * eigenvalues * lookahead_time)

        if eval_mode == "weights":
            response_x = float(
                np.imag(np.vdot(reference_state, weights * (-mu_x @ reference_state)))
            )
            response_y = float(
                np.imag(np.vdot(reference_state, weights * (-mu_y @ reference_state)))
            )
            control_x = gain * shape * response_x
            control_y = gain * shape * response_y
            seed_triggered = abs(response_x) < 1.0e-18 and abs(response_y) < 1.0e-18
            if seed_triggered and seed_segments_left > 0:
                control_x = seed_amplitude * shape_for_seed
                control_y = seed_amplitude * shape_for_seed
                seed_segments_left -= 1
        else:
            overlap = complex(np.vdot(target, reference_state))
            derivative_x = complex(np.vdot(target, -mu_x @ reference_state))
            derivative_y = complex(np.vdot(target, -mu_y @ reference_state))
            control_x = gain * shape * float(np.imag(np.conj(overlap) * derivative_x))
            control_y = gain * shape * float(np.imag(np.conj(overlap) * derivative_y))
            if abs(overlap) < 1.0e-1 and seed_segments_left > 0:
                sign_x = 1.0 if derivative_x.real >= 0.0 else -1.0
                sign_y = 1.0 if derivative_y.real >= 0.0 else -1.0
                control_x = sign_x * seed_amplitude * shape_for_seed
                control_y = sign_y * seed_amplitude * shape_for_seed
                seed_segments_left -= 1

        field[start + 1 : end + 1, 0] = np.clip(control_x, -field_limit, field_limit)
        field[start + 1 : end + 1, 1] = np.clip(control_y, -field_limit, field_limit)
        state = _direct_rk4(
            h0,
            dipoles,
            field[start : end + 1],
            state,
            propagation_dt_fs,
        )[-1]

    full_trajectory = _direct_rk4(
        h0,
        dipoles,
        field[:9],
        initial,
        propagation_dt_fs,
    )
    return tlist, field, full_trajectory


@pytest.mark.physics
@pytest.mark.parametrize("eval_mode", ["weights", "target"])
def test_local_update_matches_direct_legacy_grid_reference(
    eval_mode: Literal["weights", "target"],
) -> None:
    basis = TwoLevelBasis(
        energy_gap=0.47,
        input_units="rad/fs",
        output_units="rad/fs",
    )
    hamiltonian = basis.generate_H0()
    dipole = TwoLevelDipoleMatrix(
        basis=basis,
        mu0=2.1e-29,
        units="C*m",
        units_input="C*m",
    )

    actual = run_local_optimization(
        basis=basis,
        hamiltonian=hamiltonian,
        dipole=dipole,
        states={"initial": (0,), "target": (1,)},
        time_cfg={"total_fs": 0.8, "field_dt_fs": 0.1, "sample_stride": 1},
        params={
            "control_axes": "xy",
            "gain": 20.0,
            "gain_units": "(GV/m)^2 fs",
            "initialization": {
                "method": "seed_field",
                "amplitude": 2.0e8,
                "amplitude_units": "V/m",
                "max_segments": 1,
            },
            "segment_size_steps": 4,
            "segment_size_fs": None,
            "use_sin2_shape": True,
            "shape_floor": 1.0e-2,
            "lookahead_enable": True,
            "lookahead_fraction": 0.5,
            "eval_mode": eval_mode,
            "field_max_v_per_m": 1.0e12,
        },
    )
    expected_tlist, expected_field, expected_trajectory = _direct_local_update(
        eval_mode=eval_mode
    )

    assert isinstance(actual, OptimizationResult)
    assert actual.control_layout is ControlLayout.LOCAL_LEGACY_FIELD_SAMPLES
    np.testing.assert_array_equal(actual.control_times_fs, expected_tlist)
    assert np.linalg.norm(expected_field[5:9]) > 1.0e6
    np.testing.assert_allclose(
        actual.controls_v_per_m, expected_field, rtol=2e-15, atol=0.0
    )
    np.testing.assert_allclose(
        actual.trajectory, expected_trajectory, rtol=2e-15, atol=2e-18
    )
    assert actual.controls_v_per_m[-1, 0] == 0.0
    assert actual.metrics["seed_segments_used"] == 1
    assert actual.metrics["clipped_segment_fraction"] == 0.0
    assert actual.metrics["fidelity"] == pytest.approx(
        abs(expected_trajectory[-1, 1]) ** 2,
        rel=2e-15,
        abs=0.0,
    )
