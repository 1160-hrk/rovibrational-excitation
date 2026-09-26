"""Independent one-iteration reference for standard Krotov control."""

from __future__ import annotations

import numpy as np

from rovibrational_excitation.optimization.krotov_rk4 import (
    evaluate_krotov_iteration,
)


def _direct_step(h0, dipoles, control, state, dt):
    hamiltonian = h0 - control[0] * dipoles[0] - control[1] * dipoles[1]

    def rhs(vector):
        return -1j * (hamiltonian @ vector)

    k1 = rhs(state)
    k2 = rhs(state + 0.5 * dt * k1)
    k3 = rhs(state + 0.5 * dt * k2)
    k4 = rhs(state + dt * k3)
    return state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def _direct_iteration(h0, dipoles, controls, initial, target, dt, penalty, shape):
    old = np.empty((controls.shape[0] + 1, initial.size), dtype=np.complex128)
    old[0] = initial
    for index, control in enumerate(controls):
        old[index + 1] = _direct_step(h0, dipoles, control, old[index], dt)

    overlap = np.vdot(target, old[-1])
    costates = np.empty_like(old)
    costates[-1] = overlap * target
    for index in range(controls.shape[0] - 1, -1, -1):
        costates[index] = _direct_step(
            h0, dipoles, controls[index], costates[index + 1], -dt
        )

    updated = controls.copy()
    new = np.empty_like(old)
    new[0] = initial
    for index in range(controls.shape[0]):
        for axis in range(2):
            matrix_element = np.vdot(costates[index], -dipoles[axis] @ new[index])
            updated[index, axis] += shape[index] * matrix_element.imag / penalty
        new[index + 1] = _direct_step(h0, dipoles, updated[index], new[index], dt)
    return old, costates, updated, new


def test_standard_krotov_matches_direct_sequential_one_iteration_reference():
    h0 = np.array([[0.0, 0.03], [0.03, 0.47]], dtype=np.complex128)
    dipoles = (
        np.array([[0.0, 2.1e-10], [2.1e-10, 0.0]], dtype=np.complex128),
        np.array([[0.0, -0.4e-10j], [0.4e-10j, 0.0]], dtype=np.complex128),
    )
    controls = np.array(
        [[2.0e8, -1.0e8], [3.0e8, 0.5e8], [-2.0e8, 1.5e8]],
        dtype=np.float64,
    )
    initial = np.array([1.0, 0.0], dtype=np.complex128)
    target = np.array([0.0, 1.0], dtype=np.complex128)
    dt = 0.08
    penalty = 2.5e-20
    shape = np.array([0.25, 1.0, 0.25])

    expected = _direct_iteration(
        h0, dipoles, controls, initial, target, dt, penalty, shape
    )
    actual = evaluate_krotov_iteration(
        h0_rad_per_fs=h0,
        dipoles_rad_per_fs_per_v_per_m=dipoles,
        controls_v_per_m=controls,
        initial_state=initial,
        target_state=target,
        control_dt_fs=dt,
        lambda_a_inverse_v_per_m_squared_fs=penalty,
        shape_values=shape,
    )

    for observed, reference in zip(
        (
            actual.old_trajectory,
            actual.costates,
            actual.updated_controls,
            actual.updated_trajectory,
        ),
        expected,
        strict=True,
    ):
        np.testing.assert_allclose(observed, reference, rtol=0.0, atol=2e-15)

    terminal_overlap = np.vdot(target, expected[0][-1])
    np.testing.assert_allclose(
        np.linalg.norm(actual.costates[-1]), abs(terminal_overlap), rtol=0.0, atol=1e-18
    )
    assert not np.isclose(np.linalg.norm(actual.costates[-1]), 1.0)
    np.testing.assert_allclose(
        actual.fidelity_before, abs(terminal_overlap) ** 2, rtol=0.0, atol=1e-20
    )
    np.testing.assert_allclose(
        actual.fidelity_after,
        abs(np.vdot(target, expected[3][-1])) ** 2,
        rtol=0.0,
        atol=1e-20,
    )
