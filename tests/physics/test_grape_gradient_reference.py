"""Independent finite-difference reference for the discrete GRAPE gradient."""

from __future__ import annotations

import numpy as np
import pytest

from rovibrational_excitation.dynamics.utils import cm_to_rad_phz
from rovibrational_excitation.models.two_level import (
    TwoLevelBasis,
    TwoLevelDipoleMatrix,
)
from rovibrational_excitation.optimization import ControlLayout, OptimizationResult
from rovibrational_excitation.optimization.grape import run_grape_optimization
from rovibrational_excitation.optimization.grape_rk4 import evaluate_discrete_rk4


def _reference_objective(
    h0: np.ndarray,
    dipoles: tuple[np.ndarray, np.ndarray],
    field: np.ndarray,
    initial: np.ndarray,
    target: np.ndarray,
    dt_fs: float,
    lambda_a: float,
) -> tuple[float, np.ndarray]:
    """Direct, deliberately slow RK4 oracle independent of production code."""
    state = np.array(initial, dtype=np.complex128, copy=True)
    trajectory = [state.copy()]
    for step_index in range((field.shape[0] - 1) // 2):
        left = 2 * step_index

        def rhs(value: np.ndarray, sample_index: int) -> np.ndarray:
            hamiltonian = (
                h0
                - field[sample_index, 0] * dipoles[0]
                - field[sample_index, 1] * dipoles[1]
            )
            return -1j * (hamiltonian @ value)

        k1 = rhs(state, left)
        k2 = rhs(state + 0.5 * dt_fs * k1, left + 1)
        k3 = rhs(state + 0.5 * dt_fs * k2, left + 1)
        k4 = rhs(state + dt_fs * k3, left + 2)
        state = state + (dt_fs / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        state = state / np.linalg.norm(state)
        trajectory.append(state.copy())

    fidelity = float(np.abs(np.vdot(target, state)) ** 2)
    objective = float(1.0 - fidelity + 0.5 * lambda_a * float(np.sum(field * field)))
    return objective, np.asarray(trajectory)


def _central_difference(
    *,
    h0: np.ndarray,
    dipoles: tuple[np.ndarray, np.ndarray],
    field: np.ndarray,
    initial: np.ndarray,
    target: np.ndarray,
    dt_fs: float,
    lambda_a: float,
    epsilon_v_per_m: float,
) -> np.ndarray:
    gradient = np.empty_like(field)
    for sample_index in range(field.shape[0]):
        for axis_index in range(field.shape[1]):
            plus = field.copy()
            minus = field.copy()
            plus[sample_index, axis_index] += epsilon_v_per_m
            minus[sample_index, axis_index] -= epsilon_v_per_m
            objective_plus, _ = _reference_objective(
                h0, dipoles, plus, initial, target, dt_fs, lambda_a
            )
            objective_minus, _ = _reference_objective(
                h0, dipoles, minus, initial, target, dt_fs, lambda_a
            )
            gradient[sample_index, axis_index] = (objective_plus - objective_minus) / (
                2.0 * epsilon_v_per_m
            )
    return gradient


@pytest.mark.physics
def test_discrete_grape_gradient_matches_converged_central_difference() -> None:
    h0 = np.diag([0.0, 0.7]).astype(np.complex128)
    coupling = 1.9e-10
    dipoles = (
        coupling * np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
        coupling * np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128),
    )
    field = np.array(
        [
            [2.0e6, -3.0e6],
            [4.0e6, 1.0e6],
            [-2.0e6, 5.0e6],
            [3.0e6, -4.0e6],
            [1.0e6, 2.0e6],
        ]
    )
    initial = np.array([1.0, 0.0], dtype=np.complex128)
    target = np.array([0.0, 1.0], dtype=np.complex128)
    dt_fs = 0.2
    lambda_a = 0.0

    actual = evaluate_discrete_rk4(
        h0_rad_per_fs=h0,
        dipoles_rad_per_fs_per_v_per_m=dipoles,
        field_v_per_m=field,
        initial_state=initial,
        target_state=target,
        propagation_dt_fs=dt_fs,
        lambda_a=lambda_a,
    )
    reference_objective, reference_trajectory = _reference_objective(
        h0, dipoles, field, initial, target, dt_fs, lambda_a
    )
    assert actual.objective == pytest.approx(reference_objective, abs=2e-16)
    np.testing.assert_allclose(
        actual.trajectory, reference_trajectory, rtol=3e-16, atol=2e-19
    )

    relative_errors = []
    for epsilon in (1.0e8, 1.0e7, 1.0e6):
        reference_gradient = _central_difference(
            h0=h0,
            dipoles=dipoles,
            field=field,
            initial=initial,
            target=target,
            dt_fs=dt_fs,
            lambda_a=lambda_a,
            epsilon_v_per_m=epsilon,
        )
        relative_errors.append(
            float(
                np.linalg.norm(actual.gradient - reference_gradient)
                / np.linalg.norm(actual.gradient)
            )
        )

    # Observed plateau on this fixed problem is 5.8e-9 at 1e6 V/m.  The
    # 1e-7 bound is deliberately above that plateau and below D-072's 1e-5
    # initial target; the ordered values prove step-size convergence.
    assert relative_errors[1] < relative_errors[0]
    assert relative_errors[2] < relative_errors[1]
    assert relative_errors[2] < 1.0e-7


@pytest.mark.physics
def test_grape_runner_applies_the_exact_discrete_gradient() -> None:
    basis = TwoLevelBasis(
        energy_gap=0.7,
        input_units="rad/fs",
        output_units="rad/fs",
    )
    hamiltonian = basis.generate_H0()
    dipole = TwoLevelDipoleMatrix(
        basis=basis,
        mu0=2.0e-29,
        units="C*m",
        units_input="C*m",
    )
    seed = np.array(
        [
            [2.0e6, -3.0e6],
            [4.0e6, 1.0e6],
            [-2.0e6, 5.0e6],
            [3.0e6, -4.0e6],
            [1.0e6, 2.0e6],
        ]
    )
    learning_rate = 1.0e16
    mu_x = np.asarray(cm_to_rad_phz(dipole.get_mu_x_SI()), dtype=np.complex128)
    mu_y = np.asarray(cm_to_rad_phz(dipole.get_mu_y_SI()), dtype=np.complex128)
    expected = evaluate_discrete_rk4(
        h0_rad_per_fs=hamiltonian.get_matrix("rad/fs"),
        dipoles_rad_per_fs_per_v_per_m=(mu_x, mu_y),
        field_v_per_m=seed,
        initial_state=np.array([1.0, 0.0], dtype=np.complex128),
        target_state=np.array([0.0, 1.0], dtype=np.complex128),
        propagation_dt_fs=0.2,
        lambda_a=0.0,
    )

    result = run_grape_optimization(
        basis=basis,
        hamiltonian=hamiltonian,
        dipole=dipole,
        states={"initial": (0,), "target": (1,)},
        time_cfg={"total_fs": 0.4, "field_dt_fs": 0.1, "output_stride": 1},
        params={
            "control_axes": "xy",
            "initial_field_kind": "sampled",
            "initial_field_samples": seed,
            "initial_field_units": "V/m",
            "max_iter": 1,
            "learning_rate": learning_rate,
            "lambda_a": 0.0,
            "target_fidelity": 1.0,
        },
    )

    assert isinstance(result, OptimizationResult)
    assert result.control_layout is ControlLayout.RK4_FIELD_SAMPLES
    np.testing.assert_allclose(
        result.controls_v_per_m,
        seed - learning_rate * expected.gradient,
        rtol=0.0,
        atol=2e-10,
    )
    assert result.metrics["fidelity"] > expected.fidelity
