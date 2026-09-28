"""Exact discrete adjoint for the normalized dense NumPy RK4 map used by GRAPE."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .objective import DiscreteL2TargetObjective, VectorTargetPopulation

ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class GrapeEvaluation:
    """One terminal-population objective evaluation and its exact gradient."""

    fidelity: float
    objective: float
    gradient: RealArray
    trajectory: ComplexArray


@dataclass(frozen=True, slots=True)
class _Rk4Step:
    """Forward intermediates required to reverse one normalized RK4 step."""

    state: ComplexArray
    state_k2: ComplexArray
    state_k3: ComplexArray
    state_k4: ComplexArray
    unnormalized_state: ComplexArray
    norm: float


def _stage_operator(
    h0: ComplexArray,
    dipoles: tuple[ComplexArray, ComplexArray],
    field_sample: RealArray,
) -> ComplexArray:
    """Return the Schrödinger RHS matrix for one Cartesian field sample."""
    return np.asarray(
        -1j * (h0 - field_sample[0] * dipoles[0] - field_sample[1] * dipoles[1]),
        dtype=np.complex128,
    )


def _control_derivative(
    adjoint: ComplexArray,
    dipole: ComplexArray,
    state: ComplexArray,
) -> float:
    """Derivative through ``(-i (H0 - mu E)) @ state`` for real ``E``."""
    return 2.0 * float(np.real(np.vdot(adjoint, 1j * (dipole @ state))))


def evaluate_discrete_rk4(
    *,
    h0_rad_per_fs: np.ndarray,
    dipoles_rad_per_fs_per_v_per_m: tuple[np.ndarray, np.ndarray],
    field_v_per_m: np.ndarray,
    initial_state: np.ndarray,
    target_state: np.ndarray,
    propagation_dt_fs: float,
    lambda_a: float,
) -> GrapeEvaluation:
    """Evaluate ``1-F + lambda_a/2 * sum(E**2)`` and its discrete gradient.

    The reverse pass differentiates the actual left/midpoint/right RK4 graph,
    including the per-step normalization used by the production propagator.
    Field endpoints shared by adjacent RK4 steps therefore receive both
    contributions.  This routine intentionally supports only dense NumPy
    arrays and is the numerical definition of the current GRAPE solver.
    """
    h0 = np.ascontiguousarray(h0_rad_per_fs, dtype=np.complex128)
    dipoles = (
        np.ascontiguousarray(dipoles_rad_per_fs_per_v_per_m[0], dtype=np.complex128),
        np.ascontiguousarray(dipoles_rad_per_fs_per_v_per_m[1], dtype=np.complex128),
    )
    field = np.ascontiguousarray(field_v_per_m, dtype=np.float64)
    state = np.array(initial_state, dtype=np.complex128, copy=True).ravel()
    target = np.array(target_state, dtype=np.complex128, copy=True).ravel()
    dt = float(propagation_dt_fs)

    if h0.ndim != 2 or h0.shape[0] != h0.shape[1]:
        raise ValueError("h0_rad_per_fs must be a square matrix")
    dimension = h0.shape[0]
    if any(matrix.shape != h0.shape for matrix in dipoles):
        raise ValueError("dipole matrices must have the same shape as h0_rad_per_fs")
    if field.ndim != 2 or field.shape[1] != 2 or field.shape[0] < 3:
        raise ValueError("field_v_per_m must have shape (2 * n_steps + 1, 2)")
    if field.shape[0] % 2 == 0:
        raise ValueError("field_v_per_m must contain an odd number of samples")
    if state.shape != (dimension,) or target.shape != (dimension,):
        raise ValueError("initial_state and target_state must match the operator size")
    if not np.all(np.isfinite(field)):
        raise ValueError("field_v_per_m must contain only finite values")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("propagation_dt_fs must be positive and finite")
    if not np.isfinite(lambda_a):
        raise ValueError("lambda_a must be finite")

    initial_norm = float(np.linalg.norm(state))
    target_norm = float(np.linalg.norm(target))
    if initial_norm == 0.0 or not np.isfinite(initial_norm):
        raise ValueError("initial_state must have a positive finite norm")
    if target_norm == 0.0 or not np.isfinite(target_norm):
        raise ValueError("target_state must have a positive finite norm")

    trajectory = np.empty(
        ((field.shape[0] - 1) // 2 + 1, dimension), dtype=np.complex128
    )
    trajectory[0] = state
    tape: list[_Rk4Step] = []

    for step_index in range(trajectory.shape[0] - 1):
        field_index = 2 * step_index
        operator_left = _stage_operator(h0, dipoles, field[field_index])
        operator_mid = _stage_operator(h0, dipoles, field[field_index + 1])
        operator_right = _stage_operator(h0, dipoles, field[field_index + 2])

        k1 = operator_left @ state
        state_k2 = state + 0.5 * dt * k1
        k2 = operator_mid @ state_k2
        state_k3 = state + 0.5 * dt * k2
        k3 = operator_mid @ state_k3
        state_k4 = state + dt * k3
        k4 = operator_right @ state_k4
        unnormalized_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        norm = float(np.linalg.norm(unnormalized_state))
        if norm == 0.0 or not np.isfinite(norm):
            raise ValueError("cannot normalize a zero or non-finite RK4 state")

        tape.append(
            _Rk4Step(
                state=state,
                state_k2=state_k2,
                state_k3=state_k3,
                state_k4=state_k4,
                unnormalized_state=unnormalized_state,
                norm=norm,
            )
        )
        state = unnormalized_state / norm
        trajectory[step_index + 1] = state

    target_evaluator = VectorTargetPopulation(target)
    overlap, target_evaluation = target_evaluator.evaluate_with_overlap(state)
    fidelity = target_evaluation.fidelity
    objective = DiscreteL2TargetObjective(float(lambda_a)).evaluate(
        target_evaluation,
        field,
    )
    gradient = np.asarray(float(lambda_a) * field, dtype=np.float64)

    # For a real objective, variations use dJ = 2 Re(<adjoint, dstate>).
    adjoint = -overlap * target
    for step_index in range(len(tape) - 1, -1, -1):
        step = tape[step_index]
        field_index = 2 * step_index
        operator_left = _stage_operator(h0, dipoles, field[field_index])
        operator_mid = _stage_operator(h0, dipoles, field[field_index + 1])
        operator_right = _stage_operator(h0, dipoles, field[field_index + 2])

        # Reverse y = z / ||z||, including the production per-step renormalization.
        projection = float(np.real(np.vdot(adjoint, step.unnormalized_state)))
        adjoint_z = (
            adjoint / step.norm - step.unnormalized_state * projection / step.norm**3
        )

        adjoint_state = adjoint_z.copy()
        adjoint_k1 = (dt / 6.0) * adjoint_z
        adjoint_k2 = (dt / 3.0) * adjoint_z
        adjoint_k3 = (dt / 3.0) * adjoint_z
        adjoint_k4 = (dt / 6.0) * adjoint_z

        for axis, dipole in enumerate(dipoles):
            gradient[field_index + 2, axis] += _control_derivative(
                adjoint_k4, dipole, step.state_k4
            )
        adjoint_state_k4 = operator_right.conj().T @ adjoint_k4
        adjoint_state += adjoint_state_k4
        adjoint_k3 += dt * adjoint_state_k4

        for axis, dipole in enumerate(dipoles):
            gradient[field_index + 1, axis] += _control_derivative(
                adjoint_k3, dipole, step.state_k3
            )
        adjoint_state_k3 = operator_mid.conj().T @ adjoint_k3
        adjoint_state += adjoint_state_k3
        adjoint_k2 += 0.5 * dt * adjoint_state_k3

        for axis, dipole in enumerate(dipoles):
            gradient[field_index + 1, axis] += _control_derivative(
                adjoint_k2, dipole, step.state_k2
            )
        adjoint_state_k2 = operator_mid.conj().T @ adjoint_k2
        adjoint_state += adjoint_state_k2
        adjoint_k1 += 0.5 * dt * adjoint_state_k2

        for axis, dipole in enumerate(dipoles):
            gradient[field_index, axis] += _control_derivative(
                adjoint_k1, dipole, step.state
            )
        adjoint_state += operator_left.conj().T @ adjoint_k1
        adjoint = adjoint_state

    return GrapeEvaluation(
        fidelity=fidelity,
        objective=objective,
        gradient=gradient,
        trajectory=trajectory,
    )


__all__ = ["GrapeEvaluation", "evaluate_discrete_rk4"]
