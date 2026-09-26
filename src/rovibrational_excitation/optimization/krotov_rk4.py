"""Transparent piecewise-constant RK4 construction for standard Krotov."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class KrotovIteration:
    """All observable arrays from one first-order Krotov iteration."""

    old_trajectory: ComplexArray
    costates: ComplexArray
    updated_controls: RealArray
    updated_trajectory: ComplexArray
    fidelity_before: float
    fidelity_after: float


def _generator(
    h0: ComplexArray,
    dipoles: tuple[ComplexArray, ComplexArray],
    control: RealArray,
) -> ComplexArray:
    return np.asarray(
        -1j * (h0 - control[0] * dipoles[0] - control[1] * dipoles[1]),
        dtype=np.complex128,
    )


def rk4_constant_step(
    *,
    h0_rad_per_fs: ComplexArray,
    dipoles_rad_per_fs_per_v_per_m: tuple[ComplexArray, ComplexArray],
    control_v_per_m: RealArray,
    state: ComplexArray,
    dt_fs: float,
) -> ComplexArray:
    """Advance one state under one constant interval Hamiltonian."""
    operator = _generator(
        h0_rad_per_fs, dipoles_rad_per_fs_per_v_per_m, control_v_per_m
    )
    k1 = operator @ state
    k2 = operator @ (state + 0.5 * dt_fs * k1)
    k3 = operator @ (state + 0.5 * dt_fs * k2)
    k4 = operator @ (state + dt_fs * k3)
    result = state + (dt_fs / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    if not np.all(np.isfinite(result)):
        raise ValueError("standard Krotov RK4 produced a non-finite state")
    return np.asarray(result, dtype=np.complex128)


def propagate_interval_controls(
    *,
    h0_rad_per_fs: np.ndarray,
    dipoles_rad_per_fs_per_v_per_m: tuple[np.ndarray, np.ndarray],
    controls_v_per_m: np.ndarray,
    initial_state: np.ndarray,
    control_dt_fs: float,
) -> ComplexArray:
    """Propagate endpoint states under piecewise-constant interval controls."""
    h0, dipoles, controls, initial, dt = _validated_inputs(
        h0_rad_per_fs,
        dipoles_rad_per_fs_per_v_per_m,
        controls_v_per_m,
        initial_state,
        control_dt_fs,
    )
    trajectory = np.empty((controls.shape[0] + 1, h0.shape[0]), dtype=np.complex128)
    trajectory[0] = initial
    for interval in range(controls.shape[0]):
        trajectory[interval + 1] = rk4_constant_step(
            h0_rad_per_fs=h0,
            dipoles_rad_per_fs_per_v_per_m=dipoles,
            control_v_per_m=controls[interval],
            state=trajectory[interval],
            dt_fs=dt,
        )
    return trajectory


def evaluate_krotov_iteration(
    *,
    h0_rad_per_fs: np.ndarray,
    dipoles_rad_per_fs_per_v_per_m: tuple[np.ndarray, np.ndarray],
    controls_v_per_m: np.ndarray,
    initial_state: np.ndarray,
    target_state: np.ndarray,
    control_dt_fs: float,
    lambda_a_inverse_v_per_m_squared_fs: float,
    shape_values: np.ndarray,
) -> KrotovIteration:
    """Apply one sequential first-order Krotov interval update.

    ``H = H0 - sum_a mu_a E_a`` implies ``dH/dE_a = -mu_a``.
    The update therefore uses ``S/lambda_a * Im(<chi|-mu_a|psi_new>)``
    exactly once, with no additional factor of two.  Costates retain their
    terminal overlap scale and are never normalized.
    """
    h0, dipoles, controls, initial, dt = _validated_inputs(
        h0_rad_per_fs,
        dipoles_rad_per_fs_per_v_per_m,
        controls_v_per_m,
        initial_state,
        control_dt_fs,
    )
    target = np.asarray(target_state, dtype=np.complex128).reshape(-1)
    if target.shape != initial.shape or not np.all(np.isfinite(target)):
        raise ValueError("target_state must be finite and match initial_state")
    target_norm = float(np.linalg.norm(target))
    if target_norm == 0.0 or not np.isfinite(target_norm):
        raise ValueError("target_state must have a positive finite norm")
    penalty = float(lambda_a_inverse_v_per_m_squared_fs)
    if not np.isfinite(penalty) or penalty <= 0.0:
        raise ValueError("lambda_a must be positive and finite in canonical units")
    shape = np.asarray(shape_values, dtype=np.float64)
    if shape.shape != (controls.shape[0],) or not np.all(np.isfinite(shape)):
        raise ValueError("shape_values must be finite with one value per interval")
    if np.any(shape < 0.0):
        raise ValueError("shape_values must be nonnegative")

    old_trajectory = propagate_interval_controls(
        h0_rad_per_fs=h0,
        dipoles_rad_per_fs_per_v_per_m=dipoles,
        controls_v_per_m=controls,
        initial_state=initial,
        control_dt_fs=dt,
    )
    overlap = np.vdot(target, old_trajectory[-1])
    costates = np.empty_like(old_trajectory)
    costates[-1] = overlap * target
    for interval in range(controls.shape[0] - 1, -1, -1):
        costates[interval] = rk4_constant_step(
            h0_rad_per_fs=h0,
            dipoles_rad_per_fs_per_v_per_m=dipoles,
            control_v_per_m=controls[interval],
            state=costates[interval + 1],
            dt_fs=-dt,
        )

    updated_controls = np.array(controls, dtype=np.float64, copy=True)
    updated_trajectory = np.empty_like(old_trajectory)
    updated_trajectory[0] = initial
    for interval in range(controls.shape[0]):
        state = updated_trajectory[interval]
        costate = costates[interval]
        for axis, dipole in enumerate(dipoles):
            derivative = float(np.imag(np.vdot(costate, -(dipole @ state))))
            updated_controls[interval, axis] += shape[interval] * derivative / penalty
        updated_trajectory[interval + 1] = rk4_constant_step(
            h0_rad_per_fs=h0,
            dipoles_rad_per_fs_per_v_per_m=dipoles,
            control_v_per_m=updated_controls[interval],
            state=state,
            dt_fs=dt,
        )

    fidelity_before = float(np.abs(np.vdot(target, old_trajectory[-1])) ** 2)
    fidelity_after = float(np.abs(np.vdot(target, updated_trajectory[-1])) ** 2)
    return KrotovIteration(
        old_trajectory=old_trajectory,
        costates=costates,
        updated_controls=updated_controls,
        updated_trajectory=updated_trajectory,
        fidelity_before=fidelity_before,
        fidelity_after=fidelity_after,
    )


def _validated_inputs(
    h0_raw: np.ndarray,
    dipoles_raw: tuple[np.ndarray, np.ndarray],
    controls_raw: np.ndarray,
    initial_raw: np.ndarray,
    dt_raw: float,
) -> tuple[
    ComplexArray, tuple[ComplexArray, ComplexArray], RealArray, ComplexArray, float
]:
    h0 = np.ascontiguousarray(h0_raw, dtype=np.complex128)
    if h0.ndim != 2 or h0.shape[0] != h0.shape[1] or not np.all(np.isfinite(h0)):
        raise ValueError("h0_rad_per_fs must be a finite square matrix")
    dipoles = (
        np.ascontiguousarray(dipoles_raw[0], dtype=np.complex128),
        np.ascontiguousarray(dipoles_raw[1], dtype=np.complex128),
    )
    if any(matrix.shape != h0.shape for matrix in dipoles) or any(
        not np.all(np.isfinite(matrix)) for matrix in dipoles
    ):
        raise ValueError("dipole matrices must be finite and match h0_rad_per_fs")
    controls = np.ascontiguousarray(controls_raw, dtype=np.float64)
    if controls.ndim != 2 or controls.shape[1] != 2 or controls.shape[0] < 1:
        raise ValueError("controls_v_per_m must have shape (n_intervals, 2)")
    if not np.all(np.isfinite(controls)):
        raise ValueError("controls_v_per_m must contain only finite values")
    initial = np.asarray(initial_raw, dtype=np.complex128).reshape(-1)
    if initial.shape != (h0.shape[0],) or not np.all(np.isfinite(initial)):
        raise ValueError("initial_state must be finite and match the operator size")
    norm = float(np.linalg.norm(initial))
    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError("initial_state must have a positive finite norm")
    dt = float(dt_raw)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("control_dt_fs must be positive and finite")
    return h0, dipoles, controls, initial, dt


__all__ = [
    "KrotovIteration",
    "evaluate_krotov_iteration",
    "propagate_interval_controls",
    "rk4_constant_step",
]
