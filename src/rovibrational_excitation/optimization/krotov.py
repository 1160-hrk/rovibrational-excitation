"""Standard first-order Krotov optimization on interval controls."""

from __future__ import annotations

from typing import Any

import numpy as np

from rovibrational_excitation.core.units import KrotovPenalty
from rovibrational_excitation.dynamics.utils import cm_to_rad_phz

from .krotov_controls import parse_krotov_initial_control
from .krotov_rk4 import evaluate_krotov_iteration, propagate_interval_controls
from .krotov_timegrid import KrotovIntervalGrid
from .objective import IndexedTargetPopulation
from .options import validate_algorithm_options
from .result import ControlLayout, OptimizationResult
from .timegrid import sample_optimization_output

DEFAULT_MAX_ITER = 1000
DEFAULT_TARGET_FIDELITY = 1.0


def _shape_function(control_times_fs: np.ndarray, total_fs: float) -> np.ndarray:
    return np.sin(np.pi * control_times_fs / total_fs) ** 2


def run_krotov_optimization(
    *, basis, hamiltonian, dipole, states: dict[str, Any], time_cfg: dict, params: dict
) -> OptimizationResult:
    """Optimize terminal population using sequential interval updates.

    Controls are piecewise constant on ``[t_n, t_{n+1})`` and stored at the
    interval midpoints.  This route intentionally does not accept the old
    shared left/mid/right RK4 field grid; that calculation is available only
    as ``legacy_batch_overlap``.
    """
    initial_control = parse_krotov_initial_control(params)
    control_axes = validate_algorithm_options("krotov", params)
    grid = KrotovIntervalGrid.from_config(time_cfg)

    initial_idx = basis.get_index(tuple(states["initial"]))
    target_raw = states.get("target")
    if target_raw is None:
        raise ValueError("standard Krotov requires a target state")
    target_idx = basis.get_index(tuple(target_raw))

    penalty = KrotovPenalty(params["lambda_a"], params["lambda_a_units"])
    lambda_a = penalty.inverse_volts_per_meter_squared_femtoseconds
    max_iter = int(params.get("max_iter", DEFAULT_MAX_ITER))
    target_fidelity = float(params.get("target_fidelity", DEFAULT_TARGET_FIDELITY))

    dimension = basis.size()
    initial_state = np.zeros(dimension, dtype=np.complex128)
    initial_state[initial_idx] = 1.0
    target_state = np.zeros(dimension, dtype=np.complex128)
    target_state[target_idx] = 1.0

    h0 = np.asarray(hamiltonian.get_matrix("rad/fs"), dtype=np.complex128)
    dipoles: list[np.ndarray] = []
    for axis in control_axes:
        component = getattr(dipole, f"get_mu_{axis}_SI")()
        if hasattr(component, "toarray"):
            component = component.toarray()
        dipoles.append(np.asarray(cm_to_rad_phz(component), dtype=np.complex128))
    dipole_pair = (dipoles[0], dipoles[1])

    controls = initial_control.samples_on(grid)
    shape = _shape_function(grid.control_times_fs, grid.state_times_fs[-1])
    trajectory = propagate_interval_controls(
        h0_rad_per_fs=h0,
        dipoles_rad_per_fs_per_v_per_m=dipole_pair,
        controls_v_per_m=controls,
        initial_state=initial_state,
        control_dt_fs=grid.control_dt_fs,
    )
    target_evaluator = IndexedTargetPopulation(target_idx)
    target_evaluation = target_evaluator.evaluate(trajectory[-1])
    fidelity = target_evaluation.fidelity
    fidelity_history = [fidelity]
    completed_iterations = 0

    for _ in range(max_iter):
        if fidelity >= target_fidelity:
            break
        iteration = evaluate_krotov_iteration(
            h0_rad_per_fs=h0,
            dipoles_rad_per_fs_per_v_per_m=dipole_pair,
            controls_v_per_m=controls,
            initial_state=initial_state,
            target_state=target_state,
            control_dt_fs=grid.control_dt_fs,
            lambda_a_inverse_v_per_m_squared_fs=lambda_a,
            shape_values=shape,
        )
        controls = iteration.updated_controls
        trajectory = iteration.updated_trajectory
        fidelity = iteration.fidelity_after
        fidelity_history.append(fidelity)
        completed_iterations += 1

    time_out, trajectory_out = sample_optimization_output(
        grid.state_times_fs, trajectory, output_stride=grid.output_stride
    )
    control_times = np.array(grid.control_times_fs, copy=True)
    control_data = np.array(controls, copy=True)
    return OptimizationResult(
        trajectory_times_fs=time_out,
        trajectory=trajectory_out,
        metrics={
            "fidelity": fidelity,
            "terminal_objective": 1.0 - fidelity,
            "iterations": completed_iterations,
            "fidelity_history": fidelity_history,
        },
        control_times_fs=control_times,
        controls_v_per_m=control_data,
        target_index=target_idx,
        control_layout=ControlLayout.PIECEWISE_CONSTANT_INTERVALS,
        electric_field=None,
    )


__all__ = ["run_krotov_optimization"]
