from __future__ import annotations

from typing import Any

import numpy as np

from rovibrational_excitation.dynamics import SchrodingerPropagator
from rovibrational_excitation.dynamics.utils import cm_to_rad_phz
from rovibrational_excitation.fields import ElectricField
from rovibrational_excitation.optimization.timegrid import (
    build_optimization_time_settings,
    sample_optimization_output,
)

from .grape_rk4 import evaluate_discrete_rk4
from .krotov_initial_field import parse_grape_initial_field
from .objective import DiscreteL2TargetObjective, IndexedTargetPopulation
from .options import validate_algorithm_options
from .result import ControlLayout, OptimizationResult

DEFAULT_PARAMS = {
    "max_iter": 200,
    "convergence_tol": 1e-18,
    "learning_rate": 5e18,
    "lambda_a": 1e-19,
    "target_fidelity": 1.0,
}


def run_grape_optimization(
    *, basis, hamiltonian, dipole, states: dict[str, Any], time_cfg: dict, params: dict
) -> OptimizationResult:
    """Optimize terminal target population with an exact discrete RK4 gradient.

    The minimized objective is ``1 - fidelity + lambda_a / 2 * sum(E**2)``.
    Its gradient differentiates the normalized dense NumPy RK4 computation,
    including shared left/midpoint/right field samples.  A generated or sampled
    non-zero-capable seed is explicit because target-population fidelity has a
    zero first derivative at the zero field for the usual diagonal-H0 transfer
    problem.
    """
    initial_field = parse_grape_initial_field(params)
    control_axes = validate_algorithm_options("grape", params)

    initial_state = tuple(states["initial"])
    target_state = tuple(states["target"]) if states.get("target") is not None else None

    initial_idx = basis.get_index(initial_state)
    target_idx = basis.get_index(target_state) if target_state is not None else None
    if target_idx is None:
        raise ValueError("GRAPE requires a target state.")

    time_settings = build_optimization_time_settings(time_cfg)
    time_grid = time_settings.grid
    output_stride = time_settings.output_stride

    max_iter = int(params.get("max_iter", DEFAULT_PARAMS["max_iter"]))
    convergence_tol = float(
        params.get("convergence_tol", DEFAULT_PARAMS["convergence_tol"])
    )
    learning_rate = float(params.get("learning_rate", DEFAULT_PARAMS["learning_rate"]))
    lambda_a = float(params.get("lambda_a", DEFAULT_PARAMS["lambda_a"]))
    target_fidelity = float(
        params.get("target_fidelity", DEFAULT_PARAMS["target_fidelity"])
    )
    tlist = time_grid.field_times_fs

    psi_initial = np.zeros(basis.size(), dtype=np.complex128)
    psi_initial[initial_idx] = 1.0
    psi_target = np.zeros(basis.size(), dtype=np.complex128)
    psi_target[target_idx] = 1.0

    mu_si: dict[str, np.ndarray] = {}
    for axis in control_axes:
        component = getattr(dipole, f"get_mu_{axis}_SI")()
        if hasattr(component, "toarray"):
            component = component.toarray()
        mu_si[axis] = np.asarray(component)
    dipoles_prime = (
        np.asarray(cm_to_rad_phz(mu_si[control_axes[0]]), dtype=np.complex128),
        np.asarray(cm_to_rad_phz(mu_si[control_axes[1]]), dtype=np.complex128),
    )

    field_data = initial_field.samples_on(time_grid)
    h0_rad_per_fs = None
    if max_iter > 0:
        h0_rad_per_fs = np.asarray(
            hamiltonian.get_matrix("rad/fs"), dtype=np.complex128
        )

    prev_fid = -1.0
    for _ in range(max_iter):
        assert h0_rad_per_fs is not None
        evaluation = evaluate_discrete_rk4(
            h0_rad_per_fs=h0_rad_per_fs,
            dipoles_rad_per_fs_per_v_per_m=dipoles_prime,
            field_v_per_m=field_data,
            initial_state=psi_initial,
            target_state=psi_target,
            propagation_dt_fs=time_grid.propagation_dt_fs,
            lambda_a=lambda_a,
        )
        fid = evaluation.fidelity
        if fid >= target_fidelity:
            break
        if prev_fid >= 0 and abs(fid - prev_fid) < convergence_tol:
            # The historical solver observed this condition but did not stop.
            pass
        prev_fid = fid
        field_data = field_data - learning_rate * evaluation.gradient

    propagator = SchrodingerPropagator(
        backend="numpy", validate_units=True, renorm=True
    )
    ef_total = ElectricField.from_time_grid(time_grid)
    ef_total.add_arbitrary_Efield(field_data, field_units="V/m")
    internal_time, internal_trajectory = propagator._propagate_array(
        hamiltonian=hamiltonian,
        efield=ef_total,
        dipole_matrix=dipole,
        initial_state=psi_initial,
        axes=control_axes,
        return_traj=True,
        return_time_psi=True,
        sample_stride=1,
        algorithm="rk4",
        sparse=False,
    )
    target_evaluation = IndexedTargetPopulation(target_idx).evaluate(
        internal_trajectory[-1]
    )
    fidelity = target_evaluation.fidelity
    objective = DiscreteL2TargetObjective(lambda_a).evaluate(
        target_evaluation,
        field_data,
    )
    time_full, psi_traj_full = sample_optimization_output(
        internal_time,
        internal_trajectory,
        output_stride=output_stride,
    )

    return OptimizationResult(
        electric_field=ef_total,
        trajectory_times_fs=time_full,
        trajectory=psi_traj_full,
        metrics={"fidelity": fidelity, "objective": objective},
        control_times_fs=tlist,
        controls_v_per_m=field_data,
        target_index=target_idx,
        control_layout=ControlLayout.RK4_FIELD_SAMPLES,
    )
