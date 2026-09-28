from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from rovibrational_excitation.dynamics import (
    PropagationDirection,
    SchrodingerPropagator,
)
from rovibrational_excitation.dynamics.utils import cm_to_rad_phz
from rovibrational_excitation.fields import ElectricField
from rovibrational_excitation.optimization.timegrid import (
    build_optimization_time_settings,
    sample_optimization_output,
)

from .krotov_initial_field import parse_krotov_initial_field
from .objective import IndexedTargetPopulation
from .options import validate_algorithm_options
from .result import ControlLayout, OptimizationResult
from .spectral_constraints import parse_legacy_spectral_constraint

DEFAULT_PARAMS: dict[str, Any] = {
    "max_iter": 1000,
    "convergence_tol": 1.0e-18,
    "lambda_a": 1.0e-20,
    "target_fidelity": 1.0,
    "propagator_func": None,
}


def _shape_function(t: np.ndarray, T: float) -> np.ndarray:
    return np.sin(np.pi * t / T) ** 2


def run_legacy_batch_overlap_optimization(
    *,
    basis: Any,
    hamiltonian: Any,
    dipole: Any,
    states: Mapping[str, Any],
    time_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
) -> OptimizationResult:
    initial_field = parse_krotov_initial_field(params)
    control_axes = validate_algorithm_options("legacy_batch_overlap", params)

    initial_state = tuple(states["initial"])  # (v,J,...) expected
    target_state = tuple(states["target"]) if states.get("target") is not None else None

    initial_idx = basis.get_index(initial_state)
    target_idx = basis.get_index(target_state) if target_state is not None else None
    if target_idx is None:
        raise ValueError("legacy_batch_overlap requires a target state.")

    time_settings = build_optimization_time_settings(time_cfg)
    time_grid = time_settings.grid
    output_stride = time_settings.output_stride

    max_iter = int(params.get("max_iter", DEFAULT_PARAMS["max_iter"]))
    convergence_tol = float(
        params.get("convergence_tol", DEFAULT_PARAMS["convergence_tol"])
    )
    lambda_a = float(params.get("lambda_a", DEFAULT_PARAMS["lambda_a"]))
    target_fidelity = float(
        params.get("target_fidelity", DEFAULT_PARAMS["target_fidelity"])
    )
    propagator_func = params.get("propagator_func", DEFAULT_PARAMS["propagator_func"])

    # Build the canonical RK4 field grid from its explicit field spacing.
    tlist = time_grid.field_times_fs
    n_field_steps = len(tlist)

    # Propagator
    propagator = SchrodingerPropagator(
        backend="numpy", validate_units=True, renorm=True
    )

    # States
    psi_initial = np.zeros(basis.size(), dtype=complex)
    psi_initial[initial_idx] = 1.0
    psi_target = np.zeros(basis.size(), dtype=complex)
    psi_target[target_idx] = 1.0

    # Dipole in propagation units
    mu_x_si = dipole.get_mu_x_SI()
    mu_y_si = dipole.get_mu_y_SI()
    mu_z_si = dipole.get_mu_z_SI()
    if hasattr(mu_x_si, "toarray"):
        mu_x_si = mu_x_si.toarray()
    if hasattr(mu_y_si, "toarray"):
        mu_y_si = mu_y_si.toarray()
    if hasattr(mu_z_si, "toarray"):
        mu_z_si = mu_z_si.toarray()
    mu_x_prime = cm_to_rad_phz(mu_x_si)
    mu_y_prime = cm_to_rad_phz(mu_y_si)
    mu_z_prime = cm_to_rad_phz(mu_z_si)
    mu_map = {
        "x": mu_x_prime,
        "y": mu_y_prime,
        "z": mu_z_prime,
    }
    mu_a_prime = mu_map[control_axes[0]]
    mu_b_prime = mu_map[control_axes[1]]

    field_data = initial_field.samples_on(time_grid)

    # Precompute helpers
    S_t = _shape_function(tlist, tlist[-1])
    dt_fs = float(tlist[1] - tlist[0])
    # Frequency grid for potential spectral constraints (PHz = cycles/fs)
    freq_phz = np.fft.rfftfreq(n_field_steps, d=dt_fs)

    # Compile the already-strict legacy-only constraint on this exact rFFT grid.
    spectral_constraint = (
        parse_legacy_spectral_constraint(params["spectrum_constraints"])
        if "spectrum_constraints" in params
        else None
    )
    spectral_filter = (
        spectral_constraint.compile(freq_phz)
        if spectral_constraint is not None
        else None
    )

    def forward(ef_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ef = ElectricField.from_time_grid(time_grid)
        ef.add_arbitrary_Efield(ef_data, field_units="V/m")
        result = propagator._propagate_array(
            hamiltonian=hamiltonian,
            efield=ef,
            dipole_matrix=dipole,
            initial_state=psi_initial,
            axes=control_axes,
            return_traj=True,
            return_time_psi=True,
            sample_stride=1,
            algorithm="rk4",
            sparse=False,
            propagator_func=propagator_func,
        )
        return result[0], result[1]

    def backward(
        ef_data: np.ndarray, psi_traj: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        psi_final = psi_traj[-1]
        overlap = np.vdot(psi_target, psi_final)
        chi_T = overlap * psi_target
        ef = ElectricField.from_time_grid(time_grid)
        ef.add_arbitrary_Efield(ef_data, field_units="V/m")
        result = propagator._propagate_array(
            hamiltonian=hamiltonian,
            efield=ef,
            dipole_matrix=dipole,
            initial_state=chi_T,
            axes=control_axes,
            return_traj=True,
            return_time_psi=True,
            sample_stride=1,
            algorithm="rk4",
            sparse=False,
            propagator_func=propagator_func,
            direction=PropagationDirection.BACKWARD,
        )
        time_b = -result[0][::-1]
        chi_traj = result[1][::-1]
        return time_b, chi_traj

    target_evaluator = IndexedTargetPopulation(target_idx)

    prev_fid = -1.0
    for it in range(max_iter):
        # Forward
        time_f, psi_traj = forward(field_data)
        fid = target_evaluator.evaluate(psi_traj[-1]).fidelity

        # Convergence checks
        if fid >= target_fidelity:
            break
        if prev_fid >= 0 and abs(fid - prev_fid) < convergence_tol:
            # small change
            pass
        prev_fid = fid

        # Backward
        _, chi_traj = backward(field_data, psi_traj)

        # Apply the historical batch update, optionally through its spectral filter.
        n_traj = len(psi_traj)
        delta_field = np.zeros_like(field_data)
        for i in range(n_traj):
            jf = i * 2
            if jf >= n_field_steps:
                break
            psi_i = psi_traj[i]
            chi_i = chi_traj[i]
            grad_x = -2.0 * float(np.imag(np.vdot(chi_i, (mu_a_prime @ psi_i))))
            grad_y = -2.0 * float(np.imag(np.vdot(chi_i, (mu_b_prime @ psi_i))))
            S = float(S_t[jf])
            dEx = (S / lambda_a) * grad_x
            dEy = (S / lambda_a) * grad_y
            delta_field[jf, 0] += dEx
            delta_field[jf, 1] += dEy
            if jf + 1 < n_field_steps:
                delta_field[jf + 1, 0] += dEx
                delta_field[jf + 1, 1] += dEy

        if spectral_filter is not None:
            # Solve the explicitly configured frequency-domain filtered update.
            field_data = field_data + spectral_filter.apply(delta_field)
        else:
            # The absent optional constraint selects the documented unfiltered update.
            field_data = field_data + delta_field

    # Final forward for outputs
    ef_total = ElectricField.from_time_grid(time_grid)
    ef_total.add_arbitrary_Efield(field_data, field_units="V/m")
    internal_time, internal_trajectory = forward(field_data)
    fidelity = target_evaluator.evaluate(internal_trajectory[-1]).fidelity
    time_full, psi_traj_full = sample_optimization_output(
        internal_time,
        internal_trajectory,
        output_stride=output_stride,
    )

    return OptimizationResult(
        electric_field=ef_total,
        trajectory_times_fs=time_full,
        trajectory=psi_traj_full,
        metrics={"fidelity": fidelity},
        control_times_fs=tlist,
        controls_v_per_m=field_data,
        target_index=target_idx,
        control_layout=ControlLayout.RK4_FIELD_SAMPLES,
    )
