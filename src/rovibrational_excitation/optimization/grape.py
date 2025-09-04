from __future__ import annotations

from typing import Any, TypedDict

import numpy as np

from rovibrational_excitation.core.electric_field import ElectricField
from rovibrational_excitation.core.propagation import SchrodingerPropagator
from rovibrational_excitation.core.propagation.utils import cm_to_rad_phz

DEFAULT_PARAMS = {
    "max_iter": 200,
    "convergence_tol": 1e-18,
    "learning_rate": 5e18,
    "lambda_a": 1e-19,
    "target_fidelity": 1.0,
    "propagator_func": None,
}

class RunResult(TypedDict, total=False):
    efield: ElectricField
    time: np.ndarray
    psi_traj: np.ndarray
    metrics: dict
    tlist: np.ndarray
    field_data: np.ndarray
    target_idx: int


def _rk4_consistent_tlist(time_total: float, dt: float) -> np.ndarray:
    target_traj_steps = int(time_total / dt) + 1
    required_field_steps = 2 * (target_traj_steps - 1) + 1
    return np.linspace(0.0, time_total, required_field_steps)


def run_grape_optimization(*, basis, hamiltonian, dipole, states: dict[str, Any], time_cfg: dict, params: dict) -> RunResult:
    initial_state = tuple(states["initial"])  # (v,J,...) expected
    target_state = tuple(states["target"]) if states.get("target") is not None else None

    initial_idx = basis.get_index(initial_state)
    target_idx = basis.get_index(target_state) if target_state is not None else None
    if target_idx is None:
        raise ValueError("GRAPE requires a target state.")

    time_total = float(time_cfg["total_fs"])
    dt = float(time_cfg["dt_fs"])
    sample_stride = int(time_cfg.get("sample_stride", 1))

    max_iter = int(params.get("max_iter", DEFAULT_PARAMS["max_iter"]))
    convergence_tol = float(params.get("convergence_tol", DEFAULT_PARAMS["convergence_tol"]))
    learning_rate = float(params.get("learning_rate", DEFAULT_PARAMS["learning_rate"]))
    lambda_a = float(params.get("lambda_a", DEFAULT_PARAMS["lambda_a"]))
    target_fidelity = float(params.get("target_fidelity", DEFAULT_PARAMS["target_fidelity"]))
    propagator_func = params.get("propagator_func", DEFAULT_PARAMS["propagator_func"])
    tlist = _rk4_consistent_tlist(time_total, dt)
    n_field_steps = len(tlist)

    propagator = SchrodingerPropagator(backend="numpy", validate_units=True, renorm=True)

    psi_initial = np.zeros(basis.size(), dtype=complex)
    psi_initial[initial_idx] = 1.0
    psi_target = np.zeros(basis.size(), dtype=complex)
    psi_target[target_idx] = 1.0

    mu_x_si = dipole.get_mu_x_SI(); mu_y_si = dipole.get_mu_y_SI()
    if hasattr(mu_x_si, 'toarray'): mu_x_si = mu_x_si.toarray()
    if hasattr(mu_y_si, 'toarray'): mu_y_si = mu_y_si.toarray()
    mu_x_prime = cm_to_rad_phz(mu_x_si)
    mu_y_prime = cm_to_rad_phz(mu_y_si)

    field_data = np.zeros((n_field_steps, 2), dtype=float)

    def forward(ef_data: np.ndarray, initial_state_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ef = ElectricField(tlist=tlist)
        ef.add_arbitrary_Efield(ef_data)
        result = propagator.propagate(
            hamiltonian=hamiltonian,
            efield=ef,
            dipole_matrix=dipole,
            initial_state=initial_state_vec,
            axes="xy",
            return_traj=True,
            return_time_psi=True,
            sample_stride=sample_stride,
            algorithm="rk4",
            sparse=True,
            propagator_func=propagator_func,
        )
        return result[0], result[1]

    def fidelity_of(psi: np.ndarray) -> float:
        return float(np.abs(psi[target_idx]) ** 2)

    prev_fid = -1.0
    for it in range(max_iter):
        # Forward
        time_f, psi_traj = forward(field_data, psi_initial)
        fid = fidelity_of(psi_traj[-1])
        if fid >= target_fidelity:
            break
        if prev_fid >= 0 and abs(fid - prev_fid) < convergence_tol:
            pass
        prev_fid = fid

        # Simple local gradient (time-local) akin to GRAPE step
        n_traj = len(psi_traj)
        grad = np.zeros_like(field_data)
        for i in range(n_traj):
            jf = i * 2
            if jf >= n_field_steps:
                break
            psi_i = psi_traj[i]
            # Heuristic gradient to increase population at target via dipole coupling
            grad_x = float(np.imag(np.vdot(psi_target, (-mu_x_prime @ psi_i))))
            grad_y = float(np.imag(np.vdot(psi_target, (-mu_y_prime @ psi_i))))
            grad[jf, 0] -= 2.0 * grad_x
            grad[jf, 1] -= 2.0 * grad_y
            if jf + 1 < n_field_steps:
                grad[jf + 1, 0] -= 2.0 * grad_x
                grad[jf + 1, 1] -= 2.0 * grad_y
        # Running cost gradient
        grad += lambda_a * field_data

        # Gradient descent step
        field_data = field_data - float(learning_rate) * grad

    ef_total = ElectricField(tlist=tlist)
    ef_total.add_arbitrary_Efield(field_data)
    time_full, psi_traj_full = forward(field_data, psi_initial)
    fidelity = fidelity_of(psi_traj_full[-1])

    return RunResult(
        efield=ef_total,
        time=time_full,
        psi_traj=psi_traj_full,
        metrics={"fidelity": fidelity},
        tlist=tlist,
        field_data=field_data,
        target_idx=target_idx,
    )


