from __future__ import annotations

from typing import Any, TypedDict

import numpy as np

from rovibrational_excitation.core.electric_field import ElectricField
from rovibrational_excitation.core.propagation import SchrodingerPropagator
from rovibrational_excitation.core.propagation.utils import cm_to_rad_phz
from rovibrational_excitation.core.electric_field import gaussian_fwhm
from rovibrational_excitation.core.units.converters import converter

DEFAULT_PARAMS = {
    "max_iter": 1000,
    "convergence_tol": 1.0e-18,
    "lambda_a": 1.0e-20,
    "target_fidelity": 1.0,
    "control_axes": "xy",
    "duration_initial": 200.0,
    "carrier_freq_initial": 2300,
    "unit_carrier_freq": "cm^-1",
    "amplitude_initial": 1.0e9,
    "pol_initial": [1.0, 1.0],
    "gdd_initial": 0.0,
    "tod_initial": 0.0,
    "const_polarisation": False,
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


def _shape_function(t: np.ndarray, T: float) -> np.ndarray:
    return np.sin(np.pi * t / T) ** 2


def _rk4_consistent_tlist(time_total: float, dt: float) -> np.ndarray:
    target_traj_steps = int(time_total / dt) + 1
    required_field_steps = 2 * (target_traj_steps - 1) + 1
    return np.linspace(0.0, time_total, required_field_steps)


def run_krotov_optimization(*, basis, hamiltonian, dipole, states: dict[str, Any], time_cfg: dict, params: dict) -> RunResult:
    initial_state = tuple(states["initial"])  # (v,J,...) expected
    target_state = tuple(states["target"]) if states.get("target") is not None else None

    initial_idx = basis.get_index(initial_state)
    target_idx = basis.get_index(target_state) if target_state is not None else None
    if target_idx is None:
        raise ValueError("Krotov requires a target state.")

    time_total = float(time_cfg["total_fs"])
    dt = float(time_cfg["dt_fs"])
    sample_stride = int(time_cfg.get("sample_stride", 1))

    max_iter = int(params.get("max_iter", DEFAULT_PARAMS["max_iter"]))
    convergence_tol = float(params.get("convergence_tol", DEFAULT_PARAMS["convergence_tol"]))
    lambda_a = float(params.get("lambda_a", DEFAULT_PARAMS["lambda_a"]))
    target_fidelity = float(params.get("target_fidelity", DEFAULT_PARAMS["target_fidelity"]))
    propagator_func = params.get("propagator_func", DEFAULT_PARAMS["propagator_func"])

    # Build time list consistent with RK4 propagator
    tlist = _rk4_consistent_tlist(time_total, dt)
    n_field_steps = len(tlist)

    # Propagator
    propagator = SchrodingerPropagator(backend="numpy", validate_units=True, renorm=True)

    # States
    psi_initial = np.zeros(basis.size(), dtype=complex)
    psi_initial[initial_idx] = 1.0
    psi_target = np.zeros(basis.size(), dtype=complex)
    psi_target[target_idx] = 1.0

    # Dipole in propagation units
    mu_x_si = dipole.get_mu_x_SI()
    mu_y_si = dipole.get_mu_y_SI()
    mu_z_si = dipole.get_mu_z_SI()
    if hasattr(mu_x_si, 'toarray'): mu_x_si = mu_x_si.toarray()
    if hasattr(mu_y_si, 'toarray'): mu_y_si = mu_y_si.toarray()
    if hasattr(mu_z_si, 'toarray'): mu_z_si = mu_z_si.toarray()
    mu_x_prime = cm_to_rad_phz(mu_x_si)
    mu_y_prime = cm_to_rad_phz(mu_y_si)
    mu_z_prime = cm_to_rad_phz(mu_z_si)
    mu_map = {
        "x": mu_x_prime,
        "y": mu_y_prime,
        "z": mu_z_prime,
    }
    control_axes = params.get("control_axes", DEFAULT_PARAMS["control_axes"])
    mu_a_prime = mu_map[control_axes[0]]
    mu_b_prime = mu_map[control_axes[1]]
    

    # Initial field (gaussian)
    Efield_test = ElectricField(tlist=tlist)
    carrier_freq = params.get("carrier_freq_initial", DEFAULT_PARAMS["carrier_freq_initial"])
    unit_carrier_freq = params.get("unit_carrier_freq", DEFAULT_PARAMS["unit_carrier_freq"])
    carrier_freq_phz = converter.convert_frequency(carrier_freq, unit_carrier_freq, "PHz") # PHz
    Efield_test.add_dispersed_Efield(
        envelope_func=gaussian_fwhm,
        duration=params.get("duration_initial", time_total / 2),
        t_center=params.get("t_center_initial", time_total / 2),
        carrier_freq=float(carrier_freq_phz), # PHz
        duration_units="fs",
        t_center_units="fs",
        carrier_freq_units="PHz",
        amplitude=params.get("amplitude_initial", DEFAULT_PARAMS["amplitude_initial"]),
        polarization=params.get("pol_initial", DEFAULT_PARAMS["pol_initial"]),
        phase_rad=0.0,
        gdd=params.get("gdd_initial", DEFAULT_PARAMS["gdd_initial"]),
        tod=params.get("tod_initial", DEFAULT_PARAMS["tod_initial"]),
        gdd_units="fs^2",
        tod_units="fs^3",
        const_polarisation=params.get("const_polarisation", DEFAULT_PARAMS["const_polarisation"]),
    )
    field_data = params.get("efield_initial", Efield_test.get_Efield_SI())

    # Precompute helpers
    S_t = _shape_function(tlist, tlist[-1])

    def forward(ef_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ef = ElectricField(tlist=tlist)
        ef.add_arbitrary_Efield(ef_data)
        result = propagator.propagate(
            hamiltonian=hamiltonian,
            efield=ef,
            dipole_matrix=dipole,
            initial_state=psi_initial,
            axes=control_axes,
            return_traj=True,
            return_time_psi=True,
            sample_stride=sample_stride,
            algorithm="rk4",
            sparse=True,
            propagator_func=propagator_func,
        )
        return result[0], result[1]

    def backward(ef_data: np.ndarray, psi_traj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        psi_final = psi_traj[-1]
        overlap = np.vdot(psi_target, psi_final)
        chi_T = overlap * psi_target
        # Reverse-time field
        ef = ElectricField(tlist=tlist[::-1])
        ef.add_arbitrary_Efield(ef_data[::-1])
        result = propagator.propagate(
            hamiltonian=hamiltonian,
            efield=ef,
            dipole_matrix=dipole,
            initial_state=chi_T,
            axes=control_axes,
            return_traj=True,
            return_time_psi=True,
            sample_stride=sample_stride,
            algorithm="rk4",
            sparse=True,
            propagator_func=propagator_func,
        )
        time_b = -result[0][::-1]
        chi_traj = result[1][::-1]
        return time_b, chi_traj

    def fidelity_of(psi: np.ndarray) -> float:
        return float(np.abs(psi[target_idx]) ** 2)

    prev_fid = -1.0
    for it in range(max_iter):
        # Forward
        time_f, psi_traj = forward(field_data)
        fid = fidelity_of(psi_traj[-1])

        # Convergence checks
        if fid >= target_fidelity:
            break
        if prev_fid >= 0 and abs(fid - prev_fid) < convergence_tol:
            # small change
            pass
        prev_fid = fid

        # Backward
        _, chi_traj = backward(field_data, psi_traj)

        # Update (time-local Krotov-style)
        n_traj = len(psi_traj)
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
            field_data[jf, 0] += dEx
            field_data[jf, 1] += dEy
            if jf + 1 < n_field_steps:
                field_data[jf + 1, 0] += dEx
                field_data[jf + 1, 1] += dEy

    # Final forward for outputs
    ef_total = ElectricField(tlist=tlist)
    ef_total.add_arbitrary_Efield(field_data)
    time_full, psi_traj_full = forward(field_data)
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


