"""End-to-end transfer checks for the standard interval-control Krotov route."""

from __future__ import annotations

import numpy as np
import pytest

from rovibrational_excitation.dynamics.utils import cm_to_rad_phz
from rovibrational_excitation.optimization.krotov import run_krotov_optimization
from rovibrational_excitation.optimization.krotov_rk4 import (
    propagate_interval_controls,
)
from rovibrational_excitation.optimization.model import build_optimization_model


def _generated_params(*, axes: str, penalty: float, iterations: int) -> dict:
    return {
        "control_axes": axes,
        "lambda_a": penalty,
        "lambda_a_units": "1 / ((V/m)^2 fs)",
        "max_iter": iterations,
        "target_fidelity": 0.9999,
        "initial_control_kind": "generated",
        "initial_carrier_frequency": 2300.0,
        "initial_carrier_frequency_units": "cm^-1",
        "initial_amplitude": 1.0e9,
        "initial_amplitude_units": "V/m",
        "initial_polarization": [1.0, 0.0],
        "initial_duration": 200.0,
        "initial_duration_units": "fs",
        "initial_center": 250.0,
        "initial_center_units": "fs",
    }


def _fine_final_state(model, result, axes: str, initial_index: int, dt_fs: float):
    h0 = np.asarray(model.hamiltonian.get_matrix("rad/fs"), dtype=np.complex128)
    dipoles = []
    for axis in axes:
        component = getattr(model.dipole, f"get_mu_{axis}_SI")()
        if hasattr(component, "toarray"):
            component = component.toarray()
        dipoles.append(np.asarray(cm_to_rad_phz(component), dtype=np.complex128))
    initial = np.zeros(model.basis.size(), dtype=np.complex128)
    initial[initial_index] = 1.0
    refined_controls = np.repeat(result["control_data"], 2, axis=0)
    trajectory = propagate_interval_controls(
        h0_rad_per_fs=h0,
        dipoles_rad_per_fs_per_v_per_m=(dipoles[0], dipoles[1]),
        controls_v_per_m=refined_controls,
        initial_state=initial,
        control_dt_fs=dt_fs / 2.0,
    )
    return trajectory[-1]


def test_standard_krotov_transfers_two_level_population():
    model = build_optimization_model(
        {
            "type": "twolevel",
            "params": {
                "energy_gap": 2300.0,
                "energy_gap_units": "cm^-1",
                "dipole_scale": 0.3,
                "dipole_scale_units": "D",
            },
        }
    )
    params = _generated_params(axes="xy", penalty=1.0e-20, iterations=20)
    params.update(
        {
            "initial_duration": 40.0,
            "initial_center": 50.0,
        }
    )
    result = run_krotov_optimization(
        basis=model.basis,
        hamiltonian=model.hamiltonian,
        dipole=model.dipole,
        states={"initial": (0,), "target": (1,)},
        time_cfg={"total_fs": 100.0, "control_dt_fs": 0.2, "output_stride": 100},
        params=params,
    )

    history = np.asarray(result["metrics"]["fidelity_history"])
    assert history[0] < 0.05
    assert result["metrics"]["fidelity"] > 0.999
    assert np.all(np.diff(history) >= -2e-12)
    np.testing.assert_allclose(
        np.sum(np.abs(result["psi_traj"][-1]) ** 2), 1.0, rtol=0.0, atol=2e-5
    )


@pytest.mark.slow
def test_standard_krotov_concentrates_v0_to_v3_in_five_level_ladder():
    model = build_optimization_model(
        {
            "type": "vibladder",
            "params": {
                "V_max": 4,
                "vibrational_frequency": 2349.1,
                "vibrational_frequency_units": "cm^-1",
                "anharmonic_shift": 25.0,
                "anharmonic_shift_units": "cm^-1",
                "dipole_scale": 0.3,
                "dipole_scale_units": "D",
                "potential_type": "harmonic",
            },
        }
    )
    dt_fs = 0.1
    result = run_krotov_optimization(
        basis=model.basis,
        hamiltonian=model.hamiltonian,
        dipole=model.dipole,
        states={"initial": (0,), "target": (3,)},
        time_cfg={
            "total_fs": 500.0,
            "control_dt_fs": dt_fs,
            "output_stride": 1000,
        },
        params=_generated_params(axes="zx", penalty=3.0e-21, iterations=40),
    )

    populations = np.abs(result["psi_traj"][-1]) ** 2
    history = np.asarray(result["metrics"]["fidelity_history"])
    assert populations.shape == (5,)
    assert populations[3] > 0.985
    assert np.max(np.delete(populations, 3)) < 0.01
    assert np.all(np.diff(history) >= -2e-12)
    assert abs(np.sum(populations) - 1.0) < 1.1e-3

    fine_state = _fine_final_state(model, result, "zx", 0, dt_fs)
    fine_populations = np.abs(fine_state) ** 2
    assert abs(np.sum(fine_populations) - 1.0) < 4e-5
    assert abs(fine_populations[3] - populations[3]) < 1.5e-3
