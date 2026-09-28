"""Reference behavior for GRAPE/Krotov time grids and backward RK4."""

from __future__ import annotations

import numpy as np
import pytest

from rovibrational_excitation.core.operators import Hamiltonian
from rovibrational_excitation.dynamics import (
    PropagationDirection,
    SchrodingerPropagator,
)
from rovibrational_excitation.dynamics.algorithms.rk4.schrodinger import (
    rk4_schrodinger,
)
from rovibrational_excitation.fields import ElectricField
from rovibrational_excitation.models.two_level import (
    TwoLevelBasis,
    TwoLevelDipoleMatrix,
)
from rovibrational_excitation.optimization.legacy_batch_overlap import (
    run_legacy_batch_overlap_optimization,
)
from rovibrational_excitation.optimization.timegrid import (
    build_optimization_time_settings,
)


@pytest.mark.parametrize("total_fs", [200.0, 500.0, 1000.0])
def test_grape_and_krotov_share_the_existing_repository_time_grid(
    total_fs: float,
) -> None:
    propagation_dt_fs = 0.1
    propagation_steps = int(total_fs / propagation_dt_fs)
    expected = np.linspace(0.0, total_fs, 2 * propagation_steps + 1)

    settings = build_optimization_time_settings(
        {"total_fs": total_fs, "field_dt_fs": propagation_dt_fs / 2.0}
    )
    migrated_grid = settings.grid.field_times_fs

    np.testing.assert_array_equal(migrated_grid, expected)
    assert migrated_grid[1] - migrated_grid[0] == pytest.approx(0.05)
    assert migrated_grid[2] - migrated_grid[0] == pytest.approx(0.1)


def _manual_rk4(
    h0: np.ndarray,
    mu_x: np.ndarray,
    field_x: np.ndarray,
    psi0: np.ndarray,
    dt: float,
) -> np.ndarray:
    psi = psi0.copy()
    trajectory = [psi.copy()]
    for step in range((field_x.size - 1) // 2):
        left = 2 * step

        def rhs(state: np.ndarray, sample: int) -> np.ndarray:
            hamiltonian = h0 - field_x[sample] * mu_x
            return -1j * (hamiltonian @ state)

        k1 = rhs(psi, left)
        k2 = rhs(psi + 0.5 * dt * k1, left + 1)
        k3 = rhs(psi + 0.5 * dt * k2, left + 1)
        k4 = rhs(psi + dt * k3, left + 2)
        psi = psi + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        trajectory.append(psi.copy())
    return np.asarray(trajectory)


def test_legacy_krotov_backward_is_reversed_field_with_negative_dt() -> None:
    h0 = np.array([[0.0, 0.04], [0.04, 0.7]], dtype=np.complex128)
    mu_x = np.array([[0.0, 0.3], [0.3, 0.0]], dtype=np.complex128)
    mu_y = np.zeros_like(mu_x)
    forward_field = np.array([0.0, 0.2, -0.1, 0.4, 0.1], dtype=float)
    backward_field = forward_field[::-1].copy()
    psi_final = np.array([0.6 + 0.2j, -0.3 + 0.7j], dtype=np.complex128)
    psi_final /= np.linalg.norm(psi_final)
    backward_dt = -0.2

    expected = _manual_rk4(h0, mu_x, backward_field, psi_final, backward_dt)
    actual = rk4_schrodinger(
        h0,
        mu_x,
        mu_y,
        backward_field,
        np.zeros_like(backward_field),
        psi_final,
        backward_dt,
        return_traj=True,
        stride=1,
        renorm=False,
        sparse=False,
        backend="numpy",
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-15)


def test_current_electric_field_rejects_the_old_decreasing_time_container() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        ElectricField(tlist=np.linspace(1.0, 0.0, 5), time_units="fs")


class _ArrayDipole:
    def __init__(self, mu_x: np.ndarray, mu_y: np.ndarray) -> None:
        self._components = {
            "x": mu_x,
            "y": mu_y,
            "z": np.zeros_like(mu_x),
        }

    def get_mu_in_units(self, axis: str, units: str) -> np.ndarray:
        assert units == "rad/fs/(V/m)"
        return self._components[axis]


def test_explicit_backward_direction_reproduces_legacy_krotov_kernel() -> None:
    h0 = np.array([[0.0, 0.04], [0.04, 0.7]], dtype=np.complex128)
    mu_x = np.array([[0.0, 0.3], [0.3, 0.0]], dtype=np.complex128)
    mu_y = np.zeros_like(mu_x)
    forward_field = np.array([0.0, 0.2, -0.1, 0.4, 0.1], dtype=float)
    psi_final = np.array([0.6 + 0.2j, -0.3 + 0.7j], dtype=np.complex128)
    psi_final /= np.linalg.norm(psi_final)

    expected = rk4_schrodinger(
        h0,
        mu_x,
        mu_y,
        forward_field[::-1].copy(),
        np.zeros_like(forward_field),
        psi_final,
        -0.2,
        return_traj=True,
        stride=1,
        renorm=False,
        sparse=False,
        backend="numpy",
    )
    efield = ElectricField(tlist=np.linspace(0.0, 0.4, 5), time_units="fs")
    efield.Efield[:, 0] = forward_field
    times, actual = SchrodingerPropagator(
        validate_units=False,
        renorm=False,
    )._propagate_array(
        Hamiltonian(h0, units="rad/fs"),
        efield,
        _ArrayDipole(mu_x, mu_y),
        psi_final,
        axes="xy",
        return_traj=True,
        return_time_psi=True,
        sample_stride=1,
        algorithm="rk4",
        sparse=False,
        direction=PropagationDirection.BACKWARD,
    )

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(times, np.array([0.4, 0.2, 0.0]))


def test_backward_direction_rejects_ambiguous_or_unsupported_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h0 = Hamiltonian(np.diag([0.0, 0.7]), units="rad/fs")
    zeros = np.zeros((2, 2), dtype=np.complex128)
    dipole = _ArrayDipole(zeros, zeros)
    efield = ElectricField(tlist=np.linspace(0.0, 0.4, 5), time_units="fs")
    psi = np.array([1.0 + 0.0j, 0.0 + 0.0j])
    propagator = SchrodingerPropagator(validate_units=False)

    with pytest.raises(TypeError, match="PropagationDirection"):
        propagator._propagate_array(h0, efield, dipole, psi, direction="backward")
    with pytest.raises(ValueError, match="RK4"):
        propagator._propagate_array(
            h0,
            efield,
            dipole,
            psi,
            algorithm="split_operator",
            direction=PropagationDirection.BACKWARD,
        )
    with pytest.raises(ValueError, match="dimensional"):
        propagator._propagate_array(
            h0,
            efield,
            dipole,
            psi,
            nondimensional=True,
            direction=PropagationDirection.BACKWARD,
        )

    import rovibrational_excitation.dynamics.schrodinger as schrodinger_module

    monkeypatch.setattr(schrodinger_module, "HAS_CUPY", True)
    cupy_propagator = schrodinger_module.SchrodingerPropagator(
        backend="cupy", validate_units=False
    )
    with pytest.raises(ValueError, match="NumPy only"):
        cupy_propagator._propagate_array(
            h0,
            efield,
            dipole,
            psi,
            direction=PropagationDirection.BACKWARD,
        )


def test_krotov_completes_one_real_forward_backward_iteration() -> None:
    basis = TwoLevelBasis(
        energy_gap=0.7,
        input_units="rad/fs",
        output_units="rad/fs",
    )
    hamiltonian = basis.generate_H0()
    dipole = TwoLevelDipoleMatrix(
        basis=basis,
        mu0=2e-29,
        units="C*m",
        units_input="C*m",
    )

    result = run_legacy_batch_overlap_optimization(
        basis=basis,
        hamiltonian=hamiltonian,
        dipole=dipole,
        states={"initial": (0,), "target": (1,)},
        time_cfg={"total_fs": 0.8, "field_dt_fs": 0.1, "output_stride": 1},
        params={
            "control_axes": "xy",
            "initial_field_kind": "generated",
            "initial_duration": 0.2,
            "initial_duration_units": "fs",
            "initial_center": 0.4,
            "initial_center_units": "fs",
            "initial_carrier_frequency": 2300.0,
            "initial_carrier_frequency_units": "cm^-1",
            "initial_amplitude": 1e9,
            "initial_amplitude_units": "V/m",
            "initial_polarization": [1.0, 1.0],
            "max_iter": 1,
            "target_fidelity": 1.0,
        },
    )

    np.testing.assert_array_equal(
        result.trajectory_times_fs, np.arange(5, dtype=float) * 0.2
    )
    assert result.trajectory.shape == (5, 2)
    assert result.controls_v_per_m.shape == (9, 2)
    assert np.all(np.isfinite(result.trajectory))
    assert np.isfinite(result.metrics["fidelity"])
