"""Reference behavior for GRAPE/Krotov time grids and backward RK4."""

from __future__ import annotations

import numpy as np
import pytest

from rovibrational_excitation.core.electric_field import ElectricField
from rovibrational_excitation.core.propagation.algorithms.rk4.schrodinger import (
    rk4_schrodinger,
)
from rovibrational_excitation.optimization.grape import (
    _rk4_consistent_tlist as grape_tlist,
)
from rovibrational_excitation.optimization.krotov import (
    _rk4_consistent_tlist as krotov_tlist,
)


@pytest.mark.parametrize("total_fs", [200.0, 500.0, 1000.0])
def test_grape_and_krotov_share_the_existing_repository_time_grid(
    total_fs: float,
) -> None:
    propagation_dt_fs = 0.1
    propagation_steps = int(total_fs / propagation_dt_fs)
    expected = np.linspace(0.0, total_fs, 2 * propagation_steps + 1)

    grape_grid = grape_tlist(total_fs, propagation_dt_fs)
    krotov_grid = krotov_tlist(total_fs, propagation_dt_fs)

    np.testing.assert_array_equal(grape_grid, expected)
    np.testing.assert_array_equal(krotov_grid, expected)
    assert grape_grid[1] - grape_grid[0] == pytest.approx(0.05)
    assert grape_grid[2] - grape_grid[0] == pytest.approx(0.1)


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
        ElectricField(tlist=np.linspace(1.0, 0.0, 5))
