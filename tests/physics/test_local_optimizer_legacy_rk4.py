"""Numerical equivalence of the local optimizer's legacy RK4 field view."""

from __future__ import annotations

import numpy as np

from rovibrational_excitation.dynamics.algorithms.rk4.schrodinger import (
    _rk4_cpu_numba,
    rk4_schrodinger,
)
from rovibrational_excitation.optimization.timegrid import (
    LocalOptimizerLegacyGridV1,
)


def test_odd_prefix_is_bitwise_identical_to_legacy_even_length_rk4() -> None:
    h0 = np.array([[0.2, 0.03], [0.03, 0.7]], dtype=np.complex128)
    mu_x = np.array([[0.0, 0.4], [0.4, 0.0]], dtype=np.complex128)
    mu_y = np.array([[0.0, -0.2j], [0.2j, 0.0]], dtype=np.complex128)
    ex = np.array([0.0, 0.1, 0.2, -0.1, 0.3, 0.2, -0.2, 0.1, 0.0, 1e9])
    ey = np.array([0.0, -0.2, 0.1, 0.3, -0.1, 0.2, 0.4, -0.3, 0.0, -1e9])
    psi0 = np.array([1.0 + 0.0j, 0.0 + 0.0j])
    dt = 0.2

    legacy_result = _rk4_cpu_numba(
        h0,
        mu_x,
        mu_y,
        ex,
        ey,
        psi0,
        dt,
        True,
        1,
        True,
    )
    grid = LocalOptimizerLegacyGridV1(
        segments=[(0, 4), (4, 8)],
        tlist=np.arange(0.0, 1.0, 0.1),
    )
    effective = grid.full_rk4_slice
    validated_result = rk4_schrodinger(
        h0,
        mu_x,
        mu_y,
        ex[effective],
        ey[effective],
        psi0,
        dt,
        return_traj=True,
        stride=1,
        renorm=True,
        sparse=False,
        backend="numpy",
    )

    np.testing.assert_array_equal(validated_result, legacy_result)
