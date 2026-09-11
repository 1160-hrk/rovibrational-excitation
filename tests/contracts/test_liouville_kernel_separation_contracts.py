"""Characterization contracts for separating the NumPy Liouville kernel."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from rovibrational_excitation.dynamics.algorithms.rk4.lvne import (
    rk4_lvne,
    rk4_lvne_traj,
)


def _complex_liouville_problem() -> tuple[np.ndarray, ...]:
    h0 = np.array(
        [[0.1, 0.02 - 0.03j], [0.02 + 0.03j, 0.7]],
        dtype=np.complex128,
    )
    mu_x = np.array(
        [[0.05, 0.4 + 0.1j], [0.4 - 0.1j, -0.02]],
        dtype=np.complex128,
    )
    mu_y = np.array(
        [[0.0, -0.2j], [0.2j, 0.03]],
        dtype=np.complex128,
    )
    field_x = np.array([0.2, -0.1, 0.3, 0.05, -0.2, 0.4, 0.1])
    field_y = np.array([-0.15, 0.2, 0.1, -0.05, 0.25, -0.1, 0.3])
    rho0 = np.array(
        [[0.65, 0.1 + 0.05j], [0.1 - 0.05j, 0.35]],
        dtype=np.complex128,
    )
    return h0, mu_x, mu_y, field_x, field_y, rho0


def test_liouville_wrappers_preserve_frozen_complex_trajectory_and_final_state() -> (
    None
):
    problem = _complex_liouville_problem()
    snapshots = tuple(array.copy() for array in problem)

    trajectory = rk4_lvne_traj(*problem, dt=0.025, steps=3, sample_stride=2)
    final = rk4_lvne(*problem, dt=0.025, steps=3)

    expected_trajectory = np.array(
        [
            [[0.65 + 0.0j, 0.1 + 0.05j], [0.1 - 0.05j, 0.35 + 0.0j]],
            [
                [0.6497835595986577 + 0.0j, 0.09872914760839287 + 0.05308104408271012j],
                [0.09872914760839287 - 0.05308104408271012j, 0.3502164404013422 + 0.0j],
            ],
        ],
        dtype=np.complex128,
    )
    expected_final = np.array(
        [
            [0.6497531773938858 + 0.0j, 0.0982767719721641 + 0.05400065597029492j],
            [0.0982767719721641 - 0.05400065597029492j, 0.35024682260611417 + 0.0j],
        ],
        dtype=np.complex128,
    )

    np.testing.assert_allclose(trajectory, expected_trajectory, rtol=0.0, atol=1.0e-17)
    np.testing.assert_allclose(final, expected_final, rtol=0.0, atol=1.0e-17)
    assert trajectory.shape == (2, 2, 2)
    assert final.shape == (2, 2)
    for array, snapshot in zip(problem, snapshots, strict=True):
        np.testing.assert_array_equal(array, snapshot)


def test_liouville_numpy_kernel_is_separate_from_validation_boundary() -> None:
    from rovibrational_excitation.dynamics.algorithms.rk4.liouville_numpy import (
        rk4_liouville_numpy_dense,
    )

    root = Path(__file__).resolve().parents[2]
    kernel_path = (
        root
        / "src"
        / "rovibrational_excitation"
        / "dynamics"
        / "algorithms"
        / "rk4"
        / "liouville_numpy.py"
    )
    boundary_path = kernel_path.with_name("lvne.py")
    kernel_source = kernel_path.read_text()
    boundary_source = boundary_path.read_text()
    kernel_tree = ast.parse(kernel_source, filename=str(kernel_path))

    assert rk4_liouville_numpy_dense.__module__.endswith("rk4.liouville_numpy")
    assert "validate_density_matrix_problem" not in kernel_source
    assert "@njit" not in boundary_source
    assert "validate_density_matrix_problem" in boundary_source
    for node in ast.walk(kernel_tree):
        if isinstance(node, ast.Import):
            imported = {alias.name for alias in node.names}
            assert imported <= {"numpy", "numba"}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
