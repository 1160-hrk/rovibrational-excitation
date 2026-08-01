"""Independent solver invariants and convergence references for Phase P0.6.

The RK4 convergence ratio is bounded around the analytic fourth-order value
2^4=16. Direct one-step and normalized-state comparisons use 2e-15, near
float64 roundoff for two components. The 50-step split-operator norm uses
2e-14 to cover accumulated eigendecomposition and matrix-product roundoff.
Physical time uses 2e-15 for an O(1 fs) grid, while Liouville trace and
Hermiticity use 3e-15 for 40 commutator steps. The density threshold test
derives its boundary directly from D-012 rather than selecting an empirical
tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

from rovibrational_excitation.core.basis import TwoLevelBasis
from rovibrational_excitation.core.electric_field import ElectricField
from rovibrational_excitation.core.propagation import (
    LiouvillePropagator,
    PropagatorFactory,
    SchrodingerPropagator,
)
from rovibrational_excitation.core.propagation.algorithms.rk4.lvne import (
    rk4_lvne_traj,
)
from rovibrational_excitation.core.propagation.algorithms.rk4.schrodinger import (
    rk4_schrodinger,
)
from rovibrational_excitation.core.propagation.algorithms.split_operator.schrodinger import (
    splitop_schrodinger,
)
from rovibrational_excitation.core.propagation.algorithms.validation import (
    validate_density_matrix_properties,
)
from rovibrational_excitation.dipole import TwoLevelDipoleMatrix

pytestmark = pytest.mark.physics


def _zero_fields(steps: int) -> tuple[np.ndarray, np.ndarray]:
    field = np.zeros(2 * steps + 1, dtype=np.float64)
    return field, field.copy()


def _rk4_free_final(step: float, total_time: float) -> np.ndarray:
    steps = round(total_time / step)
    assert steps * step == pytest.approx(total_time)
    h0 = np.diag([0.0, 1.7]).astype(np.complex128)
    zero = np.zeros_like(h0)
    field_x, field_y = _zero_fields(steps)
    initial = np.array(
        [np.sqrt(0.3), np.sqrt(0.7) * np.exp(0.23j)], dtype=np.complex128
    )
    return rk4_schrodinger(
        h0,
        zero,
        zero,
        field_x,
        field_y,
        initial,
        step,
        return_traj=False,
    )[0]


def test_rk4_has_fourth_order_global_convergence():
    """Halving dt reduces fixed-time free-evolution error by about 2^4."""
    total_time = 2.0
    initial = np.array(
        [np.sqrt(0.3), np.sqrt(0.7) * np.exp(0.23j)], dtype=np.complex128
    )
    exact = initial * np.exp(-1j * np.array([0.0, 1.7]) * total_time)
    errors = np.array(
        [
            np.linalg.norm(_rk4_free_final(step, total_time) - exact)
            for step in (0.2, 0.1, 0.05)
        ]
    )
    ratios = errors[:-1] / errors[1:]

    assert np.all(np.diff(errors) < 0.0)
    assert np.all((ratios > 14.0) & (ratios < 18.0))


def test_rk4_norm_drift_without_renorm_matches_amplification_polynomial():
    """renorm=False exposes, rather than repairs, the RK4 truncation drift."""
    energy = 4.0
    step = 0.3
    steps = 20
    h0 = np.diag([0.0, energy]).astype(np.complex128)
    zero = np.zeros_like(h0)
    field_x, field_y = _zero_fields(steps)
    initial = np.array([0.0, 1.0], dtype=np.complex128)
    trajectory = rk4_schrodinger(
        h0,
        zero,
        zero,
        field_x,
        field_y,
        initial,
        step,
        renorm=False,
    )

    z = -1j * energy * step
    amplification = 1.0 + z + z**2 / 2.0 + z**3 / 6.0 + z**4 / 24.0
    expected_norm = abs(amplification) ** steps
    actual_norm = np.linalg.norm(trajectory[-1])

    assert abs(actual_norm - 1.0) > 1.0e-3
    assert actual_norm == pytest.approx(expected_norm, rel=2.0e-14, abs=0.0)


def test_rk4_renorm_true_normalizes_each_saved_state():
    """Current O-004 behavior normalizes after every integration step."""
    energy = 4.0
    step = 0.3
    steps = 20
    h0 = np.diag([0.0, energy]).astype(np.complex128)
    zero = np.zeros_like(h0)
    field_x, field_y = _zero_fields(steps)
    initial = np.array([0.0, 1.0], dtype=np.complex128)
    trajectory = rk4_schrodinger(
        h0,
        zero,
        zero,
        field_x,
        field_y,
        initial,
        step,
        renorm=True,
    )

    np.testing.assert_allclose(
        np.linalg.norm(trajectory, axis=1), 1.0, rtol=0.0, atol=2.0e-15
    )


def _manual_rk4_step(
    h0: np.ndarray,
    dipole: np.ndarray,
    field: np.ndarray,
    initial: np.ndarray,
    step: float,
) -> np.ndarray:
    def derivative(field_value: float, state: np.ndarray) -> np.ndarray:
        return -1j * ((h0 - dipole * field_value) @ state)

    k1 = derivative(field[0], initial)
    k2 = derivative(field[1], initial + step * k1 / 2.0)
    k3 = derivative(field[1], initial + step * k2 / 2.0)
    k4 = derivative(field[2], initial + step * k3)
    return initial + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def test_rk4_uses_left_mid_right_field_and_minus_mu_e_sign():
    """One update consumes E_left, E_mid twice, and E_right with H=H0-mu E."""
    h0 = np.diag([0.1, 0.4]).astype(np.complex128)
    dipole = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    zero_dipole = np.zeros_like(dipole)
    field = np.array([0.2, 0.7, 0.1], dtype=np.float64)
    initial = np.array([1.0, 0.0], dtype=np.complex128)
    step = 0.01

    actual = rk4_schrodinger(
        h0,
        dipole,
        zero_dipole,
        field,
        np.zeros(3),
        initial,
        step,
    )[-1]
    expected = _manual_rk4_step(h0, dipole, field, initial, step)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-15)
    assert actual[1].imag > 0.0


def test_rk4_trajectory_and_final_only_results_agree():
    h0 = np.diag([0.0, 0.8]).astype(np.complex128)
    dipole = np.array([[0.0, 0.6], [0.6, 0.0]], dtype=np.complex128)
    field = np.linspace(-0.3, 0.5, 21)
    initial = np.array([np.sqrt(0.4), 1j * np.sqrt(0.6)])
    trajectory = rk4_schrodinger(
        h0, dipole, np.zeros_like(dipole), field, np.zeros_like(field), initial, 0.02
    )
    final = rk4_schrodinger(
        h0,
        dipole,
        np.zeros_like(dipole),
        field,
        np.zeros_like(field),
        initial,
        0.02,
        return_traj=False,
    )

    assert final.shape == (1, 2)
    np.testing.assert_allclose(final[0], trajectory[-1], rtol=0.0, atol=0.0)


def test_split_operator_preserves_norm_and_final_state_contract():
    steps = 50
    step = 0.02
    h0 = np.diag([0.0, 0.8])
    dipole = np.array([[0.0, 0.6], [0.6, 0.0]], dtype=np.complex128)
    field = np.sin(np.linspace(0.0, 2.0 * np.pi, 2 * steps + 1))
    initial = np.array([np.sqrt(0.4), 1j * np.sqrt(0.6)])
    trajectory = splitop_schrodinger(
        h0,
        dipole,
        np.zeros_like(dipole),
        np.array([1.0, 0.0]),
        field,
        initial,
        step,
    )
    final = splitop_schrodinger(
        h0,
        dipole,
        np.zeros_like(dipole),
        np.array([1.0, 0.0]),
        field,
        initial,
        step,
        return_traj=False,
    )

    np.testing.assert_allclose(
        np.linalg.norm(trajectory, axis=1), 1.0, rtol=0.0, atol=2.0e-14
    )
    np.testing.assert_allclose(final[0], trajectory[-1], rtol=0.0, atol=0.0)


def test_split_operator_rejects_non_diagonal_h0():
    h0 = np.array([[0.0, 0.1], [0.1, 0.8]], dtype=np.complex128)
    dipole = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    with pytest.raises(ValueError, match="requires a diagonal H0"):
        splitop_schrodinger(
            h0,
            dipole,
            np.zeros_like(dipole),
            np.array([1.0, 0.0]),
            np.zeros(3),
            np.array([1.0, 0.0]),
            0.1,
        )


def _flat_envelope(time: np.ndarray, center: float, duration: float) -> np.ndarray:
    del center, duration
    return np.ones_like(time)


def _high_level_time_problem():
    field = ElectricField(np.linspace(1.0, 1.5, 11))
    field.add_dispersed_Efield(
        _flat_envelope,
        duration=1.0,
        t_center=0.0,
        carrier_freq=0.0,
        amplitude=0.0,
        polarization=np.array([1.0, 0.0]),
        const_polarisation=True,
    )
    basis = TwoLevelBasis(
        energy_gap=0.3,
        input_units="rad/fs",
        output_units="rad/fs",
    )
    dipole = TwoLevelDipoleMatrix(basis, mu0=1.0e-30)
    return field, basis.generate_H0(), dipole


def test_physical_time_uses_two_field_intervals_and_characterizes_stride_endpoint():
    """O-001 currently keeps a regular stride grid and omits a remainder endpoint."""
    field, hamiltonian, dipole = _high_level_time_problem()
    initial = np.array([np.sqrt(0.3), np.sqrt(0.7)], dtype=np.complex128)
    solver = SchrodingerPropagator(validate_units=False)
    full_time, full = solver.propagate(
        hamiltonian,
        field,
        dipole,
        initial,
        coupling_mode="scalar",
        coupling_axis="x",
        return_time_psi=True,
    )
    stride_time, stride = solver.propagate(
        hamiltonian,
        field,
        dipole,
        initial,
        coupling_mode="scalar",
        coupling_axis="x",
        return_time_psi=True,
        sample_stride=2,
    )
    final_time, final = solver.propagate(
        hamiltonian,
        field,
        dipole,
        initial,
        coupling_mode="scalar",
        coupling_axis="x",
        return_traj=False,
        return_time_psi=True,
    )

    np.testing.assert_allclose(full_time, np.linspace(1.0, 1.5, 6), atol=2.0e-15)
    np.testing.assert_allclose(stride_time, [1.0, 1.2, 1.4], atol=2.0e-15)
    np.testing.assert_allclose(stride, full[::2], rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(final_time, [1.5])
    np.testing.assert_allclose(final, full[-1], rtol=0.0, atol=0.0)


def test_liouville_preserves_trace_and_hermiticity_without_projection():
    steps = 40
    step = 0.01
    h0 = np.diag([0.0, 0.9]).astype(np.complex128)
    dipole = np.array([[0.1, 0.7], [0.7, -0.2]], dtype=np.complex128)
    field = np.sin(np.linspace(0.0, np.pi, 2 * steps + 1))
    initial_pure = np.array([np.sqrt(0.4), np.sqrt(0.6) * np.exp(0.31j)])
    rho0 = np.outer(initial_pure, initial_pure.conj())
    trajectory = rk4_lvne_traj(
        h0,
        dipole,
        np.zeros_like(dipole),
        field,
        np.zeros_like(field),
        rho0,
        step,
        steps,
    )

    traces = np.trace(trajectory, axis1=1, axis2=2)
    hermiticity_error = np.max(
        np.linalg.norm(trajectory - trajectory.conj().transpose(0, 2, 1), axis=(1, 2))
    )
    np.testing.assert_allclose(traces, 1.0, rtol=0.0, atol=3.0e-15)
    assert hermiticity_error < 3.0e-15


def test_density_positivity_threshold_accepts_roundoff_and_rejects_beyond_it():
    """D-012 uses 100*n*eps*||rho||_2 as its negative-eigenvalue tolerance."""
    reference_tolerance = 100.0 * 2.0 * np.finfo(np.float64).eps
    inside = 0.5 * reference_tolerance
    outside = 2.0 * reference_tolerance

    validate_density_matrix_properties(np.diag([1.0 + inside, -inside]))
    with pytest.raises(ValueError, match="positive semidefinite"):
        validate_density_matrix_properties(np.diag([1.0 + outside, -outside]))


def test_unsupported_solver_capabilities_raise_explicitly():
    h0 = np.diag([0.0, 1.0]).astype(np.complex128)
    dipole = np.zeros_like(h0)
    field = np.zeros(3)
    initial = np.array([1.0, 0.0])

    with pytest.raises(ValueError, match="backend must be"):
        rk4_schrodinger(h0, dipole, dipole, field, field, initial, 0.1, backend="jax")
    with pytest.raises(ValueError, match="only backend='numpy'"):
        LiouvillePropagator(backend="cupy", validate_units=False)
    with pytest.raises(ValueError, match="only supports pure states"):
        PropagatorFactory.create_propagator(
            state_type="mixed",
            algorithm="split_operator",
            validate_units=False,
        )
