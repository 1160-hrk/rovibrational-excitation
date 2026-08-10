"""Polarization contracts for the spectral split-operator propagator."""

import numpy as np
import pytest

from rovibrational_excitation.core.basis import LinMolBasis
from rovibrational_excitation.core.electric_field import ElectricField
from rovibrational_excitation.core.propagation import SchrodingerPropagator
from rovibrational_excitation.core.propagation.algorithms.rk4.schrodinger import (
    rk4_schrodinger,
)
from rovibrational_excitation.core.propagation.algorithms.split_operator.schrodinger import (
    build_helicity_projected_interaction,
    splitop_schrodinger,
)
from rovibrational_excitation.dipole.linmol import LinMolDipoleMatrix


def _linmol_problem():
    basis = LinMolBasis(
        V_max=1,
        J_max=1,
        use_M=True,
        omega=1.0,
        B=0.05,
        input_units="rad/fs",
        output_units="rad/fs",
        alpha=0.0,
        delta_omega=0.0,
    )
    h0 = basis.generate_H0().matrix
    dipole = LinMolDipoleMatrix(basis, mu0=1.0, potential_type="harmonic")
    initial = np.zeros(basis.size(), dtype=np.complex128)
    initial[basis.get_index((0, 0, 0))] = 1.0
    return basis, h0, np.asarray(dipole.mu_x), np.asarray(dipole.mu_y), initial


@pytest.mark.parametrize("angle", [0.2, 0.7, -1.1])
def test_linmol_cartesian_dipoles_rotate_with_m_phase(angle):
    basis, _h0, mu_x, mu_y, _initial = _linmol_problem()
    rotation = np.exp(1j * basis.M_array * angle)
    rotated = rotation[:, None] * mu_x * rotation.conj()[None, :]
    expected = np.cos(angle) * mu_x + np.sin(angle) * mu_y
    np.testing.assert_allclose(rotated, expected, rtol=0.0, atol=1.0e-15)


def test_helicity_projected_operator_selects_delta_m_sign():
    basis, _h0, mu_x, mu_y, _initial = _linmol_problem()
    ground = basis.get_index((0, 0, 0))
    excited_plus = basis.get_index((1, 1, 1))
    excited_minus = basis.get_index((1, 1, -1))

    sigma_plus = np.array([1.0, 1.0j]) / np.sqrt(2.0)
    sigma_minus = np.array([1.0, -1.0j]) / np.sqrt(2.0)
    interaction_plus = build_helicity_projected_interaction(mu_x, mu_y, sigma_plus)
    interaction_minus = build_helicity_projected_interaction(mu_x, mu_y, sigma_minus)

    assert abs(interaction_plus[excited_plus, ground]) > 0.0
    assert interaction_plus[excited_minus, ground] == 0.0
    assert interaction_minus[excited_plus, ground] == 0.0
    assert abs(interaction_minus[excited_minus, ground]) > 0.0
    np.testing.assert_array_equal(interaction_plus, interaction_plus.conj().T)
    np.testing.assert_array_equal(interaction_minus, interaction_minus.conj().T)


def test_helicity_projected_rejects_nonhermitian_dipole():
    _basis, _h0, mu_x, mu_y, _initial = _linmol_problem()
    mu_x = mu_x.copy()
    mu_x[0, 4] += 0.1

    with pytest.raises(ValueError, match="mu_x must be Hermitian"):
        build_helicity_projected_interaction(
            mu_x,
            mu_y,
            np.array([1.0, 1.0j]) / np.sqrt(2.0),
        )


def test_helicity_projected_requires_normalized_jones_vector():
    _basis, _h0, mu_x, mu_y, _initial = _linmol_problem()

    with pytest.raises(ValueError, match="must be normalized"):
        build_helicity_projected_interaction(mu_x, mu_y, np.array([1.0, 1.0j]))


def _circular_final_state(step):
    basis, h0, mu_x, mu_y, initial = _linmol_problem()
    duration = 0.4
    steps = round(duration / step)
    field_times = np.linspace(0.0, duration, 2 * steps + 1)
    amplitude = 0.08
    carrier = 1.1
    field_x = amplitude * np.cos(carrier * field_times) / np.sqrt(2.0)
    field_y = -amplitude * np.sin(carrier * field_times) / np.sqrt(2.0)

    rk4_final = rk4_schrodinger(
        h0,
        mu_x,
        mu_y,
        field_x,
        field_y,
        initial,
        step,
        return_traj=False,
    )[0]
    split_final = splitop_schrodinger(
        h0,
        mu_x,
        mu_y,
        field_x,
        field_y,
        initial,
        step,
        return_traj=False,
        magnetic_quantum_numbers=basis.M_array,
    )[0]
    return rk4_final, split_final


def test_cartesian_circular_split_converges_quadratically_to_rk4():
    rk4_coarse, split_coarse = _circular_final_state(0.02)
    rk4_fine, split_fine = _circular_final_state(0.01)
    coarse_error = float(np.max(np.abs(split_coarse - rk4_coarse)))
    fine_error = float(np.max(np.abs(split_fine - rk4_fine)))

    assert fine_error < coarse_error / 3.9
    assert fine_error < 2.0e-7
    np.testing.assert_allclose(np.linalg.norm(split_fine), 1.0, rtol=0.0, atol=2.0e-14)


def test_changing_cartesian_direction_requires_m_labels():
    _basis, h0, mu_x, mu_y, initial = _linmol_problem()
    field_x = np.array([1.0, 0.0, -1.0])
    field_y = np.array([0.0, 1.0, 0.0])

    with pytest.raises(ValueError, match="magnetic_quantum_numbers"):
        splitop_schrodinger(h0, mu_x, mu_y, field_x, field_y, initial, 0.1)


def _flat_envelope(time, center, duration):
    del center, duration
    return np.ones_like(time)


def test_high_level_circular_modes_use_explicit_physics_contracts():
    basis = LinMolBasis(
        V_max=1,
        J_max=1,
        use_M=True,
        omega=1.0,
        B=0.05,
        input_units="rad/fs",
        output_units="rad/fs",
        alpha=0.0,
        delta_omega=0.0,
    )
    hamiltonian = basis.generate_H0()
    dipole = LinMolDipoleMatrix(basis, mu0=1.0e-30, potential_type="harmonic")
    field = ElectricField(np.linspace(0.0, 0.2, 201))
    field.add_dispersed_Efield(
        _flat_envelope,
        duration=1.0,
        t_center=0.0,
        carrier_freq=0.16,
        amplitude=2.0e8,
        polarization=np.array([1.0, 1.0j]) / np.sqrt(2.0),
        const_polarisation=True,
    )
    initial = np.zeros(basis.size(), dtype=np.complex128)
    initial[basis.get_index((0, 0, 0))] = 1.0

    rk4_final = SchrodingerPropagator(algorithm="rk4", validate_units=False).propagate(
        hamiltonian, field, dipole, initial, return_traj=False
    )
    cartesian_final = SchrodingerPropagator(
        algorithm="split_operator",
        split_interaction="cartesian",
        validate_units=False,
    ).propagate(hamiltonian, field, dipole, initial, return_traj=False)
    projected_final = SchrodingerPropagator(
        algorithm="split_operator",
        split_interaction="helicity_projected",
        validate_units=False,
    ).propagate(hamiltonian, field, dipole, initial, return_traj=False)

    np.testing.assert_allclose(cartesian_final, rk4_final, rtol=0.0, atol=1.0e-10)
    excited_plus = basis.get_index((1, 1, 1))
    excited_minus = basis.get_index((1, 1, -1))
    assert abs(projected_final[excited_plus]) > 0.0
    assert projected_final[excited_minus] == 0.0
