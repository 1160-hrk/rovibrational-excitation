"""Independent VibLadder references for the Phase 0 physics baseline.

Energies are angular frequencies in rad/fs. The fixed tolerance of 2e-15 is
near float64 roundoff for the O(1) analytic values used below.
"""

from __future__ import annotations

import numpy as np
import pytest

from rovibrational_excitation.core.basis import VibLadderBasis
from rovibrational_excitation.core.electric_field import ElectricField
from rovibrational_excitation.core.propagation import SchrodingerPropagator
from rovibrational_excitation.dipole import VibLadderDipoleMatrix
from rovibrational_excitation.dipole.vib.harmonic import tdm_vib_harm
from rovibrational_excitation.dipole.vib.morse import (
    omega01_domega_to_N,
    tdm_vib_morse,
)

pytestmark = pytest.mark.physics

OMEGA01_RAD_PER_FS = 1.2
ANHARMONIC_SHIFT_RAD_PER_FS = 0.08
V_MAX = 4
ENERGY_ATOL = 2.0e-15
DIPOLE_C_M = 2.0e-29
FIELD_DT_FS = 1.0e-3
FINAL_TIME_FS = 0.1
FIELD_V_PER_M = 5.0e8
PROPAGATION_ATOL = 2.0e-12


def _expected_energies(
    levels: np.ndarray,
    omega01: float,
    anharmonic_shift: float,
) -> np.ndarray:
    vterm = levels + 0.5
    return (omega01 + anharmonic_shift) * vterm - (anharmonic_shift / 2) * vterm**2


def test_harmonic_energies_and_adjacent_spacings():
    """For delta=0, E_v=omega01(v+1/2) and every spacing is omega01."""
    basis = VibLadderBasis(
        V_max=V_MAX,
        omega=OMEGA01_RAD_PER_FS,
        delta_omega=0.0,
        input_units="rad/fs",
        output_units="rad/fs",
    )
    energies = np.diag(basis.generate_H0().matrix)
    levels = np.arange(V_MAX + 1)

    np.testing.assert_allclose(
        energies,
        OMEGA01_RAD_PER_FS * (levels + 0.5),
        rtol=0.0,
        atol=ENERGY_ATOL,
    )
    np.testing.assert_allclose(
        np.diff(energies),
        OMEGA01_RAD_PER_FS,
        rtol=0.0,
        atol=ENERGY_ATOL,
    )


def test_anharmonic_energies_and_spacings_follow_omega01_convention():
    """E_v=(omega01+delta)x-delta*x^2/2 gives dE_v=omega01-v*delta."""
    basis = VibLadderBasis(
        V_max=V_MAX,
        omega=OMEGA01_RAD_PER_FS,
        delta_omega=ANHARMONIC_SHIFT_RAD_PER_FS,
        input_units="rad/fs",
        output_units="rad/fs",
    )
    energies = np.diag(basis.generate_H0().matrix)
    levels = np.arange(V_MAX + 1)

    np.testing.assert_allclose(
        energies,
        _expected_energies(
            levels,
            OMEGA01_RAD_PER_FS,
            ANHARMONIC_SHIFT_RAD_PER_FS,
        ),
        rtol=0.0,
        atol=ENERGY_ATOL,
    )
    np.testing.assert_allclose(
        np.diff(energies),
        OMEGA01_RAD_PER_FS - levels[:-1] * ANHARMONIC_SHIFT_RAD_PER_FS,
        rtol=0.0,
        atol=ENERGY_ATOL,
    )


def test_stored_and_override_hamiltonian_paths_use_the_same_formula():
    """Stored and temporary parameters must not reinterpret omega01 or delta."""
    configured = VibLadderBasis(
        V_max=V_MAX,
        omega=OMEGA01_RAD_PER_FS,
        delta_omega=ANHARMONIC_SHIFT_RAD_PER_FS,
        input_units="rad/fs",
        output_units="rad/fs",
    )
    override_host = VibLadderBasis(
        V_max=V_MAX,
        omega=0.5,
        delta_omega=0.0,
        input_units="rad/fs",
        output_units="rad/fs",
    )

    stored = configured.generate_H0().matrix
    overridden = override_host.generate_H0_with_params(
        omega=OMEGA01_RAD_PER_FS,
        delta_omega=ANHARMONIC_SHIFT_RAD_PER_FS,
        input_units="rad/fs",
        units="rad/fs",
    ).matrix

    np.testing.assert_allclose(overridden, stored, rtol=0.0, atol=ENERGY_ATOL)


def _flat_envelope(
    time_fs: np.ndarray, center_fs: float, duration_fs: float
) -> np.ndarray:
    """Return an exactly constant envelope; center and duration are irrelevant."""
    del center_fs, duration_fs
    return np.ones_like(time_fs)


def _constant_field(polarization: np.ndarray) -> ElectricField:
    intervals = round(FINAL_TIME_FS / FIELD_DT_FS)
    time_fs = np.linspace(0.0, FINAL_TIME_FS, intervals + 1)
    field = ElectricField(time_fs)
    field.add_dispersed_Efield(
        _flat_envelope,
        duration=1.0,
        t_center=0.0,
        carrier_freq=0.0,
        amplitude=FIELD_V_PER_M,
        polarization=polarization,
        const_polarisation=True,
    )
    return field


def _propagate_vib_ladder(
    polarization: np.ndarray,
    *,
    nondimensional: bool = False,
):
    basis = VibLadderBasis(
        V_max=2,
        omega=0.37,
        delta_omega=0.015,
        input_units="rad/fs",
        output_units="rad/fs",
    )
    dipole = VibLadderDipoleMatrix(
        basis,
        mu0=DIPOLE_C_M,
        potential_type="harmonic",
        units="C*m",
        units_input="C*m",
    )
    initial = np.array([np.sqrt(0.6), np.sqrt(0.3) * np.exp(0.17j), np.sqrt(0.1)])
    return SchrodingerPropagator(validate_units=False).propagate(
        basis.generate_H0(),
        _constant_field(polarization),
        dipole,
        initial,
        coupling_mode="scalar",
        coupling_axis="z",
        return_traj=True,
        return_time_psi=True,
        nondimensional=nondimensional,
    )


def test_harmonic_transition_selection_rule_and_elements():
    """Harmonic mu has <v+1|mu|v>/mu0=sqrt(v+1) and no other elements."""
    basis = VibLadderBasis(
        V_max=V_MAX,
        omega=OMEGA01_RAD_PER_FS,
        input_units="rad/fs",
        output_units="rad/fs",
    )
    matrix = VibLadderDipoleMatrix(
        basis,
        mu0=1.0,
        potential_type="harmonic",
    ).mu_z

    for v1 in range(V_MAX + 1):
        for v2 in range(V_MAX + 1):
            expected = np.sqrt(max(v1, v2)) if abs(v1 - v2) == 1 else 0.0
            assert tdm_vib_harm(v1, v2) == pytest.approx(expected)
            assert matrix[v1, v2] == pytest.approx(expected)


@pytest.mark.parametrize(
    ("omega01", "anharmonic_shift"),
    [(1.0, 0.1), (0.9, 0.2)],
)
def test_morse_level_parameter_is_derived_from_each_parameter_pair(
    omega01,
    anharmonic_shift,
):
    """The Morse parameter is N=(omega01+delta)/delta-1/2."""
    expected = (omega01 + anharmonic_shift) / anharmonic_shift - 0.5
    assert omega01_domega_to_N(omega01, anharmonic_shift) == pytest.approx(expected)


def test_morse_rejects_zero_anharmonicity():
    basis = VibLadderBasis(
        V_max=1,
        omega=1.0,
        delta_omega=0.0,
        input_units="rad/fs",
    )

    with pytest.raises(ValueError, match="delta_omega must be non-zero"):
        VibLadderDipoleMatrix(basis, potential_type="morse")


def test_morse_accepts_maximum_bound_level_and_rejects_next_level():
    """For omega01=1 and delta=0.1, N=10.5 and max V is floor(N)-1=9."""
    accepted_basis = VibLadderBasis(
        V_max=9,
        omega=1.0,
        delta_omega=0.1,
        input_units="rad/fs",
    )
    accepted = VibLadderDipoleMatrix(accepted_basis, potential_type="morse")
    assert accepted.mu_z.shape == (10, 10)

    rejected_basis = VibLadderBasis(
        V_max=10,
        omega=1.0,
        delta_omega=0.1,
        input_units="rad/fs",
    )
    with pytest.raises(ValueError, match="Morse limit 9"):
        VibLadderDipoleMatrix(rejected_basis, potential_type="morse")


def test_morse_parameters_do_not_leak_between_instances():
    """Construct B before evaluating A so any shared mutable N would corrupt A."""
    basis_a = VibLadderBasis(
        V_max=4,
        omega=1.0,
        delta_omega=0.1,
        input_units="rad/fs",
    )
    basis_b = VibLadderBasis(
        V_max=4,
        omega=0.9,
        delta_omega=0.2,
        input_units="rad/fs",
    )
    dipole_a = VibLadderDipoleMatrix(basis_a, potential_type="morse")
    dipole_b = VibLadderDipoleMatrix(basis_b, potential_type="morse")

    matrix_b = dipole_b.mu_z
    matrix_a = dipole_a.mu_z
    expected_a = tdm_vib_morse(0, 2, 10.5)
    expected_b = tdm_vib_morse(0, 2, 5.0)

    assert matrix_a[0, 2] == pytest.approx(expected_a)
    assert matrix_b[0, 2] == pytest.approx(expected_b)
    assert abs(matrix_a[0, 2] - matrix_b[0, 2]) > 1.0e-3


@pytest.mark.parametrize(
    "polarization",
    [
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([1.0, 1.0]) / np.sqrt(2.0),
        np.array([1.0, 1.0j]) / np.sqrt(2.0),
    ],
)
def test_scalar_workflow_is_independent_of_polarization(polarization):
    """VibLadder scalar coupling has no laboratory polarization direction."""
    reference_time, reference = _propagate_vib_ladder(np.array([1.0, 0.0]))
    time_fs, trajectory = _propagate_vib_ladder(polarization)

    np.testing.assert_array_equal(time_fs, reference_time)
    np.testing.assert_allclose(trajectory, reference, rtol=0.0, atol=2.0e-14)


def test_dimensional_and_nondimensional_population_and_time_agree():
    """Nondimensional scaling preserves physical output times and populations."""
    polarization = np.array([1.0, 1.0]) / np.sqrt(2.0)
    time_dim, psi_dim = _propagate_vib_ladder(polarization, nondimensional=False)
    time_nd, psi_nd = _propagate_vib_ladder(polarization, nondimensional=True)

    np.testing.assert_allclose(time_nd, time_dim, rtol=0.0, atol=ENERGY_ATOL)
    np.testing.assert_allclose(
        np.abs(psi_nd) ** 2,
        np.abs(psi_dim) ** 2,
        rtol=0.0,
        atol=PROPAGATION_ATOL,
    )


@pytest.mark.parametrize("transition", [(1, 2), (2, 3)])
def test_morse_elements_converge_toward_harmonic_limit(transition):
    """Increasing N must reduce the Morse-to-harmonic adjacent-element error."""
    harmonic = tdm_vib_harm(*transition)
    moderate_n_error = abs(tdm_vib_morse(*transition, 50.0) - harmonic)
    large_n_error = abs(tdm_vib_morse(*transition, 500.0) - harmonic)

    assert large_n_error < moderate_n_error
