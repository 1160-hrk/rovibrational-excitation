"""Independent references for linear-molecule and M-averaged workflows."""

from __future__ import annotations

import numpy as np
import pytest

from rovibrational_excitation.core.basis import LinMolBasis
from rovibrational_excitation.core.electric_field import ElectricField, gaussian_fwhm
from rovibrational_excitation.core.propagation import SchrodingerPropagator
from rovibrational_excitation.dipole import LinMolDipoleMatrix
from rovibrational_excitation.simulation.models.linmol_m_average import (
    build_m_average_blocks,
    canonicalize_fixed_linear_polarization,
)
from rovibrational_excitation.simulation.runner import _run_one
from rovibrational_excitation.simulation.timegrid import build_time_grid
from rovibrational_excitation.simulation.validation import (
    SimulationConfigurationError,
)

pytestmark = pytest.mark.physics

OMEGA_RAD_PER_FS = 0.37
ANHARMONIC_SHIFT_RAD_PER_FS = 0.015
ROTATION_RAD_PER_FS = 0.004
VIBRATION_ROTATION_RAD_PER_FS = 0.0002
DIPOLE_C_M = 2.0e-29
ENERGY_ATOL = 2.0e-15
PROPAGATION_ATOL = 3.0e-13


def _runner_params(**overrides):
    params = {
        "basis_type": "linmol",
        "V_max": 1,
        "J_max": 2,
        "use_M": False,
        "omega_rad_phz": OMEGA_RAD_PER_FS,
        "delta_omega_rad_phz": ANHARMONIC_SHIFT_RAD_PER_FS,
        "B_rad_phz": ROTATION_RAD_PER_FS,
        "alpha_rad_phz": VIBRATION_ROTATION_RAD_PER_FS,
        "mu0_Cm": DIPOLE_C_M,
        "potential_type": "harmonic",
        "t_start": 0.0,
        "t_end": 0.04,
        "dt": 0.001,
        "duration": 0.03,
        "t_center": 0.02,
        "carrier_freq": 0.0,
        "amplitude": 4.0e8,
        "polarization": [1.0, 0.0],
        "initial_states": [1],  # reduced state |v=0,J=1>
        "backend": "numpy",
        "algorithm": "rk4",
        "validate_units": False,
        "return_traj": True,
        "save": False,
    }
    params.update(overrides)
    return params


@pytest.mark.parametrize("use_m", [False, True])
def test_state_index_round_trip_and_basis_size(use_m):
    basis = LinMolBasis(
        V_max=2,
        J_max=3,
        use_M=use_m,
        omega=OMEGA_RAD_PER_FS,
        B=ROTATION_RAD_PER_FS,
        input_units="rad/fs",
        alpha=0.0,
        delta_omega=0.0,
    )
    expected_size = 3 * (16 if use_m else 4)
    assert basis.size() == expected_size
    for index in range(basis.size()):
        state = tuple(int(value) for value in basis.get_state(index))
        assert basis.get_index(state) == index


def test_low_lying_rovibrational_energies_follow_documented_formula():
    basis = LinMolBasis(
        V_max=2,
        J_max=2,
        use_M=True,
        omega=OMEGA_RAD_PER_FS,
        delta_omega=ANHARMONIC_SHIFT_RAD_PER_FS,
        B=ROTATION_RAD_PER_FS,
        alpha=VIBRATION_ROTATION_RAD_PER_FS,
        input_units="rad/fs",
        output_units="rad/fs",
    )
    energies = np.diag(basis.generate_H0().matrix)
    for index, (v, j, _m) in enumerate(basis.basis):
        x = v + 0.5
        expected = (
            (OMEGA_RAD_PER_FS + ANHARMONIC_SHIFT_RAD_PER_FS) * x
            - ANHARMONIC_SHIFT_RAD_PER_FS * x**2 / 2.0
            + (ROTATION_RAD_PER_FS - VIBRATION_ROTATION_RAD_PER_FS * x) * j * (j + 1)
        )
        assert energies[index] == pytest.approx(expected, abs=ENERGY_ATOL)


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_cartesian_dipoles_are_hermitian_and_obey_selection_rules(axis):
    basis = LinMolBasis(
        V_max=2,
        J_max=2,
        use_M=True,
        omega=OMEGA_RAD_PER_FS,
        B=ROTATION_RAD_PER_FS,
        input_units="rad/fs",
        alpha=0.0,
        delta_omega=0.0,
    )
    matrix = LinMolDipoleMatrix(basis, mu0=1.0, potential_type="harmonic").mu(axis)
    np.testing.assert_allclose(matrix, matrix.conj().T, rtol=0.0, atol=2e-15)

    rows, columns = np.nonzero(np.abs(matrix) > 0.0)
    for row, column in zip(rows, columns):
        v1, j1, m1 = basis.get_state(row)
        v2, j2, m2 = basis.get_state(column)
        assert abs(v1 - v2) == 1
        assert abs(j1 - j2) == 1
        if axis == "z":
            assert m1 == m2
        else:
            assert abs(m1 - m2) == 1


def test_dense_and_sparse_dipoles_agree():
    basis = LinMolBasis(
        V_max=1,
        J_max=2,
        use_M=True,
        omega=OMEGA_RAD_PER_FS,
        B=ROTATION_RAD_PER_FS,
        input_units="rad/fs",
        alpha=0.0,
        delta_omega=0.0,
    )
    dense = LinMolDipoleMatrix(
        basis, mu0=DIPOLE_C_M, dense=True, potential_type="harmonic"
    )
    sparse = LinMolDipoleMatrix(
        basis, mu0=DIPOLE_C_M, dense=False, potential_type="harmonic"
    )
    for axis in "xyz":
        np.testing.assert_allclose(
            sparse.mu(axis).toarray(), dense.mu(axis), rtol=0.0, atol=0.0
        )


def test_reduced_basis_cannot_be_silently_treated_as_m_zero_dipole():
    basis = LinMolBasis(
        V_max=1,
        J_max=1,
        use_M=False,
        omega=OMEGA_RAD_PER_FS,
        B=ROTATION_RAD_PER_FS,
        input_units="rad/fs",
        alpha=0.0,
        delta_omega=0.0,
    )
    with pytest.raises(ValueError, match="M-averaged simulation workflow"):
        LinMolDipoleMatrix(basis, mu0=DIPOLE_C_M, potential_type="harmonic")


def _resolved_propagation(initial, polarization, *, sparse=False):
    time_grid = build_time_grid(0.0, 0.08, 0.001)
    field = ElectricField(time_grid)
    field.add_dispersed_Efield(
        gaussian_fwhm,
        duration=0.06,
        t_center=0.04,
        carrier_freq=0.0,
        amplitude=8.0e8,
        polarization=np.asarray(polarization),
    )
    basis = LinMolBasis(
        V_max=1,
        J_max=2,
        use_M=True,
        omega=OMEGA_RAD_PER_FS,
        delta_omega=ANHARMONIC_SHIFT_RAD_PER_FS,
        B=ROTATION_RAD_PER_FS,
        alpha=VIBRATION_ROTATION_RAD_PER_FS,
        input_units="rad/fs",
        output_units="J",
    )
    dipole = LinMolDipoleMatrix(
        basis,
        mu0=DIPOLE_C_M,
        dense=not sparse,
        potential_type="harmonic",
    )
    trajectory = SchrodingerPropagator(validate_units=False, sparse=sparse).propagate(
        basis.generate_H0(),
        field,
        dipole,
        initial,
        axes="xz",
        return_traj=True,
        sparse=sparse,
    )
    return basis, trajectory


def _resolved_initial_state(state):
    basis = LinMolBasis(
        V_max=1,
        J_max=2,
        use_M=True,
        omega=OMEGA_RAD_PER_FS,
        B=ROTATION_RAD_PER_FS,
        input_units="rad/fs",
        alpha=0.0,
        delta_omega=0.0,
    )
    initial = np.zeros(basis.size(), dtype=np.complex128)
    initial[basis.get_index(state)] = 1.0
    return basis, initial


def test_explicit_m_mode_distinguishes_x_and_z_polarization():
    _basis, initial = _resolved_initial_state((0, 1, 0))
    _basis, x_trajectory = _resolved_propagation(initial, [1.0, 0.0])
    _basis, z_trajectory = _resolved_propagation(initial, [0.0, 1.0])
    difference = np.max(
        np.abs(np.abs(x_trajectory[-1]) ** 2 - np.abs(z_trajectory[-1]) ** 2)
    )
    assert difference > 1.0e-7


def test_explicit_m_dense_and_sparse_propagation_agree():
    _basis, initial = _resolved_initial_state((0, 1, 0))
    _basis, dense = _resolved_propagation(initial, [1.0, 0.0], sparse=False)
    _basis, sparse = _resolved_propagation(initial, [1.0, 0.0], sparse=True)
    np.testing.assert_allclose(sparse, dense, rtol=0.0, atol=PROPAGATION_ATOL)


def test_coherent_superposition_has_cross_terms_absent_from_incoherent_sum():
    basis, state_a = _resolved_initial_state((0, 0, 0))
    _basis, state_b = _resolved_initial_state((0, 2, 0))

    _basis, coherent = _resolved_propagation(
        (state_a + state_b) / np.sqrt(2.0), [0.0, 1.0]
    )
    _basis, propagated_a = _resolved_propagation(state_a, [0.0, 1.0])
    _basis, propagated_b = _resolved_propagation(state_b, [0.0, 1.0])
    incoherent_population = (
        np.abs(propagated_a) ** 2 + np.abs(propagated_b) ** 2
    ) / 2.0
    coherent_population = np.abs(coherent) ** 2

    common_target = basis.get_index((1, 1, 0))
    difference = abs(
        coherent_population[-1, common_target]
        - incoherent_population[-1, common_target]
    )
    assert difference > 1.0e-7


def test_fixed_linear_polarization_removes_only_common_jones_phase():
    expected = np.array([1.0, -2.0]) / np.sqrt(5.0)
    vector = np.exp(0.37j) * expected
    actual = canonicalize_fixed_linear_polarization(vector)
    np.testing.assert_allclose(
        np.outer(actual, actual),
        np.outer(expected, expected),
        rtol=0.0,
        atol=2.0e-15,
    )


@pytest.mark.parametrize(
    "polarization",
    [
        [1.0, 1.0j],
        [1.0, 0.2j],
    ],
)
def test_m_average_rejects_circular_and_elliptical_polarization(polarization):
    with pytest.raises(SimulationConfigurationError, match="fixed linear"):
        _run_one(_runner_params(polarization=polarization))


def test_m_average_rejects_coherent_initial_states_spanning_different_j():
    with pytest.raises(ValueError, match="spanning different J"):
        _run_one(_runner_params(initial_states=[0, 1]))


def test_m_average_rejects_inapplicable_cartesian_axes_mapping():
    with pytest.raises(SimulationConfigurationError, match="not applicable"):
        _run_one(_runner_params(axes="xy"))


def test_m_average_accepts_coherent_vibrational_superposition_within_one_j():
    population = _run_one(_runner_params(initial_states=[1, 4], amplitude=0.0))
    np.testing.assert_allclose(population.sum(axis=1), 1.0, atol=2.0e-14)
    np.testing.assert_allclose(
        population[0, [1, 4]], np.array([0.5, 0.5]), atol=2.0e-15
    )


def _full_m_reference(params):
    time_grid = build_time_grid(params["t_start"], params["t_end"], params["dt"])
    field = ElectricField(time_grid)
    field.add_dispersed_Efield(
        gaussian_fwhm,
        duration=params["duration"],
        t_center=params["t_center"],
        carrier_freq=params["carrier_freq"],
        amplitude=params["amplitude"],
        polarization=np.array([1.0, 0.0]),
    )
    basis = LinMolBasis(
        params["V_max"],
        params["J_max"],
        use_M=True,
        omega=params["omega_rad_phz"],
        delta_omega=params["delta_omega_rad_phz"],
        B=params["B_rad_phz"],
        alpha=params["alpha_rad_phz"],
        input_units="rad/fs",
        output_units="J",
    )
    dipole = LinMolDipoleMatrix(basis, mu0=params["mu0_Cm"], potential_type="harmonic")
    propagator = SchrodingerPropagator(validate_units=False)
    full_population = None
    initial_j = 1
    for m in range(-initial_j, initial_j + 1):
        initial = np.zeros(basis.size(), dtype=np.complex128)
        initial[basis.get_index((0, initial_j, m))] = 1.0
        _time, wavefunction = propagator.propagate(
            basis.generate_H0(),
            field,
            dipole,
            initial,
            coupling_mode="scalar",
            coupling_axis="z",
            return_traj=True,
            return_time_psi=True,
        )
        contribution = np.abs(wavefunction) ** 2 / (2 * initial_j + 1)
        if full_population is None:
            full_population = contribution
        else:
            full_population += contribution

    assert full_population is not None
    reduced = np.zeros(
        (full_population.shape[0], (params["V_max"] + 1) * (params["J_max"] + 1))
    )
    reduced_indices = basis.V_array * (params["J_max"] + 1) + basis.J_array
    for full_index, reduced_index in enumerate(reduced_indices):
        reduced[:, reduced_index] += full_population[:, full_index]
    return reduced


def test_separate_m_blocks_equal_full_resolved_incoherent_reference():
    params = _runner_params()
    reduced = _run_one(params)
    reference = _full_m_reference(params)
    np.testing.assert_allclose(reduced, reference, rtol=0.0, atol=PROPAGATION_ATOL)


@pytest.mark.parametrize(
    "polarization",
    [[1.0, 0.0], [0.0, 1.0], [1.0, -2.0]],
)
def test_m_average_is_independent_of_fixed_linear_lab_direction(polarization):
    reference = _run_one(_runner_params(polarization=[1.0, 0.0]))
    actual = _run_one(_runner_params(polarization=polarization))
    np.testing.assert_allclose(actual, reference, rtol=0.0, atol=2.0e-14)


def test_block_weights_are_normalized_and_reduce_dense_work():
    params = _runner_params(V_max=2, J_max=5, initial_states=[2])
    blocks = build_m_average_blocks(params)
    assert [block.abs_m for block in blocks] == [0, 1, 2]
    np.testing.assert_allclose(
        [block.weight for block in blocks], [1 / 5, 2 / 5, 2 / 5]
    )
    assert sum(block.weight for block in blocks) == pytest.approx(1.0)

    full_dimension = (params["V_max"] + 1) * (params["J_max"] + 1) ** 2
    block_matrix_elements = sum(block.basis.size() ** 2 for block in blocks)
    assert block_matrix_elements < full_dimension**2


def test_saved_m_average_has_no_fictitious_aggregate_wavefunction(tmp_path):
    params = _runner_params(save=True, outdir=str(tmp_path))
    _run_one(params)
    with np.load(tmp_path / "result.npz", allow_pickle=False) as result:
        assert result["representation"] == "m_incoherent_average"
        assert "psi" not in result.files
        assert "psi_abs_m_0" in result.files
        assert "psi_abs_m_1" in result.files
        np.testing.assert_allclose(result["m_weight"].sum(), 1.0)
