"""Strict scaling and explicit field-free contracts."""

import numpy as np
import pytest

from rovibrational_excitation.core.basis import Hamiltonian
from rovibrational_excitation.core.electric_field import ElectricField, ZeroField
from rovibrational_excitation.core.nondimensional import (
    nondimensionalize_from_objects,
    nondimensionalize_system,
)
from rovibrational_excitation.core.propagation import SchrodingerPropagator

HBAR = 1.054571817e-34


def _constant_field(values: np.ndarray) -> ElectricField:
    field = ElectricField(np.linspace(0.0, 0.2, 5))
    field.add_arbitrary_Efield(np.asarray(values, dtype=float))
    return field


def test_array_api_requires_explicit_hamiltonian_and_time_units():
    zero = np.zeros((2, 2))
    field = ZeroField(np.linspace(0.0, 0.2, 5))
    with pytest.raises(TypeError, match="H0_units"):
        nondimensionalize_system(zero, zero, zero, field)


def test_object_api_requires_explicit_coupling_semantics():
    with pytest.raises(TypeError, match="coupling_axes"):
        nondimensionalize_from_objects(None, None, None)


def test_scaling_uses_centered_spectrum_and_complete_generator_reference():
    h0 = np.array([[3.0e-21, 4.0e-22], [4.0e-22, 8.0e-21]])
    mu_x = np.array([[0.0, 2.0e-30], [2.0e-30, 0.0]])
    mu_y = np.zeros_like(mu_x)
    field = _constant_field(np.tile([2.0e8, 0.0], (5, 1)))

    h0_prime, mu_x_prime, _, field_prime, _, _, scales = nondimensionalize_system(
        h0, mu_x, mu_y, field, H0_units="energy", time_units="fs"
    )

    eigenvalues = np.linalg.eigvalsh(h0)
    free_span = eigenvalues[-1] - eigenvalues[0]
    interaction_energy = np.linalg.norm(mu_x, ord=2) * 2.0e8
    expected_reference = max(free_span, interaction_energy)

    assert scales.energy_offset == pytest.approx(eigenvalues[0])
    assert scales.free_energy_span == pytest.approx(free_span)
    assert scales.interaction_energy == pytest.approx(interaction_energy)
    assert scales.E0 == pytest.approx(expected_reference)
    assert scales.reference_energy.source == "derived"
    np.testing.assert_allclose(
        h0_prime,
        (h0 - eigenvalues[0] * np.eye(2)) / expected_reference,
        rtol=2e-15,
        atol=0.0,
    )
    assert np.linalg.norm(mu_x_prime, ord=2) == pytest.approx(1.0)
    assert np.max(np.linalg.norm(field_prime, axis=1)) == pytest.approx(1.0)


def test_gapless_driven_system_uses_interaction_energy():
    h0 = np.zeros((2, 2))
    mu_x = np.array([[0.0, 3.0e-30], [3.0e-30, 0.0]])
    field = _constant_field(np.tile([4.0e8, 0.0], (5, 1)))

    *_, scales = nondimensionalize_system(
        h0, mu_x, np.zeros_like(mu_x), field, H0_units="energy", time_units="fs"
    )

    assert scales.free_energy_span == 0.0
    assert scales.E0 == pytest.approx(1.2e-21)
    assert scales.lambda_coupling == pytest.approx(1.0)
    assert scales.physical_coupling_ratio is None


def test_regular_electric_field_that_is_identically_zero_is_rejected():
    field = ElectricField(np.linspace(0.0, 0.2, 5))
    h0 = np.diag([0.0, 2.0e-21])
    mu = np.array([[0.0, 1.0e-30], [1.0e-30, 0.0]])

    with pytest.raises(ValueError, match="ZeroField"):
        nondimensionalize_system(
            h0, mu, np.zeros_like(mu), field, H0_units="energy", time_units="fs"
        )


def test_explicit_zero_field_has_inactive_field_scale_and_zero_coupling():
    field = ZeroField(np.linspace(0.0, 0.2, 5))
    h0 = np.diag([1.0e-21, 3.0e-21])
    mu = np.array([[0.0, 1.0e-30], [1.0e-30, 0.0]])

    _, _, _, field_prime, _, _, scales = nondimensionalize_system(
        h0, mu, np.zeros_like(mu), field, H0_units="energy", time_units="fs"
    )

    assert scales.field_scale.value is None
    assert scales.field_scale.source == "inactive"
    assert scales.lambda_coupling == 0.0
    assert scales.physical_coupling_ratio == 0.0
    np.testing.assert_array_equal(field_prime, np.zeros((5, 2)))


def test_driven_problem_with_zero_coupling_operator_is_rejected():
    field = _constant_field(np.tile([2.0e8, 0.0], (5, 1)))
    h0 = np.diag([0.0, 2.0e-21])
    zero = np.zeros((2, 2))

    with pytest.raises(ValueError, match="coupling dipole"):
        nondimensionalize_system(
            h0, zero, zero, field, H0_units="energy", time_units="fs"
        )


def test_all_zero_generator_requires_high_level_trivial_evolution():
    field = ZeroField(np.linspace(0.0, 0.2, 5))
    zero = np.zeros((2, 2))

    with pytest.raises(ValueError, match="no characteristic energy"):
        nondimensionalize_system(
            zero, zero, zero, field, H0_units="energy", time_units="fs"
        )


class _TwoLevelDipole:
    def __init__(self, matrix_cm: np.ndarray):
        self._matrices = {
            "x": matrix_cm,
            "y": np.zeros_like(matrix_cm),
            "z": np.zeros_like(matrix_cm),
        }

    def get_mu_x_SI(self, dense=True):
        return self._matrices["x"]

    def get_mu_y_SI(self, dense=True):
        return self._matrices["y"]

    def get_mu_z_SI(self, dense=True):
        return self._matrices["z"]

    def get_mu_in_units(self, axis, units):
        assert units == "rad/fs/(V/m)"
        return self._matrices[axis] * 1e-15 / HBAR


def test_centering_restores_absolute_wavefunction_phase():
    tlist = np.linspace(0.0, 2.0, 201)
    field = ElectricField(tlist)
    field.add_arbitrary_Efield(np.tile([1.0e8, 0.0], (tlist.size, 1)))

    offset = 1.0e-19
    h0 = Hamiltonian(np.diag([offset, offset + 1.0e-21]), units="J")
    dipole = _TwoLevelDipole(np.array([[0.0, 1.0e-30], [1.0e-30, 0.0]]))
    initial = np.array([1.0, 0.0], dtype=np.complex128)
    solver = SchrodingerPropagator(validate_units=False)

    time_dimensional, psi_dimensional = solver.propagate(
        h0,
        field,
        dipole,
        initial,
        return_traj=True,
        return_time_psi=True,
        nondimensional=False,
    )
    time_scaled, psi_scaled = solver.propagate(
        h0,
        field,
        dipole,
        initial,
        return_traj=True,
        return_time_psi=True,
        nondimensional=True,
    )

    np.testing.assert_allclose(time_scaled, time_dimensional, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(psi_scaled, psi_dimensional, rtol=2e-7, atol=2e-9)


def test_explicit_reference_energy_records_provenance():
    h0 = np.diag([0.0, 2.0e-21])
    mu = np.array([[0.0, 1.0e-30], [1.0e-30, 0.0]])
    field = _constant_field(np.tile([2.0e8, 0.0], (5, 1)))

    *_, scales = nondimensionalize_system(
        h0,
        mu,
        np.zeros_like(mu),
        field,
        energy_scale_J=7.0e-21,
        H0_units="energy",
        time_units="fs",
    )

    assert scales.E0 == pytest.approx(7.0e-21)
    assert scales.reference_energy.source == "explicit"
    assert scales.reference_energy.method == "caller_supplied"


def test_nonhermitian_coupling_operator_is_rejected():
    h0 = np.diag([0.0, 2.0e-21])
    nonhermitian_mu = np.array([[0.0, 1.0e-30], [0.0, 0.0]])
    field = _constant_field(np.tile([2.0e8, 0.0], (5, 1)))

    with pytest.raises(ValueError, match="coupling dipole 0 must be Hermitian"):
        nondimensionalize_system(
            h0,
            nonhermitian_mu,
            np.zeros_like(nonhermitian_mu),
            field,
            H0_units="energy",
            time_units="fs",
        )


def test_auto_timestep_is_rejected_instead_of_resampling():
    tlist = np.linspace(0.0, 0.2, 5)
    field = ElectricField(tlist)
    field.add_arbitrary_Efield(np.tile([1.0e8, 0.0], (tlist.size, 1)))
    h0 = Hamiltonian(np.diag([0.0, 1.0e-21]), units="J")
    dipole = _TwoLevelDipole(np.array([[0.0, 1.0e-30], [1.0e-30, 0.0]]))

    with pytest.raises(ValueError, match=r"auto_timestep.*removed"):
        SchrodingerPropagator(validate_units=False).propagate(
            h0,
            field,
            dipole,
            np.array([1.0, 0.0], dtype=np.complex128),
            nondimensional=True,
            auto_timestep=True,
        )


def test_schrodinger_rejects_unknown_propagation_option():
    solver = SchrodingerPropagator(validate_units=False)

    with pytest.raises(ValueError, match="unsupported propagation options: typo"):
        solver.propagate(
            None,
            None,
            None,
            np.array([1.0]),
            typo=True,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"coupling_mode": "scalar", "coupling_axis": "x", "axes": "xy"},
            "axes is not applicable",
        ),
        (
            {"coupling_mode": "cartesian", "coupling_axis": "x"},
            "coupling_axis is not applicable",
        ),
        (
            {"algorithm": "rk4", "split_interaction": "helicity_projected"},
            "split_interaction is applicable only",
        ),
    ],
)
def test_schrodinger_rejects_inapplicable_options(kwargs, message):
    solver = SchrodingerPropagator(validate_units=False)

    with pytest.raises(ValueError, match=message):
        solver.propagate(None, None, None, np.array([1.0]), **kwargs)


def test_schrodinger_rejects_noncallable_custom_propagator():
    solver = SchrodingerPropagator(validate_units=False)

    with pytest.raises(TypeError, match="propagator_func must be callable"):
        solver.propagate(None, None, None, np.array([1.0]), propagator_func=0)


def test_schrodinger_constructor_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend must be"):
        SchrodingerPropagator(backend="cuda", validate_units=False)


def test_schrodinger_constructor_rejects_inapplicable_split_interaction():
    with pytest.raises(ValueError, match="requires algorithm='split_operator'"):
        SchrodingerPropagator(
            algorithm="rk4",
            split_interaction="helicity_projected",
            validate_units=False,
        )


def test_schrodinger_rejects_noncallable_constructor_and_setter_hooks():
    with pytest.raises(TypeError, match="custom_propagator must be callable"):
        SchrodingerPropagator(custom_propagator=0, validate_units=False)

    solver = SchrodingerPropagator(validate_units=False)
    with pytest.raises(TypeError, match="propagator_func must be callable"):
        solver.set_custom_propagator(0)
