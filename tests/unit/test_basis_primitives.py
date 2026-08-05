"""Contracts migrated from the former root-level basis validation script."""

import numpy as np
import pytest

from rovibrational_excitation.core.basis.base import BasisBase
from rovibrational_excitation.core.basis.hamiltonian import Hamiltonian


def test_basis_base_is_abstract_with_expected_interface():
    with pytest.raises(TypeError):
        BasisBase()

    assert BasisBase.__abstractmethods__ == {
        "size",
        "get_index",
        "get_state",
        "generate_H0",
    }


def test_hamiltonian_unit_round_trip_and_copy_isolation():
    matrix_joule = np.array(
        [[0.0, 1.25e-21], [1.25e-21, 3.5e-21]],
        dtype=np.float64,
    )
    hamiltonian = Hamiltonian(matrix_joule, "J")

    round_trip = hamiltonian.to_frequency_units().to_energy_units()
    np.testing.assert_allclose(round_trip.matrix, matrix_joule, rtol=2e-15, atol=0.0)

    extracted = hamiltonian.matrix
    extracted[0, 0] = 99.0
    assert hamiltonian.matrix[0, 0] == 0.0


def test_hamiltonian_spectral_differences():
    hamiltonian = Hamiltonian(np.diag([0.0, 2.0, 5.0]), "rad/fs")

    np.testing.assert_array_equal(
        hamiltonian.energy_differences(),
        np.array([2.0, 5.0, 3.0]),
    )
    assert hamiltonian.max_energy_difference() == 5.0


@pytest.mark.parametrize(
    ("matrix", "error_type", "message"),
    [
        ([0.0, 1.0], TypeError, "numpy ndarray"),
        (np.array([0.0, 1.0]), ValueError, "square 2D"),
        (np.zeros((2, 3)), ValueError, "square 2D"),
    ],
)
def test_hamiltonian_rejects_invalid_matrix(matrix, error_type, message):
    with pytest.raises(error_type, match=message):
        Hamiltonian(matrix)


def test_hamiltonian_rejects_unknown_units():
    with pytest.raises(ValueError, match="units must be"):
        Hamiltonian(np.eye(2), "cm^-1")


def test_hamiltonian_rejects_unknown_requested_units():
    hamiltonian = Hamiltonian(np.eye(2), "J")

    with pytest.raises(ValueError, match="Unknown units"):
        hamiltonian.get_eigenvalues("eV")
