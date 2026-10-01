"""Contracts for the single authoritative reduced Planck constant."""

from __future__ import annotations

import math

import numpy as np

from rovibrational_excitation.core.operators import Hamiltonian
from rovibrational_excitation.core.units.constants import CONSTANTS
from rovibrational_excitation.core.units.converters import converter
from rovibrational_excitation.dynamics.scaling.utils import _HBAR as SCALING_HBAR
from rovibrational_excitation.dynamics.utils import DIRAC_HBAR


def test_reduced_planck_constant_is_derived_from_exact_planck_constant() -> None:
    assert CONSTANTS.HBAR == CONSTANTS.H / (2.0 * math.pi)


def test_all_runtime_hbar_aliases_share_the_authoritative_value() -> None:
    assert Hamiltonian._HBAR == CONSTANTS.HBAR
    assert SCALING_HBAR == CONSTANTS.HBAR
    assert DIRAC_HBAR == CONSTANTS.HBAR * 1.0e15


def test_hamiltonian_frequency_energy_round_trip_is_exact() -> None:
    matrix_rad_per_fs = np.diag([0.0, 0.37, 2.5])
    frequency_hamiltonian = Hamiltonian(matrix_rad_per_fs, "rad/fs")

    energy_hamiltonian = frequency_hamiltonian.to_energy_units()
    np.testing.assert_array_equal(
        energy_hamiltonian.matrix,
        matrix_rad_per_fs * (CONSTANTS.HBAR * 1.0e15),
    )
    np.testing.assert_array_equal(
        energy_hamiltonian.get_matrix("rad/fs"), matrix_rad_per_fs
    )
    np.testing.assert_array_equal(
        energy_hamiltonian.to_frequency_units().matrix, matrix_rad_per_fs
    )


def test_dipole_frequency_coupling_uses_the_same_hbar() -> None:
    dipole_c_m = np.array([[0.0, 2.0e-29], [2.0e-29, 0.0]])

    coupling = converter.convert_dipole_moment(dipole_c_m, "C*m", "rad/fs/(V/m)")
    np.testing.assert_array_equal(coupling, dipole_c_m / (CONSTANTS.HBAR * 1.0e15))
    np.testing.assert_array_equal(
        converter.convert_dipole_moment(coupling, "rad/fs/(V/m)", "C*m"),
        dipole_c_m,
    )
