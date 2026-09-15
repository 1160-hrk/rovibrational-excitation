"""Frozen behavior for the Phase 6 VibLadder ownership migration."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import scipy.sparse as sp

from rovibrational_excitation.core.execution import ExecutionPolicy
from rovibrational_excitation.core.units.converters import converter
from rovibrational_excitation.models.factory import build_model_from_parameters
from rovibrational_excitation.models.parameters import VibLadderParameters
from rovibrational_excitation.models.vib_ladder import (
    VibLadderBasis,
    VibLadderDipoleMatrix,
    build_mu,
    build_vibladder,
    build_vibladder_from_parameters,
    build_vibladder_operators_from_parameters,
)


def _parameters(
    *,
    potential_type: str = "harmonic",
    v_max: int = 3,
) -> VibLadderParameters:
    return VibLadderParameters.from_mapping(
        {
            "V_max": v_max,
            "vibrational_frequency": 0.37,
            "vibrational_frequency_units": "rad/fs",
            "anharmonic_shift": 0.015,
            "anharmonic_shift_units": "rad/fs",
            "dipole_scale": 0.3,
            "dipole_scale_units": "D",
            "potential_type": potential_type,
        }
    )


def _mapping() -> dict[str, object]:
    return {
        "basis_type": "vibladder",
        "V_max": 3,
        "vibrational_frequency": 0.37,
        "vibrational_frequency_units": "rad/fs",
        "anharmonic_shift": 0.015,
        "anharmonic_shift_units": "rad/fs",
        "dipole_scale": 0.3,
        "dipole_scale_units": "D",
        "potential_type": "harmonic",
        "initial_states": [0, 2],
    }


def _policy(storage: str) -> ExecutionPolicy:
    return ExecutionPolicy.from_strings(backend="numpy", storage=storage)


def _expected_energies() -> np.ndarray:
    levels = np.arange(4, dtype=np.float64)
    vterm = levels + 0.5
    return (0.37 + 0.015) * vterm - (0.015 / 2.0) * vterm**2


def test_vibladder_types_and_builders_have_one_model_owned_home() -> None:
    assert VibLadderBasis.__module__ == (
        "rovibrational_excitation.models.vib_ladder.basis"
    )
    assert VibLadderDipoleMatrix.__module__ == (
        "rovibrational_excitation.models.vib_ladder.dipole"
    )
    assert build_mu.__module__ == (
        "rovibrational_excitation.models.vib_ladder.dipole_builder"
    )
    assert build_vibladder.__module__ == (
        "rovibrational_excitation.models.vib_ladder.model"
    )
    assert build_vibladder_operators_from_parameters.__module__ == (
        "rovibrational_excitation.models.vib_ladder.model"
    )
    assert VibLadderParameters.__module__ == (
        "rovibrational_excitation.models.parameters"
    )


def test_parameters_preserve_input_quantities_and_freeze_canonical_values() -> None:
    parameters = VibLadderParameters.from_mapping(
        {
            "V_max": 3,
            "vibrational_frequency": 2300.0,
            "vibrational_frequency_units": "cm^-1",
            "anharmonic_shift": 15.0,
            "anharmonic_shift_units": "cm^-1",
            "dipole_scale": 0.3,
            "dipole_scale_units": "D",
            "potential_type": "morse",
        }
    )

    assert parameters.v_max == 3
    assert parameters.vibrational_frequency.value == 2300.0
    assert parameters.vibrational_frequency.unit == "cm^-1"
    assert parameters.vibrational_frequency.angular_rad_per_fs == (
        converter.convert_frequency(2300.0, "cm^-1", "rad/fs")
    )
    assert parameters.anharmonic_shift.value == 15.0
    assert parameters.anharmonic_shift.unit == "cm^-1"
    assert parameters.anharmonic_shift.angular_rad_per_fs == (
        converter.convert_frequency(15.0, "cm^-1", "rad/fs")
    )
    assert parameters.dipole_c_m == converter.convert_dipole_moment(0.3, "D", "C*m")
    assert parameters.potential_type == "morse"
    with pytest.raises(FrozenInstanceError):
        parameters.v_max = 4  # type: ignore[misc]


def test_basis_order_mapping_and_anharmonic_hamiltonian_are_exact() -> None:
    basis = VibLadderBasis(
        V_max=3,
        omega=0.37,
        delta_omega=0.015,
        input_units="rad/fs",
        output_units="rad/fs",
    )

    np.testing.assert_array_equal(basis.basis, np.arange(4).reshape(-1, 1))
    assert basis.index_map == {(0,): 0, (1,): 1, (2,): 2, (3,): 3}
    assert [basis.get_index(state) for state in (0, np.int64(1), [2], (3,))] == [
        0,
        1,
        2,
        3,
    ]
    for index in range(4):
        np.testing.assert_array_equal(basis.get_state(index), np.array([index]))

    hamiltonian = basis.generate_H0()
    assert hamiltonian.units == "rad/fs"
    np.testing.assert_array_equal(hamiltonian.matrix, np.diag(_expected_energies()))


def test_model_builder_owns_scalar_z_coupling_and_coherent_state_order() -> None:
    model = build_model_from_parameters(
        _parameters(),
        initial_states=[0, 2],
        representation=None,
        axes=None,
        execution_policy=_policy("dense"),
    )

    assert model.name == "vibladder"
    assert model.coupling.mode.value == "scalar"
    assert model.coupling.axes == ("z",)
    assert model.metadata == {}
    expected_state = np.zeros((4, 1), dtype=np.complex128)
    expected_state[[0, 2], 0] = 1.0 / np.sqrt(2.0)
    np.testing.assert_array_equal(model.state.data, expected_state)
    assert model.hamiltonian.units == "J"
    np.testing.assert_array_equal(
        model.hamiltonian.get_matrix("rad/fs"),
        np.diag(_expected_energies()),
    )


def test_dense_harmonic_dipole_has_only_exact_scalar_z_elements() -> None:
    basis, hamiltonian, dipole = build_vibladder_operators_from_parameters(
        _parameters(), execution_policy=_policy("dense")
    )
    dipole_c_m = _parameters().dipole_c_m
    expected_z = np.zeros((4, 4), dtype=np.complex128)
    for lower in range(3):
        expected_z[lower, lower + 1] = dipole_c_m * np.sqrt(lower + 1)
        expected_z[lower + 1, lower] = dipole_c_m * np.sqrt(lower + 1)

    assert basis.size() == 4
    assert hamiltonian.units == "J"
    np.testing.assert_array_equal(
        hamiltonian.get_matrix("rad/fs"), np.diag(_expected_energies())
    )
    np.testing.assert_array_equal(dipole.mu("x"), np.zeros((4, 4)))
    np.testing.assert_array_equal(dipole.mu("y"), np.zeros((4, 4)))
    np.testing.assert_array_equal(dipole.mu("z"), expected_z)
    assert dipole.mu("z") is dipole.mu("z")


def test_csr_operators_equal_dense_without_changing_hamiltonian_storage() -> None:
    dense_basis, dense_hamiltonian, dense_dipole = (
        build_vibladder_operators_from_parameters(
            _parameters(), execution_policy=_policy("dense")
        )
    )
    csr_basis, csr_hamiltonian, csr_dipole = build_vibladder_operators_from_parameters(
        _parameters(), execution_policy=_policy("csr")
    )

    np.testing.assert_array_equal(csr_basis.basis, dense_basis.basis)
    assert isinstance(csr_hamiltonian.matrix, np.ndarray)
    np.testing.assert_array_equal(csr_hamiltonian.matrix, dense_hamiltonian.matrix)
    for axis in "xyz":
        assert sp.isspmatrix_csr(csr_dipole.mu(axis))
        np.testing.assert_array_equal(
            csr_dipole.mu(axis).toarray(), dense_dipole.mu(axis)
        )


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_transitional_dipole_builders_match_stateful_class(axis: str) -> None:
    parameters = _parameters()
    basis = VibLadderBasis(
        V_max=parameters.v_max,
        omega=parameters.vibrational_frequency.angular_rad_per_fs,
        delta_omega=parameters.anharmonic_shift.angular_rad_per_fs,
    )
    stateful = VibLadderDipoleMatrix(
        basis,
        mu0=parameters.dipole_c_m,
        potential_type=parameters.potential_type,
        dense=False,
    )
    one_shot = build_mu(
        basis,
        axis,
        parameters.dipole_c_m,
        potential_type=parameters.potential_type,
        dense=False,
    )

    np.testing.assert_array_equal(one_shot.toarray(), stateful.mu(axis).toarray())


def test_mapping_and_frozen_parameter_builders_are_exactly_equivalent() -> None:
    policy = _policy("dense")
    mapping_result = build_vibladder(_mapping(), execution_policy=policy)
    typed_result = build_vibladder_from_parameters(
        _parameters(), [0, 2], execution_policy=policy
    )

    mapping_basis, mapping_state, mapping_hamiltonian, mapping_dipole = mapping_result
    typed_basis, typed_state, typed_hamiltonian, typed_dipole = typed_result
    np.testing.assert_array_equal(mapping_basis.basis, typed_basis.basis)
    np.testing.assert_array_equal(mapping_state.data, typed_state.data)
    np.testing.assert_array_equal(mapping_hamiltonian.matrix, typed_hamiltonian.matrix)
    for axis in "xyz":
        np.testing.assert_array_equal(mapping_dipole.mu(axis), typed_dipole.mu(axis))
