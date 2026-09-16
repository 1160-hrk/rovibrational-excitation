"""Frozen behavior for the Phase 6 LinMol ownership migration."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import scipy.sparse as sp

from rovibrational_excitation.core.execution import ExecutionPolicy
from rovibrational_excitation.core.units.converters import converter
from rovibrational_excitation.models.factory import build_model_from_parameters
from rovibrational_excitation.models.linear_molecule import (
    LinMolBasis,
    LinMolDipoleMatrix,
    build_linmol,
    build_linmol_from_parameters,
    build_linmol_operators_from_parameters,
    build_mu,
)
from rovibrational_excitation.models.parameters import LinMolParameters
from rovibrational_excitation.models.validation import LinMolRepresentation


def _mapping() -> dict[str, object]:
    return {
        "basis_type": "linmol",
        "V_max": 1,
        "J_max": 2,
        "representation": "m_resolved",
        "vibrational_frequency": 0.37,
        "vibrational_frequency_units": "rad/fs",
        "anharmonic_shift": 0.015,
        "anharmonic_shift_units": "rad/fs",
        "rotational_constant": 0.004,
        "rotational_constant_units": "rad/fs",
        "vibration_rotation_coupling": 0.0002,
        "vibration_rotation_coupling_units": "rad/fs",
        "dipole_scale": 0.3,
        "dipole_scale_units": "D",
        "potential_type": "harmonic",
        "initial_states": [1, 3],
    }


def _parameters() -> LinMolParameters:
    return LinMolParameters.from_mapping(_mapping())


def _policy(storage: str) -> ExecutionPolicy:
    return ExecutionPolicy.from_strings(backend="numpy", storage=storage)


def _expected_basis() -> np.ndarray:
    return np.asarray(
        [(v, j, m) for v in range(2) for j in range(3) for m in range(-j, j + 1)]
    )


def _expected_energies() -> np.ndarray:
    basis = _expected_basis()
    vterm = basis[:, 0] + 0.5
    jterm = basis[:, 1] * (basis[:, 1] + 1)
    return (
        (0.37 + 0.015) * vterm
        - (0.015 / 2.0) * vterm**2
        + (0.004 - 0.0002 * vterm) * jterm
    )


def test_current_linmol_owners_and_transitional_builders_are_explicit() -> None:
    assert LinMolBasis.__module__ == (
        "rovibrational_excitation.models.linear_molecule.basis"
    )
    assert LinMolDipoleMatrix.__module__ == (
        "rovibrational_excitation.models.linear_molecule.dipole"
    )
    assert build_mu.__module__ == (
        "rovibrational_excitation.models.linear_molecule.dipole_builder"
    )
    assert build_linmol.__module__ == (
        "rovibrational_excitation.models.linear_molecule.model"
    )
    assert build_linmol_operators_from_parameters.__module__ == (
        "rovibrational_excitation.models.linear_molecule.model"
    )
    assert LinMolParameters.__module__ == ("rovibrational_excitation.models.parameters")


def test_parameters_preserve_input_quantities_and_freeze_canonical_values() -> None:
    parameters = LinMolParameters.from_mapping(
        {
            **_mapping(),
            "vibrational_frequency": 2300.0,
            "vibrational_frequency_units": "cm^-1",
            "anharmonic_shift": 15.0,
            "anharmonic_shift_units": "cm^-1",
            "rotational_constant": 0.9,
            "rotational_constant_units": "cm^-1",
            "vibration_rotation_coupling": 0.02,
            "vibration_rotation_coupling_units": "cm^-1",
            "potential_type": "morse",
        }
    )

    assert parameters.vibrational_frequency.value == 2300.0
    assert parameters.vibrational_frequency.unit == "cm^-1"
    assert parameters.vibrational_frequency.angular_rad_per_fs == (
        converter.convert_frequency(2300.0, "cm^-1", "rad/fs")
    )
    assert parameters.rotational_constant.angular_rad_per_fs == (
        converter.convert_frequency(0.9, "cm^-1", "rad/fs")
    )
    assert parameters.dipole_c_m == converter.convert_dipole_moment(0.3, "D", "C*m")
    assert parameters.potential_type == "morse"
    with pytest.raises(FrozenInstanceError):
        parameters.j_max = 3  # type: ignore[misc]


def test_basis_order_mapping_and_hamiltonian_are_exact() -> None:
    basis, hamiltonian, _dipole = build_linmol_operators_from_parameters(
        _parameters(), execution_policy=_policy("dense")
    )

    expected_basis = _expected_basis()
    np.testing.assert_array_equal(basis.basis, expected_basis)
    assert basis.index_map == {
        tuple(state): index for index, state in enumerate(expected_basis)
    }
    for index, state in enumerate(expected_basis):
        assert basis.get_index(tuple(state)) == index
        np.testing.assert_array_equal(basis.get_state(index), state)
    assert hamiltonian.units == "J"
    np.testing.assert_allclose(
        hamiltonian.get_matrix("rad/fs"),
        np.diag(_expected_energies()),
        rtol=0.0,
        atol=1.2e-16,
    )


def test_model_builder_owns_cartesian_coupling_and_coherent_state_order() -> None:
    model = build_model_from_parameters(
        _parameters(),
        initial_states=[1, 3],
        representation=LinMolRepresentation.M_RESOLVED,
        axes="xz",
        execution_policy=_policy("dense"),
    )

    assert model.name == "linmol"
    assert model.coupling.mode.value == "cartesian"
    assert model.coupling.axes == ("x", "z")
    assert model.metadata == {}
    expected_state = np.zeros((_expected_basis().shape[0], 1), dtype=np.complex128)
    expected_state[[1, 3], 0] = 1.0 / np.sqrt(2.0)
    np.testing.assert_array_equal(model.state.data, expected_state)


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_stateless_and_stateful_dipole_paths_are_exactly_equal(axis: str) -> None:
    basis, _hamiltonian, stateful = build_linmol_operators_from_parameters(
        _parameters(), execution_policy=_policy("csr")
    )
    one_shot = build_mu(
        basis,
        axis,
        _parameters().dipole_c_m,
        potential_type="harmonic",
        dense=False,
    )

    assert sp.isspmatrix_csr(one_shot)
    np.testing.assert_array_equal(one_shot.toarray(), stateful.mu(axis).toarray())
    assert stateful.mu(axis) is stateful.mu(axis)


def test_dense_and_csr_model_operators_are_exactly_equal() -> None:
    dense_basis, dense_hamiltonian, dense_dipole = (
        build_linmol_operators_from_parameters(
            _parameters(), execution_policy=_policy("dense")
        )
    )
    csr_basis, csr_hamiltonian, csr_dipole = build_linmol_operators_from_parameters(
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


def test_mapping_and_frozen_parameter_builders_are_exactly_equivalent() -> None:
    mapping_result = build_linmol(_mapping(), execution_policy=_policy("dense"))
    typed_result = build_linmol_from_parameters(
        _parameters(), _mapping()["initial_states"], execution_policy=_policy("dense")
    )

    mapping_basis, mapping_state, mapping_hamiltonian, mapping_dipole = mapping_result
    typed_basis, typed_state, typed_hamiltonian, typed_dipole = typed_result
    np.testing.assert_array_equal(mapping_basis.basis, typed_basis.basis)
    np.testing.assert_array_equal(mapping_state.data, typed_state.data)
    np.testing.assert_array_equal(mapping_hamiltonian.matrix, typed_hamiltonian.matrix)
    for axis in "xyz":
        np.testing.assert_array_equal(mapping_dipole.mu(axis), typed_dipole.mu(axis))
