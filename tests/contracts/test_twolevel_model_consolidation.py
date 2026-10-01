"""Frozen behavior for the Phase 6 TwoLevel ownership migration."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import scipy.sparse as sp

from rovibrational_excitation.core.execution import ExecutionPolicy
from rovibrational_excitation.core.units.converters import converter
from rovibrational_excitation.models.factory import build_model_from_parameters
from rovibrational_excitation.models.two_level import (
    TwoLevelBasis,
    TwoLevelDipoleMatrix,
    TwoLevelParameters,
    build_twolevel_operators_from_parameters,
)


def _parameters(*, gap: float = 0.37, gap_units: str = "rad/fs") -> TwoLevelParameters:
    return TwoLevelParameters.from_mapping(
        {
            "energy_gap": gap,
            "energy_gap_units": gap_units,
            "dipole_scale": 0.3,
            "dipole_scale_units": "D",
        }
    )


def _policy(storage: str) -> ExecutionPolicy:
    return ExecutionPolicy.from_strings(backend="numpy", storage=storage)


def _unified_builder_gaps(gap_rad_per_fs: float) -> tuple[float, float]:
    """Return the approved J storage and propagation-boundary gap."""
    assert gap_rad_per_fs == 0.37
    return (
        float.fromhex("0x1.70868888018b1p-65"),
        float.fromhex("0x1.7ae147ae147aep-2"),
    )


def test_twolevel_types_and_builders_have_one_model_owned_home() -> None:
    assert TwoLevelBasis.__module__ == (
        "rovibrational_excitation.models.two_level.basis"
    )
    assert TwoLevelDipoleMatrix.__module__ == (
        "rovibrational_excitation.models.two_level.dipole"
    )
    assert build_twolevel_operators_from_parameters.__module__ == (
        "rovibrational_excitation.models.two_level.model"
    )
    assert TwoLevelParameters.__module__ == (
        "rovibrational_excitation.models.two_level.parameters"
    )


def test_parameters_preserve_gap_input_and_freeze_converted_dipole() -> None:
    parameters = _parameters(gap=2300.0, gap_units="cm^-1")

    assert parameters.energy_gap == 2300.0
    assert parameters.energy_gap_units == "cm^-1"
    assert parameters.dipole_c_m == converter.convert_dipole_moment(0.3, "D", "C*m")
    with pytest.raises(FrozenInstanceError):
        parameters.energy_gap = 1.0  # type: ignore[misc]


def test_basis_order_mapping_and_hamiltonian_formula_are_exact() -> None:
    basis = TwoLevelBasis(
        energy_gap=0.37,
        input_units="rad/fs",
        output_units="rad/fs",
    )

    np.testing.assert_array_equal(basis.basis, np.array([[0], [1]]))
    assert basis.index_map == {(0,): 0, (1,): 1}
    assert [basis.get_index(state) for state in (0, np.int64(1), [0], (1,))] == [
        0,
        1,
        0,
        1,
    ]
    np.testing.assert_array_equal(basis.get_state(0), np.array([0]))
    np.testing.assert_array_equal(basis.get_state(1), np.array([1]))

    hamiltonian = basis.generate_H0()
    assert hamiltonian.units == "rad/fs"
    np.testing.assert_array_equal(hamiltonian.matrix, np.diag([0.0, 0.37]))


def test_model_builder_owns_scalar_x_coupling_and_coherent_state_order() -> None:
    model = build_model_from_parameters(
        _parameters(),
        initial_states=[0, 1],
        representation=None,
        axes=None,
        execution_policy=_policy("dense"),
    )

    assert model.name == "twolevel"
    assert model.coupling.mode.value == "scalar"
    assert model.coupling.axes == ("x",)
    assert model.metadata == {}
    np.testing.assert_array_equal(
        model.state.data,
        np.full((2, 1), 1.0 / np.sqrt(2.0), dtype=np.complex128),
    )
    gap_j, propagation_gap = _unified_builder_gaps(0.37)
    np.testing.assert_array_equal(model.hamiltonian.matrix, np.diag([0.0, gap_j]))
    np.testing.assert_array_equal(
        model.hamiltonian.get_matrix("rad/fs"), np.diag([0.0, propagation_gap])
    )
    assert propagation_gap == 0.37


def test_dense_operators_have_exact_two_level_matrices() -> None:
    basis, hamiltonian, dipole = build_twolevel_operators_from_parameters(
        _parameters(), execution_policy=_policy("dense")
    )
    dipole_c_m = _parameters().dipole_c_m

    assert basis.size() == 2
    assert hamiltonian.units == "J"
    gap_j, propagation_gap = _unified_builder_gaps(0.37)
    np.testing.assert_array_equal(hamiltonian.matrix, np.diag([0.0, gap_j]))
    np.testing.assert_array_equal(
        hamiltonian.get_matrix("rad/fs"), np.diag([0.0, propagation_gap])
    )
    np.testing.assert_array_equal(
        dipole.mu("x"),
        dipole_c_m * np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
    )
    np.testing.assert_array_equal(
        dipole.mu("y"),
        dipole_c_m * np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
    )
    np.testing.assert_array_equal(dipole.mu("z"), np.zeros((2, 2), dtype=np.complex128))
    assert dipole.mu("x") is dipole.mu("x")


def test_csr_operators_equal_dense_without_changing_hamiltonian_storage() -> None:
    dense_basis, dense_hamiltonian, dense_dipole = (
        build_twolevel_operators_from_parameters(
            _parameters(), execution_policy=_policy("dense")
        )
    )
    csr_basis, csr_hamiltonian, csr_dipole = build_twolevel_operators_from_parameters(
        _parameters(), execution_policy=_policy("csr")
    )

    assert dense_basis.basis.tolist() == csr_basis.basis.tolist()
    assert isinstance(csr_hamiltonian.matrix, np.ndarray)
    assert sp.isspmatrix_csr(csr_dipole.mu("x"))
    assert sp.isspmatrix_csr(csr_dipole.mu("y"))
    assert sp.isspmatrix_csr(csr_dipole.mu("z"))
    np.testing.assert_array_equal(csr_hamiltonian.matrix, dense_hamiltonian.matrix)
    for axis in "xyz":
        np.testing.assert_array_equal(
            csr_dipole.mu(axis).toarray(), dense_dipole.mu(axis)
        )
