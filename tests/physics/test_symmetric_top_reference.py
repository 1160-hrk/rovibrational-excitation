"""Independent references for the rigid parallel-band symmetric-top model."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse
from sympy.physics.wigner import wigner_3j

from rovibrational_excitation.models.parameters import SymmetricTopParameters
from rovibrational_excitation.models.symmetric_top import (
    SymmetricTopBasis,
    SymmetricTopDipoleMatrix,
)
from rovibrational_excitation.models.symmetric_top.rotational import (
    parallel_spherical_direction_cosine,
)
from rovibrational_excitation.models.vibration.morse import tdm_vib_morse

pytestmark = pytest.mark.physics

OMEGA01 = 0.37
ANHARMONIC_SHIFT = 0.015
B_PERPENDICULAR = 0.004
B_PARALLEL = 0.006
ALPHA_PERPENDICULAR = 0.0002
ALPHA_PARALLEL = 0.0003
DIPOLE_C_M = 2.0e-29


def _mapping(*, isomer: str = "ortho", potential: str = "harmonic") -> dict:
    return {
        "basis_type": "symtop",
        "molecule": "methyl fluoride",
        "nuclear_spin_isomer": isomer,
        "V_max": 1,
        "J_max": 2,
        "vibrational_frequency": OMEGA01,
        "vibrational_frequency_units": "rad/fs",
        "anharmonic_shift": ANHARMONIC_SHIFT,
        "anharmonic_shift_units": "rad/fs",
        "rotational_constant_perpendicular": B_PERPENDICULAR,
        "rotational_constant_perpendicular_units": "rad/fs",
        "rotational_constant_parallel": B_PARALLEL,
        "rotational_constant_parallel_units": "rad/fs",
        "vibration_rotation_coupling_perpendicular": ALPHA_PERPENDICULAR,
        "vibration_rotation_coupling_perpendicular_units": "rad/fs",
        "vibration_rotation_coupling_parallel": ALPHA_PARALLEL,
        "vibration_rotation_coupling_parallel_units": "rad/fs",
        "dipole_scale": DIPOLE_C_M,
        "dipole_scale_units": "C*m",
        "potential_type": potential,
    }


def _parameters(**kwargs) -> SymmetricTopParameters:
    return SymmetricTopParameters.from_mapping(_mapping(**kwargs))


def _frequency_value(rad_per_fs: float, unit: str) -> float:
    factors = {
        "rad/fs": 1.0,
        "PHz": 2.0 * np.pi,
        "THz": 2.0 * np.pi * 1.0e-3,
        "cm^-1": 2.0 * np.pi * 2.99792458e8 * 1.0e-13,
    }
    return rad_per_fs / factors[unit]


def _sympy_spherical_reference(
    j_bra: int,
    k_bra: int,
    m_bra: int,
    j_ket: int,
    k_ket: int,
    m_ket: int,
    p: int,
) -> float:
    if k_bra != k_ket:
        return 0.0
    phase = -1.0 if (m_bra - k_ket) % 2 else 1.0
    return float(
        phase
        * np.sqrt((2 * j_bra + 1) * (2 * j_ket + 1))
        * wigner_3j(j_bra, 1, j_ket, -m_bra, p, m_ket)
        * wigner_3j(j_bra, 1, j_ket, -k_bra, 0, k_ket)
    )


def test_rank_one_rotational_primitive_matches_independent_sympy_reference() -> None:
    for j_bra in range(4):
        for j_ket in range(4):
            for k_bra in range(-j_bra, j_bra + 1):
                for k_ket in range(-j_ket, j_ket + 1):
                    for m_bra in range(-j_bra, j_bra + 1):
                        for m_ket in range(-j_ket, j_ket + 1):
                            for p in (-1, 0, 1):
                                expected = _sympy_spherical_reference(
                                    j_bra,
                                    k_bra,
                                    m_bra,
                                    j_ket,
                                    k_ket,
                                    m_ket,
                                    p,
                                )
                                actual = parallel_spherical_direction_cosine(
                                    j_bra,
                                    k_bra,
                                    m_bra,
                                    j_ket,
                                    k_ket,
                                    m_ket,
                                    p,
                                )
                                assert actual == pytest.approx(
                                    expected,
                                    rel=2.0e-14,
                                    abs=2.0e-15,
                                )


def test_basis_order_and_spin_isomer_filter_are_explicit() -> None:
    ortho = SymmetricTopBasis(_parameters(isomer="ortho"))
    para = SymmetricTopBasis(_parameters(isomer="para"))

    assert ortho.quantum_number_order == ("v", "J", "K", "M")
    np.testing.assert_array_equal(
        ortho.basis[:4],
        np.array(
            [
                [0, 0, 0, 0],
                [0, 1, 0, -1],
                [0, 1, 0, 0],
                [0, 1, 0, 1],
            ]
        ),
    )
    assert ortho.size() == 18
    assert para.size() == 52

    for basis, expected_sector in ((ortho, "ortho"), (para, "para")):
        for index in range(basis.size()):
            state = tuple(int(value) for value in basis.get_state(index))
            assert basis.get_index(state) == index
            assignment = basis.parameters.molecule_preset.assign(
                basis.symmetry_state(index),
                nuclear_spin_isomer=expected_sector,
            )
            assert assignment.allowed
            assert assignment.sector == expected_sector


def test_hamiltonian_uses_omega01_and_two_vibration_rotation_couplings() -> None:
    basis = SymmetricTopBasis(_parameters(isomer="para"))
    energies = np.diag(basis.generate_H0().to_frequency_units().matrix)

    for index, (v, j, k, _m) in enumerate(basis.basis):
        x = v + 0.5
        vibrational = (OMEGA01 + ANHARMONIC_SHIFT) * x - 0.5 * ANHARMONIC_SHIFT * x**2
        b_perpendicular_v = B_PERPENDICULAR - ALPHA_PERPENDICULAR * x
        b_parallel_v = B_PARALLEL - ALPHA_PARALLEL * x
        expected = (
            vibrational
            + b_perpendicular_v * j * (j + 1)
            + (b_parallel_v - b_perpendicular_v) * k**2
        )
        assert energies[index] == pytest.approx(expected, abs=2.0e-15)

    for v in range(2):
        for j in range(1, 3):
            for k in range(1, j + 1):
                plus = basis.get_index((v, j, k, -j))
                minus = basis.get_index((v, j, -k, -j))
                assert energies[plus] == energies[minus]


@pytest.mark.parametrize("unit", ["PHz", "THz", "cm^-1"])
def test_all_symtop_frequency_inputs_convert_once_to_the_same_hamiltonian(
    unit: str,
) -> None:
    reference = SymmetricTopBasis(_parameters(isomer="para"))
    mapping = _mapping(isomer="para")
    values = {
        "vibrational_frequency": OMEGA01,
        "anharmonic_shift": ANHARMONIC_SHIFT,
        "rotational_constant_perpendicular": B_PERPENDICULAR,
        "rotational_constant_parallel": B_PARALLEL,
        "vibration_rotation_coupling_perpendicular": ALPHA_PERPENDICULAR,
        "vibration_rotation_coupling_parallel": ALPHA_PARALLEL,
    }
    for key, canonical_value in values.items():
        mapping[key] = _frequency_value(canonical_value, unit)
        mapping[f"{key}_units"] = unit
    actual = SymmetricTopBasis(SymmetricTopParameters.from_mapping(mapping))

    np.testing.assert_array_equal(actual.basis, reference.basis)
    np.testing.assert_allclose(
        actual.generate_H0().matrix,
        reference.generate_H0().matrix,
        rtol=4.0e-15,
        atol=0.0,
    )


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_parallel_band_cartesian_dipoles_follow_reference_and_are_hermitian(
    axis: str,
) -> None:
    basis = SymmetricTopBasis(_parameters(isomer="para"))
    dipole = SymmetricTopDipoleMatrix(
        basis,
        mu0=DIPOLE_C_M,
        potential_type="harmonic",
        backend="numpy",
        dense=True,
    )
    matrix = dipole.mu(axis)
    np.testing.assert_allclose(matrix, matrix.conj().T, rtol=0.0, atol=2.0e-44)

    rows, columns = np.nonzero(np.abs(matrix) > 0.0)
    assert rows.size > 0
    for row, column in zip(rows, columns):
        v_bra, j_bra, k_bra, m_bra = map(int, basis.get_state(row))
        v_ket, j_ket, k_ket, m_ket = map(int, basis.get_state(column))
        assert abs(v_bra - v_ket) == 1
        assert abs(j_bra - j_ket) <= 1
        assert k_bra == k_ket
        if axis == "z":
            assert m_bra == m_ket
        else:
            assert abs(m_bra - m_ket) == 1


def test_dense_and_csr_dipoles_are_exactly_equal() -> None:
    basis = SymmetricTopBasis(_parameters(isomer="para"))
    dense = SymmetricTopDipoleMatrix(
        basis,
        mu0=DIPOLE_C_M,
        potential_type="harmonic",
        backend="numpy",
        dense=True,
    )
    csr = SymmetricTopDipoleMatrix(
        basis,
        mu0=DIPOLE_C_M,
        potential_type="harmonic",
        backend="numpy",
        dense=False,
    )

    for axis in "xyz":
        sparse_matrix = csr.mu(axis)
        assert sparse.isspmatrix_csr(sparse_matrix)
        np.testing.assert_array_equal(sparse_matrix.toarray(), dense.mu(axis))


def test_morse_overtone_factors_match_the_accepted_vibrational_reference() -> None:
    mapping = _mapping(isomer="para", potential="morse")
    mapping["V_max"] = 3
    parameters = SymmetricTopParameters.from_mapping(mapping)
    basis = SymmetricTopBasis(parameters)
    dipole = SymmetricTopDipoleMatrix(
        basis,
        mu0=DIPOLE_C_M,
        potential_type="morse",
        backend="numpy",
        dense=True,
    )
    matrix = dipole.mu("z")
    level_parameter = (OMEGA01 + ANHARMONIC_SHIFT) / ANHARMONIC_SHIFT - 0.5

    for upper_v in (1, 2, 3):
        row = basis.get_index((upper_v, 1, 1, 1))
        column = basis.get_index((0, 1, 1, 1))
        rotational = parallel_spherical_direction_cosine(1, 1, 1, 1, 1, 1, 0)
        actual = matrix[row, column] / (DIPOLE_C_M * rotational)
        expected = tdm_vib_morse(upper_v, 0, level_parameter)
        assert actual == pytest.approx(expected, rel=3.0e-15, abs=0.0)
