"""Audit guards before removing the experimental legacy SymTop implementation."""

from __future__ import annotations

import numpy as np
import pytest

from rovibrational_excitation.core.basis.symtop import SymTopBasis
from rovibrational_excitation.dipole.rot.jmk import (
    tdm_jmk_x,
    tdm_jmk_y,
    tdm_jmk_z,
)
from rovibrational_excitation.dipole.symtop import SymTopDipoleMatrix
from rovibrational_excitation.models.parameters import SymmetricTopParameters
from rovibrational_excitation.models.symmetric_top import (
    SymmetricTopBasis,
    SymmetricTopDipoleMatrix,
)
from rovibrational_excitation.models.symmetric_top.rotational import (
    parallel_cartesian_direction_cosine,
)


def _mapping() -> dict[str, object]:
    return {
        "molecule": "CH3F",
        "nuclear_spin_isomer": "ortho",
        "V_max": 1,
        "J_max": 1,
        "vibrational_frequency": 0.37,
        "vibrational_frequency_units": "rad/fs",
        "anharmonic_shift": 0.015,
        "anharmonic_shift_units": "rad/fs",
        "rotational_constant_perpendicular": 0.004,
        "rotational_constant_perpendicular_units": "rad/fs",
        "rotational_constant_parallel": 0.006,
        "rotational_constant_parallel_units": "rad/fs",
        "vibration_rotation_coupling_perpendicular": 0.0002,
        "vibration_rotation_coupling_perpendicular_units": "rad/fs",
        "vibration_rotation_coupling_parallel": 0.0002,
        "vibration_rotation_coupling_parallel_units": "rad/fs",
        "dipole_scale": 1.0,
        "dipole_scale_units": "C*m",
        "potential_type": "harmonic",
    }


def _legacy_basis() -> SymTopBasis:
    return SymTopBasis(
        1,
        1,
        omega=0.37,
        B=0.004,
        C=0.006,
        alpha=0.0002,
        delta_omega=0.015,
        output_units="rad/fs",
    )


def test_legacy_and_production_symtop_owners_are_explicitly_distinct() -> None:
    assert SymTopBasis.__module__ == "rovibrational_excitation.core.basis.symtop"
    assert SymTopDipoleMatrix.__module__ == (
        "rovibrational_excitation.dipole.symtop.cache"
    )
    assert SymmetricTopBasis.__module__ == (
        "rovibrational_excitation.models.symmetric_top.basis"
    )
    assert SymmetricTopDipoleMatrix.__module__ == (
        "rovibrational_excitation.models.symmetric_top.dipole"
    )


def test_legacy_basis_and_energy_convention_are_not_production_aliases() -> None:
    legacy = _legacy_basis()
    production = SymmetricTopBasis(SymmetricTopParameters.from_mapping(_mapping()))

    assert legacy.size() == 20
    assert production.size() == 8
    np.testing.assert_array_equal(legacy.basis[1], [0, 1, -1, -1])
    np.testing.assert_array_equal(production.basis[1], [0, 1, 0, -1])
    assert production.quantum_number_order == ("v", "J", "K", "M")

    legacy_ground = legacy.generate_H0().matrix[0, 0]
    production_ground = production.generate_H0().to_frequency_units().matrix[0, 0]
    assert legacy_ground == pytest.approx(0.18125, abs=1.0e-15)
    assert production_ground == pytest.approx(0.190625, abs=1.0e-15)


def test_legacy_transverse_rotational_phases_differ_from_production() -> None:
    quantum_numbers = (0, 0, 0, 1, -1, 0)
    legacy_x = tdm_jmk_x(*quantum_numbers)
    legacy_y = tdm_jmk_y(*quantum_numbers)
    legacy_z = tdm_jmk_z(0, 0, 0, 1, 0, 0)

    production_x = parallel_cartesian_direction_cosine("x", 0, 0, 0, 1, 0, -1)
    production_y = parallel_cartesian_direction_cosine("y", 0, 0, 0, 1, 0, -1)
    production_z = parallel_cartesian_direction_cosine("z", 0, 0, 0, 1, 0, 0)

    assert legacy_x == pytest.approx(-production_x, abs=1.0e-15)
    assert legacy_y == pytest.approx(-production_y, abs=1.0e-15)
    assert legacy_z == pytest.approx(production_z, abs=1.0e-15)
