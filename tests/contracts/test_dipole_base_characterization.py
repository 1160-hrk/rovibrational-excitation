"""Freeze the shared dipole mixin before its Phase 6 ownership move."""

import numpy as np
import pytest

from rovibrational_excitation.core.units.converters import converter
from rovibrational_excitation.dynamics.utils import get_dipole_component_SI
from rovibrational_excitation.models.dipole_base import DipoleMatrixBase
from rovibrational_excitation.models.two_level import (
    TwoLevelBasis,
    TwoLevelDipoleMatrix,
)


def test_shared_base_preserves_conversion_cache_and_si_view() -> None:
    dipole = TwoLevelDipoleMatrix(
        TwoLevelBasis(energy_gap=0.2),
        mu0=1.0,
        units_input="D",
    )

    assert isinstance(dipole, DipoleMatrixBase)
    assert dipole.mu0 == converter.convert_dipole_moment(1.0, "D", "C*m")
    assert dipole.units_input == "C*m"
    matrix = dipole.mu("x")
    assert dipole.mu("X") is matrix
    assert dipole.mu_x is matrix
    assert dipole.get_mu_x_SI() is matrix
    assert set(dipole._cache) == {("x", True)}
    assert dipole._cache[("x", True)] is matrix
    np.testing.assert_array_equal(
        matrix,
        np.array([[0.0, dipole.mu0], [dipole.mu0, 0.0]], dtype=np.complex128),
    )
    np.testing.assert_allclose(
        dipole.get_mu_in_units("x", "D"),
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
        rtol=1e-15,
        atol=0.0,
    )
    with pytest.raises(ValueError, match="axis must be"):
        dipole.mu("q")


def test_legacy_si_component_fallback_stays_duck_typed() -> None:
    matrix = np.array([[0.0, 2.0], [2.0, 0.0]])

    class LegacyDipole:
        mu_x = matrix

    legacy = LegacyDipole()
    assert get_dipole_component_SI(legacy, "x") is matrix
    with pytest.raises(AttributeError, match="mu_y"):
        get_dipole_component_SI(legacy, "y")
