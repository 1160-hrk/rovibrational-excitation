"""Explicit physical-input contracts for direct construction APIs."""

from inspect import Parameter, signature

import pytest

from rovibrational_excitation.core.basis import TwoLevelBasis, VibLadderBasis
from rovibrational_excitation.dipole import (
    LinMolDipoleMatrix,
    SymTopDipoleMatrix,
    TwoLevelDipoleMatrix,
    VibLadderDipoleMatrix,
)
from rovibrational_excitation.dipole.factory import create_dipole_matrix
from rovibrational_excitation.simulation.optimize_runner import _build_dipole


@pytest.mark.parametrize(
    "dipole_type",
    [
        LinMolDipoleMatrix,
        SymTopDipoleMatrix,
        TwoLevelDipoleMatrix,
        VibLadderDipoleMatrix,
    ],
)
def test_direct_dipole_construction_requires_mu0(dipole_type):
    assert signature(dipole_type).parameters["mu0"].default is Parameter.empty


@pytest.mark.parametrize(
    "dipole_type",
    [LinMolDipoleMatrix, SymTopDipoleMatrix, VibLadderDipoleMatrix],
)
def test_vibrational_dipole_construction_requires_potential_type(dipole_type):
    assert (
        signature(dipole_type).parameters["potential_type"].default is Parameter.empty
    )


def test_generic_factory_requires_potential_only_for_vibrational_models():
    twolevel = TwoLevelBasis(energy_gap=0.2)
    dipole = create_dipole_matrix(twolevel, mu0=1.0)
    assert isinstance(dipole, TwoLevelDipoleMatrix)

    with pytest.raises(ValueError, match="not applicable"):
        create_dipole_matrix(twolevel, mu0=1.0, potential_type="harmonic")

    vibladder = VibLadderBasis(V_max=1, omega=0.2, delta_omega=0.0)
    with pytest.raises(TypeError, match="potential_type is required"):
        create_dipole_matrix(vibladder, mu0=1.0)


def test_optimization_twolevel_does_not_require_irrelevant_potential_type():
    basis = TwoLevelBasis(energy_gap=0.2)
    dipole = _build_dipole(
        basis,
        {"params": {"mu0": 1.0, "unit_dipole": "C*m"}},
    )
    assert isinstance(dipole, TwoLevelDipoleMatrix)
