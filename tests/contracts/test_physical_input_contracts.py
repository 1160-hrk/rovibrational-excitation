"""Explicit physical-input contracts for direct construction APIs."""

from inspect import Parameter, signature

import pytest

from rovibrational_excitation.models.linear_molecule import LinMolDipoleMatrix
from rovibrational_excitation.models.symmetric_top import SymmetricTopDipoleMatrix
from rovibrational_excitation.models.two_level import TwoLevelDipoleMatrix
from rovibrational_excitation.models.vib_ladder import VibLadderDipoleMatrix
from rovibrational_excitation.optimization.model import build_optimization_model


@pytest.mark.parametrize(
    "dipole_type",
    [
        LinMolDipoleMatrix,
        SymmetricTopDipoleMatrix,
        TwoLevelDipoleMatrix,
        VibLadderDipoleMatrix,
    ],
)
def test_direct_dipole_construction_requires_mu0(dipole_type):
    assert signature(dipole_type).parameters["mu0"].default is Parameter.empty


@pytest.mark.parametrize(
    "dipole_type",
    [LinMolDipoleMatrix, SymmetricTopDipoleMatrix, VibLadderDipoleMatrix],
)
def test_vibrational_dipole_construction_requires_potential_type(dipole_type):
    assert (
        signature(dipole_type).parameters["potential_type"].default is Parameter.empty
    )


def test_optimization_twolevel_does_not_require_irrelevant_potential_type():
    model = build_optimization_model(
        {
            "type": "twolevel",
            "params": {
                "energy_gap": 0.2,
                "energy_gap_units": "rad/fs",
                "dipole_scale": 1.0,
                "dipole_scale_units": "C*m",
            },
        }
    )
    assert isinstance(model.dipole, TwoLevelDipoleMatrix)
