"""Explicit physical-input contracts for direct construction APIs."""

from inspect import Parameter, signature

import pytest

from rovibrational_excitation.dipole import (
    LinMolDipoleMatrix,
    SymTopDipoleMatrix,
)
from rovibrational_excitation.dipole.factory import create_dipole_matrix
from rovibrational_excitation.models.two_level import (
    TwoLevelBasis,
    TwoLevelDipoleMatrix,
)
from rovibrational_excitation.models.vib_ladder import (
    VibLadderBasis,
    VibLadderDipoleMatrix,
)
from rovibrational_excitation.optimization.model import build_optimization_model


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


def test_generic_factory_rejects_model_owned_systems():
    twolevel = TwoLevelBasis(energy_gap=0.2)
    with pytest.raises(TypeError, match="未知の基底クラス"):
        create_dipole_matrix(twolevel, mu0=1.0, potential_type="harmonic")

    vibladder = VibLadderBasis(V_max=1, omega=0.2, delta_omega=0.0)
    with pytest.raises(TypeError, match="未知の基底クラス"):
        create_dipole_matrix(vibladder, mu0=1.0, potential_type="harmonic")


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
