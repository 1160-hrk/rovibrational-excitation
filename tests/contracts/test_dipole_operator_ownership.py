"""Shared dipole implementation belongs to models; solvers use a core protocol."""

from pathlib import Path

from rovibrational_excitation.core.dipole import DipoleOperator
from rovibrational_excitation.models.dipole_base import DipoleMatrixBase
from rovibrational_excitation.models.linear_molecule import LinMolDipoleMatrix
from rovibrational_excitation.models.symmetric_top import SymmetricTopDipoleMatrix
from rovibrational_excitation.models.two_level import TwoLevelDipoleMatrix
from rovibrational_excitation.models.vib_ladder import VibLadderDipoleMatrix

PACKAGE = Path(__file__).resolve().parents[2] / "src/rovibrational_excitation"


def test_shared_concrete_base_has_one_model_owner() -> None:
    assert DipoleMatrixBase.__module__ == "rovibrational_excitation.models.dipole_base"
    for matrix_type in (
        LinMolDipoleMatrix,
        TwoLevelDipoleMatrix,
        VibLadderDipoleMatrix,
    ):
        assert issubclass(matrix_type, DipoleMatrixBase)
    assert not issubclass(SymmetricTopDipoleMatrix, DipoleMatrixBase)
    assert not (PACKAGE / "dipole/base.py").exists()


def test_core_protocol_names_only_required_matrix_accessors() -> None:
    assert DipoleOperator.__module__ == "rovibrational_excitation.core.dipole"
    for name in (
        "get_mu_in_units",
        "get_mu_x_SI",
        "get_mu_y_SI",
        "get_mu_z_SI",
    ):
        assert callable(getattr(DipoleOperator, name))
