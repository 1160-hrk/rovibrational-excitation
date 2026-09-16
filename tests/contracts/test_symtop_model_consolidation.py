"""Ownership contracts after removing the experimental legacy SymTop model."""

from pathlib import Path

import rovibrational_excitation.dipole as dipole_package
from rovibrational_excitation.models.symmetric_top import (
    SymmetricTopBasis,
    SymmetricTopDipoleMatrix,
)

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src" / "rovibrational_excitation"


def test_production_symtop_has_one_model_owner() -> None:
    assert SymmetricTopBasis.__module__ == (
        "rovibrational_excitation.models.symmetric_top.basis"
    )
    assert SymmetricTopDipoleMatrix.__module__ == (
        "rovibrational_excitation.models.symmetric_top.dipole"
    )
    assert dipole_package.__all__ == []


def test_experimental_legacy_symtop_paths_are_absent() -> None:
    assert not (PACKAGE / "core" / "basis" / "symtop.py").exists()
    assert not (PACKAGE / "dipole" / "symtop").exists()
    assert not (PACKAGE / "dipole" / "factory.py").exists()
    assert not (PACKAGE / "dipole" / "rot" / "jmk.py").exists()
