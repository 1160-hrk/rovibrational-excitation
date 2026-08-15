"""Ownership contracts for generic numerical validation."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src" / "rovibrational_excitation"
CORE_VALIDATION = PACKAGE / "core" / "validation.py"
LEGACY_VALIDATION = PACKAGE / "dynamics" / "algorithms" / "validation.py"


def test_generic_numerical_validation_is_owned_by_core():
    from rovibrational_excitation.core.validation import (
        NUMERICAL_VALIDATION_EPSILON_FACTOR,
        density_matrix_tolerance,
        validate_density_matrix_problem,
        validate_density_matrix_properties,
        validate_wavefunction_problem,
    )

    assert CORE_VALIDATION.is_file()
    assert not LEGACY_VALIDATION.exists()
    assert NUMERICAL_VALIDATION_EPSILON_FACTOR == 100.0
    assert density_matrix_tolerance.__module__ == (
        "rovibrational_excitation.core.validation"
    )
    assert validate_density_matrix_problem.__module__ == (
        "rovibrational_excitation.core.validation"
    )
    assert validate_density_matrix_properties.__module__ == (
        "rovibrational_excitation.core.validation"
    )
    assert validate_wavefunction_problem.__module__ == (
        "rovibrational_excitation.core.validation"
    )
