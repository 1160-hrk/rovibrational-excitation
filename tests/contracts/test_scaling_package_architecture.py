"""Import and ownership contracts for the target dynamics.scaling package."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src" / "rovibrational_excitation"
SCALING = PACKAGE / "dynamics" / "scaling"
LEGACY_SCALING = PACKAGE / "core" / "nondimensional"


def test_nondimensionalization_is_owned_by_dynamics_scaling():
    from rovibrational_excitation.dynamics.scaling import (
        NondimensionalizationScales,
        ScaleValue,
        analyze_regime,
        nondimensionalize_from_objects,
        nondimensionalize_system,
    )

    assert (SCALING / "__init__.py").is_file()
    assert (SCALING / "converter.py").is_file()
    assert (SCALING / "reporting.py").is_file()
    assert (SCALING / "scales.py").is_file()
    assert (SCALING / "utils.py").is_file()
    assert not (LEGACY_SCALING / "__init__.py").exists()
    assert not (LEGACY_SCALING / "converter.py").exists()
    assert NondimensionalizationScales.__module__ == (
        "rovibrational_excitation.dynamics.scaling.scales"
    )
    assert ScaleValue.__module__ == "rovibrational_excitation.dynamics.scaling.scales"
    assert analyze_regime.__module__ == (
        "rovibrational_excitation.dynamics.scaling.reporting"
    )
    assert nondimensionalize_from_objects.__module__ == (
        "rovibrational_excitation.dynamics.scaling.converter"
    )
    assert nondimensionalize_system.__module__ == (
        "rovibrational_excitation.dynamics.scaling.converter"
    )
