"""Contracts for explicit molecular-symmetry preset resolution."""

from __future__ import annotations

import pytest

from rovibrational_excitation.models.symmetry import (
    MolecularSymmetry,
    PointGroupFamily,
    RotationalSymmetryState,
    available_molecule_presets,
    resolve_molecule_preset,
)


def _state(*, j: int, k: int | None = None) -> RotationalSymmetryState:
    return RotationalSymmetryState(
        vibrational_quantum_number=0,
        rotational_quantum_number=j,
        body_projection_quantum_number=k,
        vibronic_symmetry="totally_symmetric",
    )


def test_parameterized_point_groups_require_exact_order_contract() -> None:
    c3v = MolecularSymmetry(PointGroupFamily.CNV, order=3)
    assert c3v.label == "C3v"

    with pytest.raises(ValueError, match="order is required"):
        MolecularSymmetry(PointGroupFamily.CNV, order=None)
    with pytest.raises(ValueError, match="order is not applicable"):
        MolecularSymmetry(PointGroupFamily.D_INFINITY_H, order=2)


def test_named_alias_resolves_to_canonical_isotopologue_without_constants() -> None:
    preset = resolve_molecule_preset("hydrogen")

    assert preset.canonical_id == "H2"
    assert preset.formula == "H2"
    assert preset.model_family == "linear"
    assert preset.symmetry.label == "Dinfh"
    assert preset.physical_parameters == {}
    assert preset.metadata["rule_version"] == 1
    assert preset.metadata["canonical_id"] == "H2"

    assert available_molecule_presets() == ("H2", "D2", "T2", "HD", "CH3F")
    with pytest.raises(ValueError, match="Unknown molecule symmetry preset"):
        resolve_molecule_preset("unknown molecule")


@pytest.mark.parametrize(
    ("name", "even_weight", "odd_weight", "even_sector", "odd_sector"),
    [
        ("H2", 1, 3, "para", "ortho"),
        ("D2", 6, 3, "ortho", "para"),
        ("T2", 1, 3, "para", "ortho"),
    ],
)
def test_hydrogen_isotopologue_presets_fix_even_odd_j_statistics(
    name: str,
    even_weight: int,
    odd_weight: int,
    even_sector: str,
    odd_sector: str,
) -> None:
    preset = resolve_molecule_preset(name)

    even = preset.assign(_state(j=0), nuclear_spin_isomer="all")
    odd = preset.assign(_state(j=1), nuclear_spin_isomer="all")
    assert (even.statistical_weight, even.sector, even.allowed) == (
        even_weight,
        even_sector,
        True,
    )
    assert (odd.statistical_weight, odd.sector, odd.allowed) == (
        odd_weight,
        odd_sector,
        True,
    )

    selected = preset.assign(_state(j=0), nuclear_spin_isomer=odd_sector)
    assert selected.statistical_weight == even_weight
    assert selected.sector == even_sector
    assert not selected.allowed


def test_hd_has_state_independent_spin_degeneracy_and_no_parity_filter() -> None:
    preset = resolve_molecule_preset("HD")

    assert preset.assign(_state(j=0), nuclear_spin_isomer="all").statistical_weight == 6
    assert preset.assign(_state(j=1), nuclear_spin_isomer="all").statistical_weight == 6
    with pytest.raises(ValueError, match="nuclear_spin_isomer must be all"):
        preset.assign(_state(j=0), nuclear_spin_isomer="ortho")


def test_ch3f_preset_classifies_k_sectors_without_inventing_weights() -> None:
    preset = resolve_molecule_preset("methyl fluoride")

    assert preset.canonical_id == "CH3F"
    assert preset.model_family == "symmetric_top"
    assert preset.symmetry.label == "C3v"
    assert preset.symmetry.permutation_inversion_group == "C3v(M)"

    ortho = preset.assign(_state(j=3, k=-3), nuclear_spin_isomer="all")
    para = preset.assign(_state(j=3, k=1), nuclear_spin_isomer="all")
    assert (ortho.sector, ortho.statistical_weight, ortho.allowed) == (
        "ortho",
        None,
        True,
    )
    assert (para.sector, para.statistical_weight, para.allowed) == (
        "para",
        None,
        True,
    )
    assert not preset.assign(_state(j=3, k=1), nuclear_spin_isomer="ortho").allowed
    assert not preset.assign(_state(j=3, k=3), nuclear_spin_isomer="para").allowed

    with pytest.raises(ValueError, match="not available until the signed-K basis"):
        preset.require_statistical_weight(_state(j=3, k=1))


def test_symmetry_state_validation_never_repairs_quantum_numbers() -> None:
    with pytest.raises(ValueError, match="rotational_quantum_number"):
        _state(j=-1)
    with pytest.raises(ValueError, match="cannot exceed"):
        _state(j=1, k=2)

    h2 = resolve_molecule_preset("H2")
    with pytest.raises(ValueError, match="does not accept a K quantum number"):
        h2.assign(_state(j=1, k=0), nuclear_spin_isomer="all")

    unsupported = RotationalSymmetryState(0, 0, None, "other")
    with pytest.raises(ValueError, match="does not support vibronic symmetry"):
        h2.assign(unsupported, nuclear_spin_isomer="all")
