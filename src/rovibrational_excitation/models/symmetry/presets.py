"""Named, source-identified molecular-symmetry presets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .groups import MolecularSymmetry, PointGroupFamily
from .policy import (
    ConstantNuclearSpinStatistics,
    EvenOddJStatistics,
    KModuloNSpinSectors,
    NuclearSpinAssignment,
    NuclearSpinPolicy,
    RotationalSymmetryState,
)

ModelFamily = Literal["linear", "symmetric_top"]

_HYDROGEN_REFERENCE = "https://pubs.acs.org/doi/10.1021/acs.jpca.1c06468"
_CH3F_REFERENCE = "https://jetp.ras.ru/cgi-bin/dn/e_070_05_0895.pdf"


@dataclass(frozen=True, slots=True)
class MoleculeSymmetryPreset:
    """A name resolver for symmetry only, never molecular constants."""

    canonical_id: str
    formula: str
    aliases: tuple[str, ...]
    model_family: ModelFamily
    symmetry: MolecularSymmetry
    supported_vibronic_symmetries: tuple[str, ...]
    nuclear_spin_policy: NuclearSpinPolicy
    rule_source: str
    rule_version: int

    def __post_init__(self) -> None:
        for name, value in (
            ("canonical_id", self.canonical_id),
            ("formula", self.formula),
            ("rule_source", self.rule_source),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.model_family not in {"linear", "symmetric_top"}:
            raise ValueError("model_family must be linear or symmetric_top")
        if not isinstance(self.symmetry, MolecularSymmetry):
            raise TypeError("symmetry must be a MolecularSymmetry")
        if not isinstance(
            self.nuclear_spin_policy,
            (EvenOddJStatistics, ConstantNuclearSpinStatistics, KModuloNSpinSectors),
        ):
            raise TypeError("nuclear_spin_policy has an unsupported policy type")
        if not self.aliases or any(
            not isinstance(alias, str) or not alias.strip() for alias in self.aliases
        ):
            raise ValueError("aliases must contain only non-empty strings")
        if not self.supported_vibronic_symmetries or any(
            not isinstance(label, str) or not label.strip()
            for label in self.supported_vibronic_symmetries
        ):
            raise ValueError(
                "supported_vibronic_symmetries must contain non-empty strings"
            )
        if isinstance(self.rule_version, bool) or not isinstance(
            self.rule_version, int
        ):
            raise TypeError("rule_version must be a positive integer")
        if self.rule_version <= 0:
            raise ValueError("rule_version must be a positive integer")

    @property
    def physical_parameters(self) -> dict[str, float]:
        """Return no constants: a symmetry preset is never a parameter fallback."""
        return {}

    @property
    def metadata(self) -> dict[str, str | int]:
        """Return serialization-safe identity and provenance."""
        return {
            "canonical_id": self.canonical_id,
            "formula": self.formula,
            "point_group": self.symmetry.label,
            "rule_source": self.rule_source,
            "rule_version": self.rule_version,
        }

    def assign(
        self,
        state: RotationalSymmetryState,
        *,
        nuclear_spin_isomer: str,
    ) -> NuclearSpinAssignment:
        """Classify one state without modifying its quantum numbers."""
        if state.vibronic_symmetry not in self.supported_vibronic_symmetries:
            raise ValueError(
                f"{self.canonical_id} preset does not support vibronic symmetry "
                f"{state.vibronic_symmetry!r}; supported values: "
                + ", ".join(self.supported_vibronic_symmetries)
            )
        return self.nuclear_spin_policy.assign(
            state,
            nuclear_spin_isomer=nuclear_spin_isomer,
        )

    def require_statistical_weight(self, state: RotationalSymmetryState) -> int:
        """Return a known weight or reject an unsupported thermal use."""
        assignment = self.assign(state, nuclear_spin_isomer="all")
        if assignment.statistical_weight is None:
            raise ValueError(
                f"statistical weights for {self.canonical_id} are not available "
                "until the signed-K basis is symmetry adapted"
            )
        return assignment.statistical_weight


_PRESETS = (
    MoleculeSymmetryPreset(
        canonical_id="H2",
        formula="H2",
        aliases=("H2", "hydrogen", "molecular hydrogen", "protium"),
        model_family="linear",
        symmetry=MolecularSymmetry(PointGroupFamily.D_INFINITY_H, order=None),
        supported_vibronic_symmetries=("totally_symmetric",),
        nuclear_spin_policy=EvenOddJStatistics(1, 3, "para", "ortho"),
        rule_source=_HYDROGEN_REFERENCE,
        rule_version=1,
    ),
    MoleculeSymmetryPreset(
        canonical_id="D2",
        formula="D2",
        aliases=("D2", "deuterium", "molecular deuterium"),
        model_family="linear",
        symmetry=MolecularSymmetry(PointGroupFamily.D_INFINITY_H, order=None),
        supported_vibronic_symmetries=("totally_symmetric",),
        nuclear_spin_policy=EvenOddJStatistics(6, 3, "ortho", "para"),
        rule_source=_HYDROGEN_REFERENCE,
        rule_version=1,
    ),
    MoleculeSymmetryPreset(
        canonical_id="T2",
        formula="T2",
        aliases=("T2", "tritium", "molecular tritium"),
        model_family="linear",
        symmetry=MolecularSymmetry(PointGroupFamily.D_INFINITY_H, order=None),
        supported_vibronic_symmetries=("totally_symmetric",),
        nuclear_spin_policy=EvenOddJStatistics(1, 3, "para", "ortho"),
        rule_source=_HYDROGEN_REFERENCE,
        rule_version=1,
    ),
    MoleculeSymmetryPreset(
        canonical_id="HD",
        formula="HD",
        aliases=("HD", "hydrogen deuteride"),
        model_family="linear",
        symmetry=MolecularSymmetry(PointGroupFamily.C_INFINITY_V, order=None),
        supported_vibronic_symmetries=("totally_symmetric",),
        nuclear_spin_policy=ConstantNuclearSpinStatistics(6),
        rule_source=_HYDROGEN_REFERENCE,
        rule_version=1,
    ),
    MoleculeSymmetryPreset(
        canonical_id="CH3F",
        formula="CH3F",
        aliases=("CH3F", "methyl fluoride", "fluoromethane"),
        model_family="symmetric_top",
        symmetry=MolecularSymmetry(
            PointGroupFamily.CNV,
            order=3,
            permutation_inversion_group="C3v(M)",
        ),
        supported_vibronic_symmetries=("totally_symmetric",),
        nuclear_spin_policy=KModuloNSpinSectors(3, "ortho", "para"),
        rule_source=_CH3F_REFERENCE,
        rule_version=1,
    ),
)


def _normalized_name(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("molecule preset name must be a string")
    result = " ".join(value.strip().casefold().split())
    if not result:
        raise ValueError("molecule preset name must be non-empty")
    return result


_ALIASES: dict[str, MoleculeSymmetryPreset] = {}
for _preset in _PRESETS:
    for _alias in (_preset.canonical_id, _preset.formula, *_preset.aliases):
        _key = _normalized_name(_alias)
        if _key in _ALIASES and _ALIASES[_key] is not _preset:
            raise RuntimeError(f"duplicate molecule symmetry preset alias: {_alias}")
        _ALIASES[_key] = _preset


def available_molecule_presets() -> tuple[str, ...]:
    """Return canonical preset identifiers in stable registry order."""
    return tuple(preset.canonical_id for preset in _PRESETS)


def resolve_molecule_preset(name: str) -> MoleculeSymmetryPreset:
    """Resolve an explicit alias; unknown names never fall back to a model."""
    key = _normalized_name(name)
    try:
        return _ALIASES[key]
    except KeyError:
        raise ValueError(
            "Unknown molecule symmetry preset: "
            f"{name!r}; available canonical IDs: "
            + ", ".join(available_molecule_presets())
        ) from None


__all__ = [
    "ModelFamily",
    "MoleculeSymmetryPreset",
    "available_molecule_presets",
    "resolve_molecule_preset",
]
