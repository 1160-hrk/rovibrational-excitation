"""Explicit nuclear-spin assignments for rotational symmetry states."""

from __future__ import annotations

from dataclasses import dataclass


def _nonnegative_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class RotationalSymmetryState:
    """Quantum numbers relevant to symmetry classification, excluding M."""

    vibrational_quantum_number: int
    rotational_quantum_number: int
    body_projection_quantum_number: int | None
    vibronic_symmetry: str

    def __post_init__(self) -> None:
        _nonnegative_integer(
            self.vibrational_quantum_number,
            name="vibrational_quantum_number",
        )
        j = _nonnegative_integer(
            self.rotational_quantum_number,
            name="rotational_quantum_number",
        )
        k = self.body_projection_quantum_number
        if k is not None:
            if isinstance(k, bool) or not isinstance(k, int):
                raise TypeError("body_projection_quantum_number must be an integer")
            if abs(k) > j:
                raise ValueError(
                    "absolute body_projection_quantum_number cannot exceed "
                    "rotational_quantum_number"
                )
        if (
            not isinstance(self.vibronic_symmetry, str)
            or not self.vibronic_symmetry.strip()
        ):
            raise ValueError("vibronic_symmetry must be a non-empty string")


@dataclass(frozen=True, slots=True)
class NuclearSpinAssignment:
    """One state's spin-isomer sector, weight, and requested-filter status."""

    sector: str
    statistical_weight: int | None
    allowed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.sector, str) or not self.sector.strip():
            raise ValueError("sector must be a non-empty string")
        if self.statistical_weight is not None:
            _nonnegative_integer(self.statistical_weight, name="statistical_weight")
        if not isinstance(self.allowed, bool):
            raise TypeError("allowed must be a bool")


@dataclass(frozen=True, slots=True)
class EvenOddJStatistics:
    """Nuclear-spin weights for two-equivalent-nucleus linear rotors."""

    even_weight: int
    odd_weight: int
    even_sector: str
    odd_sector: str

    def __post_init__(self) -> None:
        _nonnegative_integer(self.even_weight, name="even_weight")
        _nonnegative_integer(self.odd_weight, name="odd_weight")
        if (
            not isinstance(self.even_sector, str)
            or not self.even_sector.strip()
            or not isinstance(self.odd_sector, str)
            or not self.odd_sector.strip()
        ):
            raise ValueError("spin-isomer sector names must be non-empty")
        if self.even_sector == self.odd_sector:
            raise ValueError("even and odd spin-isomer sectors must differ")

    @property
    def supported_isomers(self) -> tuple[str, ...]:
        return ("all", self.even_sector, self.odd_sector)

    def assign(
        self,
        state: RotationalSymmetryState,
        *,
        nuclear_spin_isomer: str,
    ) -> NuclearSpinAssignment:
        if state.body_projection_quantum_number is not None:
            raise ValueError(
                "linear nuclear-spin policy does not accept a K quantum number"
            )
        if nuclear_spin_isomer not in self.supported_isomers:
            raise ValueError(
                "nuclear_spin_isomer must be one of "
                + ", ".join(self.supported_isomers)
            )
        is_even = state.rotational_quantum_number % 2 == 0
        sector = self.even_sector if is_even else self.odd_sector
        weight = self.even_weight if is_even else self.odd_weight
        return NuclearSpinAssignment(
            sector=sector,
            statistical_weight=weight,
            allowed=nuclear_spin_isomer in {"all", sector},
        )


@dataclass(frozen=True, slots=True)
class ConstantNuclearSpinStatistics:
    """State-independent spin degeneracy for distinguishable nuclei."""

    weight: int

    def __post_init__(self) -> None:
        _nonnegative_integer(self.weight, name="weight")

    def assign(
        self,
        state: RotationalSymmetryState,
        *,
        nuclear_spin_isomer: str,
    ) -> NuclearSpinAssignment:
        del state
        if nuclear_spin_isomer != "all":
            raise ValueError("nuclear_spin_isomer must be all for this preset")
        return NuclearSpinAssignment(
            sector="single",
            statistical_weight=self.weight,
            allowed=True,
        )


@dataclass(frozen=True, slots=True)
class KModuloNSpinSectors:
    """Sector classification for an n-fold symmetric top.

    This policy intentionally leaves weights unresolved. A raw signed-K basis
    cannot represent all symmetry-adapted +/-K combinations needed for general
    statistical weights.
    """

    order: int
    zero_residue_sector: str
    other_residue_sector: str

    def __post_init__(self) -> None:
        if isinstance(self.order, bool) or not isinstance(self.order, int):
            raise TypeError("order must be an integer")
        if self.order < 2:
            raise ValueError("order must be at least two")
        if (
            not isinstance(self.zero_residue_sector, str)
            or not self.zero_residue_sector.strip()
            or not isinstance(self.other_residue_sector, str)
            or not self.other_residue_sector.strip()
        ):
            raise ValueError("spin-isomer sector names must be non-empty")
        if self.zero_residue_sector == self.other_residue_sector:
            raise ValueError("K-residue spin-isomer sectors must differ")

    @property
    def supported_isomers(self) -> tuple[str, ...]:
        return ("all", self.zero_residue_sector, self.other_residue_sector)

    def assign(
        self,
        state: RotationalSymmetryState,
        *,
        nuclear_spin_isomer: str,
    ) -> NuclearSpinAssignment:
        if nuclear_spin_isomer not in self.supported_isomers:
            raise ValueError(
                "nuclear_spin_isomer must be one of "
                + ", ".join(self.supported_isomers)
            )
        k = state.body_projection_quantum_number
        if k is None:
            raise ValueError("symmetric-top nuclear-spin policy requires K")
        sector = (
            self.zero_residue_sector
            if abs(k) % self.order == 0
            else self.other_residue_sector
        )
        return NuclearSpinAssignment(
            sector=sector,
            statistical_weight=None,
            allowed=nuclear_spin_isomer in {"all", sector},
        )


NuclearSpinPolicy = (
    EvenOddJStatistics | ConstantNuclearSpinStatistics | KModuloNSpinSectors
)


__all__ = [
    "ConstantNuclearSpinStatistics",
    "EvenOddJStatistics",
    "KModuloNSpinSectors",
    "NuclearSpinAssignment",
    "NuclearSpinPolicy",
    "RotationalSymmetryState",
]
