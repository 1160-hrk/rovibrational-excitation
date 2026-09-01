"""Minimal molecular-symmetry descriptors without a character-table engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PointGroupFamily(str, Enum):
    """Point-group families needed by current and planned rotor models."""

    C1 = "C1"
    CS = "Cs"
    CI = "Ci"
    C2V = "C2v"
    C_INFINITY_V = "Cinfv"
    D_INFINITY_H = "Dinfh"
    CNV = "Cnv"
    DNH = "Dnh"
    DND = "Dnd"


_PARAMETERIZED_FAMILIES = {
    PointGroupFamily.CNV,
    PointGroupFamily.DNH,
    PointGroupFamily.DND,
}


@dataclass(frozen=True, slots=True)
class MolecularSymmetry:
    """Geometric point group plus an optional molecular-symmetry group label.

    This value deliberately does not derive nuclear-spin statistics from a
    point-group name. Equivalent-nucleus permutation symmetry and vibronic
    symmetry belong to a separately validated policy.
    """

    family: PointGroupFamily
    order: int | None
    permutation_inversion_group: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.family, PointGroupFamily):
            raise TypeError("family must be a PointGroupFamily")
        if self.family in _PARAMETERIZED_FAMILIES:
            if self.order is None:
                raise ValueError(f"order is required for {self.family.value}")
            if isinstance(self.order, bool) or not isinstance(self.order, int):
                raise TypeError("order must be an integer")
            if self.order < 2:
                raise ValueError("order must be at least two")
        elif self.order is not None:
            raise ValueError(f"order is not applicable to {self.family.value}")
        if self.permutation_inversion_group is not None and (
            not isinstance(self.permutation_inversion_group, str)
            or not self.permutation_inversion_group.strip()
        ):
            raise ValueError("permutation_inversion_group must be a non-empty string")

    @property
    def label(self) -> str:
        """Return one stable ASCII point-group label."""
        if self.family is PointGroupFamily.CNV:
            return f"C{self.order}v"
        if self.family is PointGroupFamily.DNH:
            return f"D{self.order}h"
        if self.family is PointGroupFamily.DND:
            return f"D{self.order}d"
        return str(self.family.value)


__all__ = ["MolecularSymmetry", "PointGroupFamily"]
