"""Dense and CSR transition dipoles for a parallel symmetric-top band."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias, cast

import numpy as np
import scipy.sparse as sp

from rovibrational_excitation.core.units.converters import converter
from rovibrational_excitation.models.parameters import SymmetricTopParameters

from .basis import SymmetricTopBasis
from .rotational import parallel_cartesian_direction_cosine

DipoleArray: TypeAlias = np.ndarray[Any, Any] | sp.csr_matrix


def _morse_level_parameter(omega01: float, anharmonic_shift: float) -> float:
    if anharmonic_shift == 0.0:
        raise ValueError("anharmonic_shift must be non-zero for a Morse potential")
    return (omega01 + anharmonic_shift) / anharmonic_shift - 0.5


def _validate_morse_v_max(v_max: int, level_parameter: float) -> None:
    maximum = math.floor(level_parameter) - 1
    if v_max > maximum:
        raise ValueError(
            f"V_max={v_max} exceeds the Morse limit {maximum} "
            f"derived from N={level_parameter:g}"
        )


def validate_symmetric_top_vibrational_basis(
    parameters: SymmetricTopParameters,
) -> None:
    """Validate Morse bound levels before allocating a transition matrix."""
    if parameters.potential_type != "morse":
        return
    level_parameter = _morse_level_parameter(
        parameters.vibrational_frequency.angular_rad_per_fs,
        parameters.anharmonic_shift.angular_rad_per_fs,
    )
    _validate_morse_v_max(parameters.v_max, level_parameter)


def _morse_vibrational_factor(
    v_bra: int,
    v_ket: int,
    level_parameter: float,
) -> float:
    normalization = (
        2.0
        / (2.0 * level_parameter - 1.0)
        * np.sqrt((level_parameter - 1.0) * level_parameter / (2.0 * level_parameter))
    )
    if v_bra == v_ket:
        return 0.0
    upper = max(v_bra, v_ket)
    lower = min(v_bra, v_ket)
    gamma_factors = np.arange(-upper + 1, -lower + 1) + 2.0 * level_parameter
    factorial_factors = np.arange(lower + 1, upper + 1)
    result = (
        2.0
        * (-1.0) ** (upper - lower + 1)
        / ((upper - lower) * (2.0 * level_parameter - lower - upper))
        * np.sqrt(
            (level_parameter - lower)
            * (level_parameter - upper)
            * np.prod(factorial_factors)
            / np.prod(gamma_factors)
        )
        / normalization
    )
    return float(result)


def _vibrational_factor(
    v_bra: int,
    v_ket: int,
    *,
    potential_type: str,
    morse_level_parameter: float,
) -> float:
    if potential_type == "harmonic":
        if v_bra == v_ket + 1:
            return math.sqrt(v_bra)
        if v_ket == v_bra + 1:
            return math.sqrt(v_ket)
        return 0.0
    return _morse_vibrational_factor(v_bra, v_ket, morse_level_parameter)


def build_parallel_dipole(
    basis: SymmetricTopBasis,
    axis: Literal["x", "y", "z"],
    mu0: float,
    *,
    potential_type: Literal["harmonic", "morse"],
    dense: bool,
) -> np.ndarray | sp.csr_matrix:
    """Build one Cartesian component from exact allowed neighbors."""
    parameters = basis.parameters
    morse_level_parameter = 0.0
    if potential_type == "morse":
        morse_level_parameter = _morse_level_parameter(
            parameters.vibrational_frequency.angular_rad_per_fs,
            parameters.anharmonic_shift.angular_rad_per_fs,
        )
        _validate_morse_v_max(parameters.v_max, morse_level_parameter)

    rows: list[int] = []
    columns: list[int] = []
    data: list[complex] = []
    spherical_components = (0,) if axis == "z" else (-1, 1)
    for row, state in enumerate(basis.basis):
        v_bra, j_bra, k_bra, m_bra = map(int, state)
        for v_ket in range(parameters.v_max + 1):
            vibrational = _vibrational_factor(
                v_bra,
                v_ket,
                potential_type=potential_type,
                morse_level_parameter=morse_level_parameter,
            )
            if vibrational == 0.0:
                continue
            for j_ket in range(max(0, j_bra - 1), min(parameters.j_max, j_bra + 1) + 1):
                if abs(k_bra) > j_ket:
                    continue
                for p in spherical_components:
                    m_ket = m_bra - p
                    if abs(m_ket) > j_ket:
                        continue
                    column = basis.index_map.get((v_ket, j_ket, k_bra, m_ket))
                    if column is None:
                        continue
                    rotational = parallel_cartesian_direction_cosine(
                        axis,
                        j_bra,
                        k_bra,
                        m_bra,
                        j_ket,
                        k_bra,
                        m_ket,
                    )
                    value = complex(mu0 * vibrational * rotational)
                    if value == 0.0j:
                        continue
                    rows.append(row)
                    columns.append(column)
                    data.append(value)

    shape = (basis.size(), basis.size())
    matrix = sp.csr_matrix((data, (rows, columns)), shape=shape, dtype=np.complex128)
    matrix.sort_indices()
    return matrix.toarray() if dense else matrix


@dataclass(slots=True)
class SymmetricTopDipoleMatrix:
    """Cached Cartesian dipole components for one production SymTop basis."""

    basis: SymmetricTopBasis
    mu0: float
    potential_type: Literal["harmonic", "morse"]
    backend: Literal["numpy", "cupy"] = "numpy"
    dense: bool = True
    units: Literal["C*m"] = "C*m"
    _cache: dict[tuple[str, bool], DipoleArray] = field(
        init=False,
        default_factory=dict,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.basis, SymmetricTopBasis):
            raise TypeError("basis must be a SymmetricTopBasis")
        if self.potential_type not in {"harmonic", "morse"}:
            raise ValueError("potential_type must be harmonic or morse")
        if self.backend != "numpy":
            raise ValueError("SymTop currently supports only backend='numpy'")

    def _build_mu_axis(
        self,
        axis: Literal["x", "y", "z"],
        *,
        dense: bool,
    ) -> DipoleArray:
        return build_parallel_dipole(
            self.basis,
            axis,
            self.mu0,
            potential_type=self.potential_type,
            dense=dense,
        )

    def mu(self, axis: str = "x", *, dense: bool | None = None) -> DipoleArray:
        axis_normalized = axis.lower()
        if axis_normalized not in {"x", "y", "z"}:
            raise ValueError("axis must be x, y, or z")
        selected_dense = self.dense if dense is None else dense
        key = (axis_normalized, selected_dense)
        if key not in self._cache:
            self._cache[key] = self._build_mu_axis(
                cast(Literal["x", "y", "z"], axis_normalized),
                dense=selected_dense,
            )
        return self._cache[key]

    def get_mu_in_units(
        self,
        axis: str,
        target_units: str,
        *,
        dense: bool | None = None,
    ) -> DipoleArray:
        matrix = self.mu(axis, dense=dense)
        if target_units == self.units:
            return matrix
        factor = converter.convert_dipole_moment(1.0, self.units, target_units)
        return matrix * factor

    def get_mu_x_SI(self, *, dense: bool | None = None) -> DipoleArray:
        return self.mu("x", dense=dense)

    def get_mu_y_SI(self, *, dense: bool | None = None) -> DipoleArray:
        return self.mu("y", dense=dense)

    def get_mu_z_SI(self, *, dense: bool | None = None) -> DipoleArray:
        return self.mu("z", dense=dense)


__all__ = [
    "SymmetricTopDipoleMatrix",
    "build_parallel_dipole",
    "validate_symmetric_top_vibrational_basis",
]
