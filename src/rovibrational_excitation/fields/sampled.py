"""Immutable sampled electric-field values on one canonical time grid."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, TypeAlias

import numpy as np
from numpy.typing import NDArray

from rovibrational_excitation.core.time import TimeGrid


def _validated_component(
    time_grid: TimeGrid,
    samples: Any,
    *,
    name: str,
) -> NDArray[np.float64]:
    if not isinstance(time_grid, TimeGrid):
        raise TypeError("time_grid must be a TimeGrid")

    raw = np.asarray(samples)
    if np.iscomplexobj(raw) or np.issubdtype(raw.dtype, np.bool_):
        raise ValueError(f"{name} must be real-valued")
    try:
        values = np.array(raw, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain real numbers") from exc

    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if values.size != time_grid.field_times_fs.size:
        raise ValueError(
            f"{name} length must exactly match time_grid; "
            f"got {values.size} and {time_grid.field_times_fs.size}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")

    values.setflags(write=False)
    return values


class _SampledFieldTiming:
    """Shared read-only timing projection required by existing solvers."""

    time_grid: TimeGrid
    time_units: ClassVar[str] = "fs"
    field_units: ClassVar[str] = "V/m"

    @property
    def tlist(self) -> NDArray[np.float64]:
        return self.time_grid.field_times_fs

    @property
    def dt(self) -> float:
        return float(self.time_grid.field_dt_fs)

    @property
    def dt_state(self) -> float:
        return float(self.time_grid.propagation_dt_fs)

    @property
    def steps_state(self) -> int:
        return int(self.time_grid.propagation_steps)

    def get_time_SI(self) -> NDArray[np.float64]:
        return self.tlist


@dataclass(frozen=True, slots=True, eq=False)
class ScalarField(_SampledFieldTiming):
    """One real scalar electric-field waveform in canonical V/m units."""

    time_grid: TimeGrid
    samples_v_per_m: NDArray[np.float64]

    def __post_init__(self) -> None:
        values = _validated_component(
            self.time_grid,
            self.samples_v_per_m,
            name="scalar field samples",
        )
        object.__setattr__(self, "samples_v_per_m", values)

    @property
    def Efield(self) -> NDArray[np.float64]:
        """Temporary one-column projection for legacy diagnostics."""
        return self.samples_v_per_m[:, np.newaxis]

    def get_Efield(self) -> NDArray[np.float64]:
        return self.Efield

    def get_Efield_SI(self) -> NDArray[np.float64]:
        return self.Efield

    def get_scalar_field(self) -> NDArray[np.float64]:
        return self.samples_v_per_m

    def get_pol(self) -> NDArray[np.complex128]:
        raise ValueError("ScalarField has no polarization")

    def get_field_scale_factor(self) -> float:
        return float(np.max(np.abs(self.samples_v_per_m)))


@dataclass(frozen=True, slots=True, eq=False)
class CartesianField(_SampledFieldTiming):
    """Two ordered real Cartesian field components in canonical V/m units."""

    time_grid: TimeGrid
    first_component_v_per_m: NDArray[np.float64]
    second_component_v_per_m: NDArray[np.float64]
    scalar_samples_v_per_m: NDArray[np.float64] | None = None
    jones_polarization: NDArray[np.complex128] | None = None
    components_v_per_m: NDArray[np.float64] = field(init=False)

    def __post_init__(self) -> None:
        first = _validated_component(
            self.time_grid,
            self.first_component_v_per_m,
            name="first Cartesian field component",
        )
        second = _validated_component(
            self.time_grid,
            self.second_component_v_per_m,
            name="second Cartesian field component",
        )
        scalar = self.scalar_samples_v_per_m
        polarization = self.jones_polarization
        if (scalar is None) != (polarization is None):
            raise ValueError(
                "scalar_samples_v_per_m and jones_polarization must be "
                "provided together"
            )
        if scalar is not None:
            scalar = _validated_component(
                self.time_grid, scalar, name="decomposed scalar field samples"
            )
            raw_polarization = np.asarray(polarization)
            if np.issubdtype(raw_polarization.dtype, np.bool_):
                raise ValueError("jones_polarization must be complex-valued numbers")
            try:
                polarization = np.array(
                    raw_polarization, dtype=np.complex128, copy=True
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "jones_polarization must contain complex-valued numbers"
                ) from exc
            if polarization.shape != (2,):
                raise ValueError("jones_polarization must have shape (2,)")
            if not np.all(np.isfinite(polarization)):
                raise ValueError("jones_polarization must contain only finite values")
            polarization.setflags(write=False)

        components = np.column_stack((first, second))
        components.setflags(write=False)
        object.__setattr__(self, "first_component_v_per_m", first)
        object.__setattr__(self, "second_component_v_per_m", second)
        object.__setattr__(self, "scalar_samples_v_per_m", scalar)
        object.__setattr__(self, "jones_polarization", polarization)
        object.__setattr__(self, "components_v_per_m", components)

    @property
    def Efield(self) -> NDArray[np.float64]:
        return self.components_v_per_m

    def get_Efield(self) -> NDArray[np.float64]:
        return self.components_v_per_m

    def get_Efield_SI(self) -> NDArray[np.float64]:
        return self.components_v_per_m

    def get_scalar_field(self) -> NDArray[np.float64]:
        if self.scalar_samples_v_per_m is None:
            raise ValueError("CartesianField has no scalar waveform decomposition")
        return self.scalar_samples_v_per_m

    def get_pol(self) -> NDArray[np.complex128]:
        if self.jones_polarization is None:
            raise ValueError(
                "CartesianField has no constant Jones polarization decomposition"
            )
        return self.jones_polarization

    def get_field_scale_factor(self) -> float:
        return float(np.max(np.linalg.norm(self.components_v_per_m, axis=1)))


SampledField: TypeAlias = ScalarField | CartesianField

__all__ = ["CartesianField", "SampledField", "ScalarField"]
