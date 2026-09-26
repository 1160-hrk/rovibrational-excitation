"""Explicit initial interval controls for standard Krotov optimization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from rovibrational_excitation.core.units import converter
from rovibrational_excitation.fields import ElectricField, gaussian_fwhm

from .krotov_initial_field import KrotovGeneratedInitialField
from .krotov_timegrid import KrotovIntervalGrid

_GENERATED_KEYS = {
    "initial_amplitude",
    "initial_amplitude_units",
    "initial_carrier_frequency",
    "initial_carrier_frequency_units",
    "initial_center",
    "initial_center_units",
    "initial_duration",
    "initial_duration_units",
    "initial_polarization",
    "initial_gdd",
    "initial_gdd_units",
    "initial_tod",
    "initial_tod_units",
}
_SAMPLED_KEYS = {"initial_control_samples", "initial_control_units"}
KROTOV_CONTROL_OPTION_KEYS = frozenset(
    {"initial_control_kind", *_GENERATED_KEYS, *_SAMPLED_KEYS}
)


@dataclass(frozen=True, slots=True)
class KrotovGeneratedControl:
    """Generated Gaussian control evaluated at interval midpoints."""

    field: KrotovGeneratedInitialField

    def samples_on(self, grid: KrotovIntervalGrid) -> NDArray[np.float64]:
        times = grid.control_times_fs
        if times.size < 2:
            raise ValueError(
                "generated standard Krotov controls require at least two intervals"
            )
        electric_field = ElectricField(times, time_units="fs")
        electric_field.add_dispersed_Efield(
            envelope_func=gaussian_fwhm,
            duration=self.field.duration_fs,
            t_center=self.field.center_fs,
            carrier_freq=self.field.carrier_cycles_per_fs,
            duration_units="fs",
            t_center_units="fs",
            carrier_freq_units="PHz",
            amplitude=self.field.amplitude_v_per_m,
            amplitude_units="V/m",
            polarization=np.asarray(self.field.polarization, dtype=np.complex128),
            phase_rad=0.0,
            gdd=self.field.gdd_fs2,
            tod=self.field.tod_fs3,
            gdd_units="fs^2",
            tod_units="fs^3",
            const_polarisation=False,
        )
        return np.array(electric_field.get_Efield_SI(), dtype=np.float64, copy=True)


@dataclass(frozen=True, slots=True, eq=False)
class KrotovSampledControl:
    """Caller-owned interval controls converted once to V/m."""

    samples_v_per_m: NDArray[np.float64]

    def samples_on(self, grid: KrotovIntervalGrid) -> NDArray[np.float64]:
        if self.samples_v_per_m.shape[0] != grid.interval_count:
            raise ValueError(
                "initial_control_samples length must exactly match the standard "
                f"Krotov interval count ({self.samples_v_per_m.shape[0]} != "
                f"{grid.interval_count})"
            )
        return np.array(self.samples_v_per_m, dtype=np.float64, copy=True)


KrotovInitialControl: TypeAlias = KrotovGeneratedControl | KrotovSampledControl


def parse_krotov_initial_control(params: Mapping[str, Any]) -> KrotovInitialControl:
    """Parse one explicit generated or sampled interval-control source."""
    if "initial_field_kind" in params or any(
        key in params for key in ("initial_field_samples", "initial_field_units")
    ):
        raise ValueError(
            "standard Krotov uses initial_control_kind and interval controls; "
            "initial_field_kind/initial_field_samples belong to "
            "legacy_batch_overlap or GRAPE"
        )
    if "initial_control_kind" not in params:
        raise ValueError(
            "missing required standard Krotov parameter: initial_control_kind"
        )
    kind = params["initial_control_kind"]
    if kind == "generated":
        inapplicable = sorted(_SAMPLED_KEYS & params.keys())
        if inapplicable:
            raise ValueError(
                "sampled initial-control parameters are not applicable when "
                "initial_control_kind=generated: " + ", ".join(inapplicable)
            )
        return KrotovGeneratedControl(KrotovGeneratedInitialField.from_mapping(params))
    if kind != "sampled":
        raise ValueError("initial_control_kind must be one of: generated, sampled")

    missing = sorted(_SAMPLED_KEYS - params.keys())
    if missing:
        raise ValueError(
            "missing required sampled standard Krotov control parameters: "
            + ", ".join(missing)
        )
    inapplicable = sorted(_GENERATED_KEYS & params.keys())
    if inapplicable:
        raise ValueError(
            "generated initial-control parameters are not applicable when "
            "initial_control_kind=sampled: " + ", ".join(inapplicable)
        )
    units = params["initial_control_units"]
    if not isinstance(units, str) or units not in converter.get_supported_units(
        "field_amplitude"
    ):
        raise ValueError(
            "initial_control_units must be a supported direct electric-field "
            "amplitude unit"
        )
    raw = np.asarray(params["initial_control_samples"])
    if np.iscomplexobj(raw) or np.issubdtype(raw.dtype, np.bool_):
        raise ValueError("initial_control_samples must be real-valued")
    try:
        values = np.array(raw, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("initial_control_samples must contain real numbers") from exc
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("initial_control_samples must have shape (n_intervals, 2)")
    if not np.all(np.isfinite(values)):
        raise ValueError("initial_control_samples must contain only finite values")
    converted = np.asarray(
        converter.convert_electric_field(values, units, "V/m"), dtype=np.float64
    )
    converted.setflags(write=False)
    return KrotovSampledControl(converted)


__all__ = [
    "KROTOV_CONTROL_OPTION_KEYS",
    "KrotovGeneratedControl",
    "KrotovInitialControl",
    "KrotovSampledControl",
    "parse_krotov_initial_control",
]
