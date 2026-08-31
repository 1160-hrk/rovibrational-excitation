"""Canonical generated-field parameters resolved from explicit value/unit pairs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from rovibrational_excitation.core.time import TimeGrid
from rovibrational_excitation.core.units import (
    ElectricFieldAmplitude,
    Frequency,
    GroupDelayDispersion,
    ThirdOrderDispersion,
    TimeQuantity,
)
from rovibrational_excitation.fields.envelopes import get_generated_envelope


def _finite_scalar(params: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = params.get(key, default)
    if isinstance(value, (bool, np.bool_)) or np.asarray(value).ndim != 0:
        raise TypeError(f"{key} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{key} must be a finite number") from exc
    if not np.isfinite(result):
        raise ValueError(f"{key} must be a finite number")
    return result


def _time(params: Mapping[str, Any], key: str) -> float:
    unit_key = f"{key}_units"
    try:
        return TimeQuantity(params[key], params[unit_key]).femtoseconds
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {key}/{unit_key}: {exc}") from exc


def _optional_dispersion_pair(
    params: Mapping[str, Any],
    key: Literal["gdd", "tod"],
) -> float:
    unit_key = f"{key}_units"
    has_value = key in params
    has_unit = unit_key in params
    if has_value != has_unit:
        missing = unit_key if has_value else key
        raise ValueError(
            f"{key}/{unit_key} must be supplied together; missing {missing}"
        )
    if not has_value:
        return 0.0
    try:
        if key == "gdd":
            return GroupDelayDispersion(
                params[key], params[unit_key]
            ).femtoseconds_squared
        return ThirdOrderDispersion(params[key], params[unit_key]).femtoseconds_cubed
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {key}/{unit_key}: {exc}") from exc


@dataclass(frozen=True, slots=True)
class GeneratedFieldParameters:
    """Validated generated pulse values in the internal canonical unit system."""

    time_grid: TimeGrid
    duration_fs: float
    t_center_fs: float
    carrier_angular_rad_per_fs: float
    carrier_cycles_per_fs: float
    amplitude_v_per_m: float
    envelope_kind: str
    phase_rad: float
    gdd_fs2: float
    tod_fs3: float
    modulation_kind: Literal["none", "sinusoidal"]
    modulation_depth: float | None
    modulation_delay_fs: float | None
    modulation_phase_rad: float
    modulation_mode: Literal["phase", "amplitude"] | None

    @classmethod
    def from_mapping(cls, params: Mapping[str, Any]) -> GeneratedFieldParameters:
        """Convert every public quantity exactly once and discard source units."""
        t_start_fs = _time(params, "t_start")
        t_end_fs = _time(params, "t_end")
        dt_fs = _time(params, "dt")
        duration_fs = _time(params, "duration")
        t_center_fs = _time(params, "t_center")
        try:
            time_grid = TimeGrid.from_bounds(t_start_fs, t_end_fs, dt_fs)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid generated-field time grid: {exc}") from exc
        if duration_fs <= 0.0:
            raise ValueError("duration must be positive")

        try:
            carrier = Frequency(
                params["carrier_frequency"],
                params["carrier_frequency_units"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "invalid carrier_frequency/carrier_frequency_units: " + str(exc)
            ) from exc
        try:
            amplitude = ElectricFieldAmplitude(
                params["amplitude"],
                params["amplitude_units"],
            ).volts_per_meter
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid amplitude/amplitude_units: " + str(exc)) from exc

        envelope_kind = params["envelope_kind"]
        get_generated_envelope(envelope_kind)
        phase_rad = _finite_scalar(params, "phase_rad")
        gdd_fs2 = _optional_dispersion_pair(params, "gdd")
        tod_fs3 = _optional_dispersion_pair(params, "tod")

        modulation_kind = params["modulation_kind"]
        if modulation_kind not in {"none", "sinusoidal"}:
            raise ValueError("modulation_kind must be one of: none, sinusoidal")
        supplied_modulation = {
            "modulation_depth",
            "modulation_delay",
            "modulation_delay_units",
            "modulation_phase_rad",
            "modulation_mode",
        } & params.keys()
        if modulation_kind == "none":
            if supplied_modulation:
                names = ", ".join(sorted(supplied_modulation))
                raise ValueError(
                    f"{names} are not applicable when modulation_kind=none"
                )
            return cls(
                time_grid=time_grid,
                duration_fs=duration_fs,
                t_center_fs=t_center_fs,
                carrier_angular_rad_per_fs=carrier.angular_rad_per_fs,
                carrier_cycles_per_fs=carrier.cycles_per_fs,
                amplitude_v_per_m=amplitude,
                envelope_kind=envelope_kind,
                phase_rad=phase_rad,
                gdd_fs2=gdd_fs2,
                tod_fs3=tod_fs3,
                modulation_kind="none",
                modulation_depth=None,
                modulation_delay_fs=None,
                modulation_phase_rad=0.0,
                modulation_mode=None,
            )

        required = {
            "modulation_depth",
            "modulation_delay",
            "modulation_delay_units",
            "modulation_mode",
        }
        missing = sorted(required - params.keys())
        if missing:
            raise ValueError(
                "Missing required sinusoidal modulation parameters: "
                + ", ".join(missing)
            )
        depth = _finite_scalar(params, "modulation_depth")
        try:
            delay_fs = TimeQuantity(
                params["modulation_delay"],
                params["modulation_delay_units"],
            ).femtoseconds
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "invalid modulation_delay/modulation_delay_units: " + str(exc)
            ) from exc
        modulation_mode = params["modulation_mode"]
        if modulation_mode not in {"phase", "amplitude"}:
            raise ValueError("modulation_mode must be one of: phase, amplitude")
        if modulation_mode == "amplitude" and not 0.0 <= depth <= 1.0:
            raise ValueError("amplitude modulation_depth must satisfy 0 <= depth <= 1")

        return cls(
            time_grid=time_grid,
            duration_fs=duration_fs,
            t_center_fs=t_center_fs,
            carrier_angular_rad_per_fs=carrier.angular_rad_per_fs,
            carrier_cycles_per_fs=carrier.cycles_per_fs,
            amplitude_v_per_m=amplitude,
            envelope_kind=envelope_kind,
            phase_rad=phase_rad,
            gdd_fs2=gdd_fs2,
            tod_fs3=tod_fs3,
            modulation_kind="sinusoidal",
            modulation_depth=depth,
            modulation_delay_fs=delay_fs,
            modulation_phase_rad=_finite_scalar(params, "modulation_phase_rad"),
            modulation_mode=modulation_mode,
        )


__all__ = ["GeneratedFieldParameters"]
