"""Strict Krotov initial-field inputs in canonical internal units."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from rovibrational_excitation.core.time import TimeGrid
from rovibrational_excitation.core.units import (
    ElectricFieldAmplitude,
    Frequency,
    GroupDelayDispersion,
    ThirdOrderDispersion,
    TimeQuantity,
    converter,
)
from rovibrational_excitation.fields import ElectricField, gaussian_fwhm
from rovibrational_excitation.fields.sampled import CartesianField

_LEGACY_KEYS = {
    "amplitude_initial",
    "carrier_freq_initial",
    "const_polarisation",
    "duration_initial",
    "efield_initial",
    "gdd_initial",
    "pol_initial",
    "t_center_initial",
    "tod_initial",
    "unit_carrier_freq",
}
_GENERATED_REQUIRED = {
    "initial_amplitude",
    "initial_amplitude_units",
    "initial_carrier_frequency",
    "initial_carrier_frequency_units",
    "initial_center",
    "initial_center_units",
    "initial_duration",
    "initial_duration_units",
    "initial_polarization",
}
_GENERATED_OPTIONAL = {
    "initial_gdd",
    "initial_gdd_units",
    "initial_tod",
    "initial_tod_units",
}
_SAMPLED_KEYS = {"initial_field_samples", "initial_field_units"}
_SUPPORTED_INITIAL_KEYS = {
    "initial_field_kind",
    *_GENERATED_REQUIRED,
    *_GENERATED_OPTIONAL,
    *_SAMPLED_KEYS,
}


def _quantity_error(label: str, exc: TypeError | ValueError) -> ValueError:
    return ValueError(f"invalid {label}: {exc}")


def _time(params: Mapping[str, Any], key: str) -> float:
    unit_key = f"{key}_units"
    try:
        return TimeQuantity(params[key], params[unit_key]).femtoseconds
    except (TypeError, ValueError) as exc:
        raise _quantity_error(f"{key}/{unit_key}", exc) from exc


def _optional_dispersion(
    params: Mapping[str, Any], key: Literal["initial_gdd", "initial_tod"]
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
        if key == "initial_gdd":
            return GroupDelayDispersion(
                params[key], params[unit_key]
            ).femtoseconds_squared
        return ThirdOrderDispersion(params[key], params[unit_key]).femtoseconds_cubed
    except (TypeError, ValueError) as exc:
        raise _quantity_error(f"{key}/{unit_key}", exc) from exc


def _polarization(value: Any) -> tuple[complex, complex]:
    raw = np.asarray(value)
    if np.issubdtype(raw.dtype, np.bool_):
        raise ValueError("initial_polarization must contain complex-valued numbers")
    try:
        polarization = np.array(raw, dtype=np.complex128, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "initial_polarization must contain complex-valued numbers"
        ) from exc
    if polarization.shape != (2,):
        raise ValueError("initial_polarization must have shape (2,)")
    norm = np.linalg.norm(polarization)
    if not np.all(np.isfinite(polarization)) or not np.isfinite(norm) or norm == 0:
        raise ValueError("initial_polarization must be finite and non-zero")
    return complex(polarization[0]), complex(polarization[1])


@dataclass(frozen=True, slots=True)
class KrotovGeneratedInitialField:
    """A generated Gaussian seed converted once to canonical units."""

    duration_fs: float
    center_fs: float
    carrier_cycles_per_fs: float
    amplitude_v_per_m: float
    polarization: tuple[complex, complex]
    gdd_fs2: float
    tod_fs3: float

    @classmethod
    def from_mapping(cls, params: Mapping[str, Any]) -> KrotovGeneratedInitialField:
        missing = sorted(_GENERATED_REQUIRED - params.keys())
        if missing:
            raise ValueError(
                "missing required generated Krotov initial-field parameters: "
                + ", ".join(missing)
            )
        inapplicable = sorted(_SAMPLED_KEYS & params.keys())
        if inapplicable:
            raise ValueError(
                "sampled initial-field parameters are not applicable when "
                "initial_field_kind='generated': " + ", ".join(inapplicable)
            )

        duration_fs = _time(params, "initial_duration")
        if duration_fs <= 0.0:
            raise ValueError("initial_duration must be positive")
        center_fs = _time(params, "initial_center")
        try:
            carrier = Frequency(
                params["initial_carrier_frequency"],
                params["initial_carrier_frequency_units"],
            )
        except (TypeError, ValueError) as exc:
            raise _quantity_error(
                "initial_carrier_frequency/initial_carrier_frequency_units", exc
            ) from exc
        amplitude_units = params["initial_amplitude_units"]
        if not isinstance(amplitude_units, str) or amplitude_units not in (
            converter.get_supported_units("field_amplitude")
        ):
            raise ValueError(
                "initial_amplitude_units must be a supported direct "
                "electric-field amplitude unit"
            )
        try:
            amplitude = ElectricFieldAmplitude(
                params["initial_amplitude"], amplitude_units
            ).volts_per_meter
        except (TypeError, ValueError) as exc:
            raise _quantity_error(
                "initial_amplitude/initial_amplitude_units", exc
            ) from exc

        return cls(
            duration_fs=duration_fs,
            center_fs=center_fs,
            carrier_cycles_per_fs=carrier.cycles_per_fs,
            amplitude_v_per_m=amplitude,
            polarization=_polarization(params["initial_polarization"]),
            gdd_fs2=_optional_dispersion(params, "initial_gdd"),
            tod_fs3=_optional_dispersion(params, "initial_tod"),
        )

    def samples_on(self, time_grid: TimeGrid) -> NDArray[np.float64]:
        """Generate the accepted historical Krotov seed on ``time_grid``."""
        field = ElectricField.from_time_grid(time_grid)
        field.add_dispersed_Efield(
            envelope_func=gaussian_fwhm,
            duration=self.duration_fs,
            t_center=self.center_fs,
            carrier_freq=self.carrier_cycles_per_fs,
            duration_units="fs",
            t_center_units="fs",
            carrier_freq_units="PHz",
            amplitude=self.amplitude_v_per_m,
            amplitude_units="V/m",
            polarization=np.asarray(self.polarization, dtype=np.complex128),
            phase_rad=0.0,
            gdd=self.gdd_fs2,
            tod=self.tod_fs3,
            gdd_units="fs^2",
            tod_units="fs^3",
            const_polarisation=False,
        )
        return np.array(field.get_Efield_SI(), dtype=np.float64, copy=True)


@dataclass(frozen=True, slots=True, eq=False)
class KrotovSampledInitialField:
    """User-supplied real Cartesian seed stored canonically in V/m."""

    samples_v_per_m: NDArray[np.float64]

    @classmethod
    def from_mapping(cls, params: Mapping[str, Any]) -> KrotovSampledInitialField:
        missing = sorted(_SAMPLED_KEYS - params.keys())
        if missing:
            raise ValueError(
                "missing required sampled Krotov initial-field parameters: "
                + ", ".join(missing)
            )
        inapplicable = sorted(
            (_GENERATED_REQUIRED | _GENERATED_OPTIONAL) & params.keys()
        )
        if inapplicable:
            raise ValueError(
                "generated initial-field parameters are not applicable when "
                "initial_field_kind='sampled': " + ", ".join(inapplicable)
            )

        units = params["initial_field_units"]
        if not isinstance(units, str) or units not in converter.get_supported_units(
            "field_amplitude"
        ):
            raise ValueError(
                "initial_field_units must be a supported direct electric-field "
                "amplitude unit"
            )
        raw = np.asarray(params["initial_field_samples"])
        if np.iscomplexobj(raw) or np.issubdtype(raw.dtype, np.bool_):
            raise ValueError("initial_field_samples must be real-valued")
        try:
            values = np.array(raw, dtype=np.float64, copy=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("initial_field_samples must contain real numbers") from exc
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError(
                "initial_field_samples must have shape (n_field_points, 2)"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("initial_field_samples must contain only finite values")
        converted = np.asarray(
            converter.convert_electric_field(values, units, "V/m"),
            dtype=np.float64,
        )
        converted.setflags(write=False)
        return cls(samples_v_per_m=converted)

    def samples_on(self, time_grid: TimeGrid) -> NDArray[np.float64]:
        """Validate exact grid length and return a writable optimizer copy."""
        field = CartesianField(
            time_grid,
            self.samples_v_per_m[:, 0],
            self.samples_v_per_m[:, 1],
        )
        return np.array(field.components_v_per_m, dtype=np.float64, copy=True)


KrotovInitialField: TypeAlias = KrotovGeneratedInitialField | KrotovSampledInitialField


def parse_krotov_initial_field(params: Mapping[str, Any]) -> KrotovInitialField:
    """Resolve an explicit generated or sampled Krotov initial field."""
    legacy = sorted(_LEGACY_KEYS & params.keys())
    if legacy:
        raise ValueError(
            "legacy Krotov initial-field parameters are not supported: "
            + ", ".join(legacy)
            + "; use initial_field_kind and initial_* value/unit pairs"
        )
    unknown = sorted(
        key
        for key in params
        if key.startswith("initial_") and key not in _SUPPORTED_INITIAL_KEYS
    )
    if unknown:
        raise ValueError(
            "unsupported Krotov initial-field parameters: " + ", ".join(unknown)
        )
    if "initial_field_kind" not in params:
        raise ValueError("missing required Krotov parameter: initial_field_kind")
    kind = params["initial_field_kind"]
    if kind == "generated":
        return KrotovGeneratedInitialField.from_mapping(params)
    if kind == "sampled":
        return KrotovSampledInitialField.from_mapping(params)
    raise ValueError("initial_field_kind must be one of: generated, sampled")


__all__ = [
    "KrotovGeneratedInitialField",
    "KrotovInitialField",
    "KrotovSampledInitialField",
    "parse_krotov_initial_field",
]
