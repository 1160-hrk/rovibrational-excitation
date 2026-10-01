"""Prepare exact sampled fields for generated normal-simulation cases."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from ..fields import SampledField
from ..io import deserialize_polarization as _deserialize_pol
from .generated import GeneratedFieldParameters


def _generated_sampled_field(
    params: Mapping[str, Any],
    *,
    generated_parameters: GeneratedFieldParameters,
    use_m_average: bool,
    expects_cartesian: bool,
) -> SampledField:
    """Generate the legacy waveform, then freeze its exact sampled values."""
    from rovibrational_excitation.fields import (
        CartesianField,
        ElectricField,
        ScalarField,
    )

    polarization = _deserialize_pol(params.get("polarization", [1.0, 0.0]))
    if use_m_average:
        from .m_average import canonicalize_fixed_linear_polarization

        polarization = canonicalize_fixed_linear_polarization(polarization)

    from rovibrational_excitation.fields.envelopes import get_generated_envelope

    time_grid = generated_parameters.time_grid
    generated = ElectricField.from_time_grid(time_grid)
    generated.add_dispersed_Efield(
        envelope_func=get_generated_envelope(generated_parameters.envelope_kind),
        duration=generated_parameters.duration_fs,
        t_center=generated_parameters.t_center_fs,
        carrier_freq=generated_parameters.carrier_angular_rad_per_fs,
        amplitude=generated_parameters.amplitude_v_per_m,
        polarization=polarization,
        phase_rad=generated_parameters.phase_rad,
        gdd=generated_parameters.gdd_fs2,
        tod=generated_parameters.tod_fs3,
        duration_units="fs",
        t_center_units="fs",
        carrier_freq_units="rad/fs",
        amplitude_units="V/m",
        gdd_units="fs^2",
        tod_units="fs^3",
    )
    if generated_parameters.modulation_kind == "sinusoidal":
        generated.apply_sinusoidal_mod(
            center_freq=generated_parameters.carrier_cycles_per_fs,
            modulation_depth=cast(float, generated_parameters.modulation_depth),
            delay_fs=cast(float, generated_parameters.modulation_delay_fs),
            phase_rad=generated_parameters.modulation_phase_rad,
            mode=cast(str, generated_parameters.modulation_mode),
        )

    if expects_cartesian:
        components = generated.get_Efield()
        return CartesianField(
            time_grid,
            components[:, 0],
            components[:, 1],
            scalar_samples_v_per_m=generated.get_scalar_field(),
            jones_polarization=generated.get_pol(),
        )
    return ScalarField(time_grid, generated.get_scalar_field())
