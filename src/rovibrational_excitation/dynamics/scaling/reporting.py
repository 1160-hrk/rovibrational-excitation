"""Neutral diagnostics for an already-derived scaling."""

from __future__ import annotations

from typing import Any

from .scales import NondimensionalizationScales
from .utils import _EV_TO_J


def analyze_regime(scales: NondimensionalizationScales) -> dict[str, Any]:
    """Report scale provenance without model-independent strength thresholds."""
    interaction = scales.interaction_energy
    physical_ratio = scales.physical_coupling_ratio
    if interaction == 0:
        regime = "field_free"
        description = "The interaction generator is inactive."
    elif physical_ratio is None:
        regime = "gapless_driven"
        description = "Driven system with no non-zero field-free spectral span."
    else:
        regime = "unclassified"
        description = (
            "No weak/strong label is assigned without a model-specific threshold."
        )

    return {
        "regime": regime,
        "numerical_coupling_coefficient": scales.lambda_coupling,
        "physical_coupling_ratio": physical_ratio,
        "description": description,
        "energy_scale_eV": scales.E0 / _EV_TO_J,
        "time_scale_fs": scales.t0 * 1e15,
        "reference_energy": {
            "value_J": scales.reference_energy.value,
            "source": scales.reference_energy.source,
            "method": scales.reference_energy.method,
        },
    }
