"""Typed observable reports for spectroscopy calculations."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SpectroscopyCalculationReport:
    """Observable record of the numerical spectroscopy path used."""

    requested_method: str
    executed_method: str
    estimated_2d_bytes: int
    memory_budget_bytes: int | None
    relative_threshold: float | None
    discarded_commutator_l2_fraction: float
    phase_matching: str
    discarded_density_l2_fraction: float
    device_function_applied: bool
