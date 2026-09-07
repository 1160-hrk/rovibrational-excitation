"""
Unit system management for rovibrational excitation calculations.

This package provides a centralized system for handling physical units,
conversions, and validation throughout the codebase.
"""

from .constants import PhysicalConstants
from .converters import UnitConverter, converter
from .frequency import Frequency
from .scalar_quantities import (
    DipoleMoment,
    ElectricFieldAmplitude,
    GroupDelayDispersion,
    LocalControlGain,
    ThirdOrderDispersion,
)
from .time_quantity import TimeQuantity
from .validators import UnitValidator, validator

__all__ = [
    "PhysicalConstants",
    "UnitConverter",
    "Frequency",
    "converter",
    "TimeQuantity",
    "DipoleMoment",
    "ElectricFieldAmplitude",
    "LocalControlGain",
    "GroupDelayDispersion",
    "ThirdOrderDispersion",
    "UnitValidator",
    "validator",
]
