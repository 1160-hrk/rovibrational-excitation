#!/usr/bin/env python3
"""Spectroscopy observables for rovibrational simulations.

Ordinary transmission absorption is constructed with
``AbsorbanceCalculator.standard_absorption``. Scalar models obtain their
storage axis from ``SystemModel``; Cartesian models require a
``CartesianProjection``. ``calculate`` returns the established scalar mOD
spectrum.

Analyzer-resolved measurements use ``CartesianAnalyzerProjection`` with
``AbsorbanceCalculator.analyzer_complex_response``.
``calculate_complex_response`` returns an immutable ``ComplexResponseSpectrum``
before scalar mOD conversion. Analyzer intensity and analyzer absorbance are
not implemented because they require an explicit reference measurement.
"""

from .absorbance_calculator import (
    AbsorbanceCalculator,
    create_calculator_from_params,
)
from .conditions import ExperimentalConditions
from .projection import CartesianAnalyzerProjection, CartesianProjection
from .report import SpectroscopyCalculationReport
from .result import ComplexResponseSpectrum

# Define what gets imported with "from spectroscopy import *"
__all__ = [
    "AbsorbanceCalculator",
    "CartesianAnalyzerProjection",
    "CartesianProjection",
    "ComplexResponseSpectrum",
    "ExperimentalConditions",
    "SpectroscopyCalculationReport",
    "create_calculator_from_params",
]


# Version information
__version__ = "1.0.0"
__author__ = "Rovibrational Excitation Team"
__email__ = "contact@example.com"

# Module-level documentation
if __doc__ is None:
    __doc__ = ""

__doc__ += f"""

Available Components
--------------------
AbsorbanceCalculator : {AbsorbanceCalculator.__doc__.split(".")[0] if AbsorbanceCalculator.__doc__ else "Main calculator class"}
ExperimentalConditions : {ExperimentalConditions.__doc__.split(".")[0] if ExperimentalConditions.__doc__ else "Experimental parameters dataclass"}

Optional Components
-------------------
"""


__doc__ += f"""

Module Version: {__version__}
"""
