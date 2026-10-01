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

__all__ = [
    "AbsorbanceCalculator",
    "CartesianAnalyzerProjection",
    "CartesianProjection",
    "ComplexResponseSpectrum",
    "ExperimentalConditions",
    "SpectroscopyCalculationReport",
    "create_calculator_from_params",
]
