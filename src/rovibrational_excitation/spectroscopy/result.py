"""Typed spectroscopy result values."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rovibrational_excitation.spectroscopy.report import SpectroscopyCalculationReport


@dataclass(frozen=True, slots=True, eq=False)
class ComplexResponseSpectrum:
    """Immutable projected per-molecule response on a cm^-1 grid."""

    wavenumber_cm_inverse: np.ndarray
    molecular_response_c2_m2_per_j: np.ndarray
    calculation_report: SpectroscopyCalculationReport

    def __post_init__(self) -> None:
        wavenumber = np.asarray(self.wavenumber_cm_inverse, dtype=np.float64)
        molecular_response = np.asarray(
            self.molecular_response_c2_m2_per_j,
            dtype=np.complex128,
        )
        if wavenumber.ndim != 1:
            raise ValueError("wavenumber_cm_inverse must be one-dimensional")
        if molecular_response.shape != wavenumber.shape:
            raise ValueError(
                "molecular_response_c2_m2_per_j must match the wavenumber shape"
            )
        if not isinstance(self.calculation_report, SpectroscopyCalculationReport):
            raise TypeError(
                "calculation_report must be a SpectroscopyCalculationReport"
            )

        wavenumber_copy = np.array(wavenumber, dtype=np.float64, copy=True)
        response_copy = np.array(molecular_response, dtype=np.complex128, copy=True)
        wavenumber_copy.setflags(write=False)
        response_copy.setflags(write=False)
        object.__setattr__(self, "wavenumber_cm_inverse", wavenumber_copy)
        object.__setattr__(
            self,
            "molecular_response_c2_m2_per_j",
            response_copy,
        )


__all__ = ["ComplexResponseSpectrum"]
