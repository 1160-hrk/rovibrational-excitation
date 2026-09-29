"""Grid validation, line broadening, and instrument response kernels."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import ndimage

from rovibrational_excitation.core.units.constants import CONSTANTS
from rovibrational_excitation.spectroscopy.conditions import require_exact_units


def uniform_grid_spacing(grid: np.ndarray, *, name: str) -> float:
    """Return the absolute spacing of a finite, monotonic uniform grid."""
    values = np.asarray(grid, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError(f"{name} must contain at least two points")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    differences = np.diff(values)
    if np.any(differences == 0.0) or not (
        np.all(differences > 0.0) or np.all(differences < 0.0)
    ):
        raise ValueError(f"{name} must be strictly monotonic")
    reference = differences[0]
    tolerance = np.finfo(float).eps * max(1.0, np.max(np.abs(values))) * 16.0
    if not np.allclose(differences, reference, rtol=1.0e-12, atol=tolerance):
        raise ValueError(f"{name} must be uniformly spaced")
    return abs(float(reference))


def filter_complex_gaussian(
    response: np.ndarray,
    sigma_pixels: float,
) -> np.ndarray:
    """Apply the existing real/imaginary reflect-mode Gaussian filter."""
    response_real = ndimage.gaussian_filter1d(
        response.real,
        sigma_pixels,
        mode="reflect",
    )
    response_imag = ndimage.gaussian_filter1d(
        response.imag,
        sigma_pixels,
        mode="reflect",
    )
    return np.asarray(response_real + 1j * response_imag)


def apply_doppler_broadening(
    omega: np.ndarray,
    response: np.ndarray,
    omega0: float,
    *,
    temperature_k: float,
    molecular_mass_kg: float,
) -> np.ndarray:
    """Apply transition-specific Doppler broadening on the actual grid."""
    if omega0 == 0.0:
        return response
    spacing = uniform_grid_spacing(omega, name="angular-frequency grid")
    sigma_doppler = abs(omega0) * np.sqrt(
        CONSTANTS.BOLTZMANN * temperature_k / (molecular_mass_kg * CONSTANTS.C**2)
    )
    return filter_complex_gaussian(
        response,
        sigma_doppler / spacing,
    )


def apply_device_function(
    spectrum: np.ndarray,
    wavenumber: np.ndarray,
    resolution: float,
    *,
    wavenumber_units: str,
    resolution_units: str,
    function_type: Literal["sinc", "sinc2", "gaussian"] = "sinc2",
) -> np.ndarray:
    """Apply the selected normalized instrument response on a cm^-1 grid."""
    require_exact_units(
        wavenumber_units,
        name="wavenumber_units",
        expected="cm^-1",
    )
    require_exact_units(
        resolution_units,
        name="resolution_units",
        expected="cm^-1",
    )
    if not np.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("resolution must be finite and positive")
    if function_type not in {"sinc", "sinc2", "gaussian"}:
        raise ValueError(f"unknown device function: {function_type}")

    dw = uniform_grid_spacing(wavenumber, name="wavenumber")

    if function_type == "gaussian":
        sigma_pixels = resolution / (2 * np.sqrt(2 * np.log(2))) / dw
        return np.asarray(
            ndimage.gaussian_filter1d(spectrum, sigma_pixels, mode="reflect")
        )

    n = len(wavenumber)
    x_device = np.arange(-n // 2, n // 2) * dw

    if function_type == "sinc":
        device_func = np.sinc(2 * x_device / resolution)
    else:
        device_func = np.sinc(2 * x_device / resolution) ** 2

    device_func /= np.sum(device_func)
    return np.convolve(spectrum, device_func, mode="same")
