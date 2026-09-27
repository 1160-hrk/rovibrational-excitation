"""Independent DFT and circular-convolution spectral-constraint references."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest

from rovibrational_excitation.core.units.constants import CONSTANTS
from rovibrational_excitation.optimization.spectral_constraints import (
    build_alpha_mask,
    solve_update_in_frequency,
)


def _direct_alpha_mask(
    frequencies: np.ndarray,
    bands: tuple[tuple[float, float], ...],
    *,
    mode: Literal["pass", "stop"],
    combine: Literal["max", "sum"],
    widths_are_fwhm: bool,
    weights: tuple[float, ...] | None,
    scale: float,
) -> np.ndarray:
    """Construct the dimensionless penalty without a production helper."""
    profiles = []
    for center, width in bands:
        sigma = width / (2.0 * np.sqrt(2.0 * np.log(2.0))) if widths_are_fwhm else width
        profiles.append(np.exp(-0.5 * ((frequencies - center) / sigma) ** 2))

    stacked = np.asarray(profiles)
    if combine == "max":
        combined = np.max(stacked, axis=0)
    else:
        coefficients = np.ones(len(bands)) if weights is None else np.asarray(weights)
        combined = np.clip(coefficients @ stacked, 0.0, 1.0)
    return scale * ((1.0 - combined) if mode == "pass" else combined)


def _two_sided_penalty(alpha_rfft: np.ndarray, sample_count: int) -> np.ndarray:
    """Mirror a real-signal rFFT penalty onto the complete DFT grid."""
    result = np.empty(sample_count, dtype=np.float64)
    result[: alpha_rfft.size] = alpha_rfft
    positive_without_endpoints = (
        alpha_rfft[1:-1] if sample_count % 2 == 0 else alpha_rfft[1:]
    )
    result[alpha_rfft.size :] = positive_without_endpoints[::-1]
    return result


def _direct_dft_solution(source: np.ndarray, alpha_rfft: np.ndarray) -> np.ndarray:
    """Solve by an explicitly constructed two-sided DFT matrix."""
    sample_count = source.shape[0]
    indices = np.arange(sample_count)
    dft = np.exp(-2j * np.pi * np.outer(indices, indices) / sample_count)
    alpha_full = _two_sided_penalty(alpha_rfft, sample_count)
    spectrum = dft @ source
    return (
        np.real(dft.conj().T @ (spectrum / (1.0 + alpha_full[:, None]))) / sample_count
    )


def _direct_convolution_solution(
    source: np.ndarray, alpha_rfft: np.ndarray
) -> np.ndarray:
    """Solve (I + K)u=s using the explicit circular-convolution matrix K."""
    sample_count = source.shape[0]
    indices = np.arange(sample_count)
    inverse_dft = np.exp(2j * np.pi * np.outer(indices, indices) / sample_count)
    alpha_full = _two_sided_penalty(alpha_rfft, sample_count)
    kernel = np.real(inverse_dft @ alpha_full) / sample_count
    convolution = np.empty((sample_count, sample_count), dtype=np.float64)
    for row in range(sample_count):
        for column in range(sample_count):
            convolution[row, column] = kernel[(row - column) % sample_count]
    return np.linalg.solve(np.eye(sample_count) + convolution, source)


@pytest.mark.physics
@pytest.mark.parametrize(
    ("mode", "combine", "widths_are_fwhm", "weights"),
    [
        ("pass", "max", True, None),
        ("stop", "max", False, None),
        ("pass", "sum", True, (0.7, 0.6)),
        ("stop", "sum", False, (0.7, 0.6)),
    ],
)
def test_alpha_mask_matches_direct_gaussian_reference(
    mode: Literal["pass", "stop"],
    combine: Literal["max", "sum"],
    widths_are_fwhm: bool,
    weights: tuple[float, ...] | None,
) -> None:
    frequencies = np.linspace(0.0, 0.5, 17)
    bands = ((0.16, 0.11), (0.31, 0.17))
    scale = 3.5
    expected = _direct_alpha_mask(
        frequencies,
        bands,
        mode=mode,
        combine=combine,
        widths_are_fwhm=widths_are_fwhm,
        weights=weights,
        scale=scale,
    )

    actual = build_alpha_mask(
        frequencies,
        bands,
        units="PHz",
        mode=mode,
        combine=combine,
        fwhm=widths_are_fwhm,
        weights=weights,
        alpha_scale=scale,
    )

    np.testing.assert_allclose(actual, expected, rtol=3e-15, atol=3e-16)


@pytest.mark.physics
def test_active_wavenumber_band_matches_direct_cycles_per_fs_conversion() -> None:
    frequencies = np.linspace(0.04, 0.10, 31)
    cycles_per_fs_per_wavenumber = CONSTANTS.C * 1.0e-13
    center_cm = 2300.0
    fwhm_cm = 100.0
    expected = _direct_alpha_mask(
        frequencies,
        (
            (
                center_cm * cycles_per_fs_per_wavenumber,
                fwhm_cm * cycles_per_fs_per_wavenumber,
            ),
        ),
        mode="pass",
        combine="max",
        widths_are_fwhm=True,
        weights=None,
        scale=10.0,
    )

    actual = build_alpha_mask(
        frequencies,
        ((center_cm, fwhm_cm),),
        units="cm^-1",
        mode="pass",
        combine="max",
        fwhm=True,
        alpha_scale=10.0,
    )

    np.testing.assert_allclose(actual, expected, rtol=3e-15, atol=2e-15)


@pytest.mark.physics
@pytest.mark.parametrize("sample_count", [7, 8])
def test_spectral_update_matches_direct_dft_and_convolution_references(
    sample_count: int,
) -> None:
    time_index = np.arange(sample_count, dtype=np.float64)
    source = np.column_stack(
        (
            0.4 + np.cos(2.0 * np.pi * time_index / sample_count),
            np.sin(4.0 * np.pi * time_index / sample_count)
            + 0.3 * np.cos(6.0 * np.pi * time_index / sample_count),
        )
    )
    alpha = np.linspace(0.2, 2.3, sample_count // 2 + 1) ** 1.3

    expected_dft = _direct_dft_solution(source, alpha)
    expected_convolution = _direct_convolution_solution(source, alpha)
    actual = solve_update_in_frequency(source, alpha)

    np.testing.assert_allclose(
        expected_dft, expected_convolution, rtol=2e-14, atol=8e-16
    )
    np.testing.assert_allclose(actual, expected_dft, rtol=2e-14, atol=8e-16)


@pytest.mark.physics
def test_spectral_update_preserves_the_analytic_single_bin_attenuation() -> None:
    sample_count = 9
    selected_bin = 3
    time_index = np.arange(sample_count, dtype=np.float64)
    source = np.cos(2.0 * np.pi * selected_bin * time_index / sample_count)
    alpha = np.array([0.1, 0.3, 0.6, 4.0, 1.2])

    actual = solve_update_in_frequency(source, alpha)

    np.testing.assert_allclose(
        actual,
        source / (1.0 + alpha[selected_bin]),
        rtol=3e-15,
        atol=5e-16,
    )
