"""Independent analytic references for the spectroscopy boundary."""

from __future__ import annotations

import numpy as np
import pytest

from rovibrational_excitation.core.units.constants import CONSTANTS
from rovibrational_excitation.models.two_level import (
    TwoLevelBasis,
    TwoLevelDipoleMatrix,
)
from rovibrational_excitation.spectroscopy import (
    AbsorbanceCalculator,
    ExperimentalConditions,
)

ENERGY_GAP_J = 1.0e-20
DIPOLE_C_M = 1.0e-30


def _conditions(*, molecular_mass: float | None = None) -> ExperimentalConditions:
    if molecular_mass is None:
        molecular_mass = 44.0e-3 / CONSTANTS.AVOGADRO
    return ExperimentalConditions(
        temperature=300.0,
        temperature_units="K",
        pressure=3.0e4,
        pressure_units="Pa",
        optical_length=1.0e-3,
        optical_length_units="m",
        coherence_time=500.0,
        coherence_time_units="ps",
        molecular_mass=molecular_mass,
        molecular_mass_units="kg",
    )


def _calculator(
    conditions: ExperimentalConditions | None = None,
) -> AbsorbanceCalculator:
    basis = TwoLevelBasis(
        energy_gap=ENERGY_GAP_J,
        input_units="J",
        output_units="J",
    )
    return AbsorbanceCalculator(
        basis,
        basis.generate_H0(),
        TwoLevelDipoleMatrix(basis, mu0=DIPOLE_C_M),
        _conditions() if conditions is None else conditions,
        phase_matching="unfiltered",
        axes="x",
        pol_int=np.array([1.0]),
        pol_det=np.array([1.0]),
    )


def _wavenumber_grid() -> np.ndarray:
    center = ENERGY_GAP_J / (2.0 * np.pi * CONSTANTS.HBAR * CONSTANTS.C * 100.0)
    return np.linspace(center - 20.0, center + 20.0, 1001)


def _absorbance_from_response(
    response_per_molecule: np.ndarray,
    angular_frequency: np.ndarray,
    conditions: ExperimentalConditions,
) -> np.ndarray:
    number_density = conditions.pressure_pa / (
        CONSTANTS.BOLTZMANN * conditions.temperature_k
    )
    susceptibility = number_density * response_per_molecule
    refractive_index = np.sqrt(1.0 + susceptibility / CONSTANTS.EPSILON0)
    return (
        2.0
        * conditions.optical_length_m
        * angular_frequency
        / CONSTANTS.C
        * refractive_index.imag
        * np.log10(np.e)
        * 1000.0
    )


def _direct_gaussian_kernel(sigma_pixels: float) -> np.ndarray:
    radius = int(4.0 * sigma_pixels + 0.5)
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma_pixels) ** 2)
    return kernel / np.sum(kernel)


@pytest.mark.parametrize(
    ("method", "options", "relative_tolerance"),
    [
        ("loop", {}, 5.0e-14),
        ("matrix", {}, 5.0e-14),
        ("2d", {}, 5.0e-14),
        ("chunked", {"chunk_size": 137}, 2.0e-12),
    ],
)
def test_two_level_absorbance_matches_direct_boltzmann_lorentzian(
    method: str,
    options: dict[str, int],
    relative_tolerance: float,
) -> None:
    calculator = _calculator()
    conditions = calculator.conditions
    wavenumber = _wavenumber_grid()
    omega = 2.0 * np.pi * CONSTANTS.C * 100.0 * wavenumber
    omega_0 = ENERGY_GAP_J / CONSTANTS.HBAR
    gamma = 1.0 / (conditions.coherence_time_ps * 1.0e-12)

    energies = np.array([0.0, ENERGY_GAP_J])
    boltzmann_weights = np.exp(
        -(energies - energies.min()) / (CONSTANTS.BOLTZMANN * conditions.temperature_k)
    )
    populations = boltzmann_weights / np.sum(boltzmann_weights)
    population_difference = populations[0] - populations[1]
    rho = np.diag(populations).astype(np.complex128)

    response = (
        DIPOLE_C_M**2
        * population_difference
        / CONSTANTS.HBAR
        * (1.0 / (omega - omega_0 - 1j * gamma) - 1.0 / (omega + omega_0 - 1j * gamma))
    )
    expected = _absorbance_from_response(response, omega, conditions)

    actual = calculator.calculate(
        rho,
        wavenumber,
        method=method,
        wavenumber_units="cm^-1",
        **options,
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=relative_tolerance,
        atol=5.0e-11,
    )
    assert np.sum(populations) == 1.0
    assert populations[0] > populations[1]


def test_single_coherence_radiation_and_pfid_match_direct_transform() -> None:
    calculator = _calculator()
    conditions = calculator.conditions
    wavenumber = _wavenumber_grid()
    omega = 2.0 * np.pi * CONSTANTS.C * 100.0 * wavenumber
    omega_0 = ENERGY_GAP_J / CONSTANTS.HBAR
    gamma = 1.0 / (conditions.coherence_time_ps * 1.0e-12)
    coherence = 0.3 + 0.2j

    rho = np.zeros((2, 2), dtype=np.complex128)
    rho[1, 0] = coherence
    response = 1j * DIPOLE_C_M * coherence / (omega - omega_0 - 1j * gamma)
    expected = _absorbance_from_response(response, omega, conditions)

    radiation = calculator.calculate_radiation_spectrum(
        rho,
        wavenumber,
        wavenumber_units="cm^-1",
    )
    pfid = calculator.calculate_pfid_spectrum(
        rho,
        wavenumber,
        wavenumber_units="cm^-1",
    )

    np.testing.assert_allclose(radiation, expected, rtol=5.0e-14, atol=2.0e-15)
    np.testing.assert_array_equal(pfid, radiation)


def test_response_to_absorbance_has_documented_weak_susceptibility_limit() -> None:
    calculator = _calculator()
    conditions = calculator.conditions
    omega = np.array([0.8e14, 1.0e14, 1.2e14])
    dimensionless_susceptibility = 1j * np.array([1.0e-10, 2.0e-10, 3.0e-10])
    number_density = conditions.pressure_pa / (
        CONSTANTS.BOLTZMANN * conditions.temperature_k
    )
    response = dimensionless_susceptibility * CONSTANTS.EPSILON0 / number_density
    linear_limit = (
        conditions.optical_length_m
        * omega
        / CONSTANTS.C
        * dimensionless_susceptibility.imag
        * np.log10(np.e)
        * 1000.0
    )

    actual = calculator._response_to_absorbance(omega, response)

    np.testing.assert_allclose(actual, linear_limit, rtol=2.0e-10, atol=0.0)
    np.testing.assert_array_equal(
        calculator._response_to_absorbance(omega, np.zeros_like(response)),
        np.zeros_like(omega),
    )


def test_doppler_matches_direct_normalized_gaussian_convolution() -> None:
    temperature_k = 300.0
    omega_0 = ENERGY_GAP_J / CONSTANTS.HBAR
    wavenumber = _wavenumber_grid()
    omega = 2.0 * np.pi * CONSTANTS.C * 100.0 * wavenumber
    spacing = float(omega[1] - omega[0])
    target_sigma_pixels = 3.0
    molecular_mass = (
        CONSTANTS.BOLTZMANN
        * temperature_k
        / CONSTANTS.C**2
        * (omega_0 / (target_sigma_pixels * spacing)) ** 2
    )
    conditions = _conditions(molecular_mass=molecular_mass)
    calculator = _calculator(conditions)
    gamma = 1.0 / (conditions.coherence_time_ps * 1.0e-12)
    lorentzian_response = 1.0 / (omega - omega_0 - 1j * gamma)

    broadened = calculator._apply_doppler_broadening(
        omega,
        lorentzian_response,
        omega_0,
    )
    kernel = _direct_gaussian_kernel(target_sigma_pixels)
    direct_voigt = np.convolve(lorentzian_response, kernel, mode="same")
    radius = kernel.size // 2

    np.testing.assert_allclose(
        broadened[radius:-radius],
        direct_voigt[radius:-radius],
        rtol=5.0e-14,
        atol=2.0e-25,
    )
    np.testing.assert_allclose(np.sum(kernel), 1.0, rtol=0.0, atol=2.0e-16)
    np.testing.assert_allclose(
        np.sum(broadened.imag),
        np.sum(lorentzian_response.imag),
        rtol=5.0e-15,
        atol=0.0,
    )
    np.testing.assert_array_equal(
        calculator._apply_doppler_broadening(omega, lorentzian_response, 0.0),
        lorentzian_response,
    )


def test_gaussian_device_function_matches_direct_normalized_kernel() -> None:
    calculator = _calculator()
    wavenumber = np.linspace(-5.0, 5.0, 1001)
    impulse = np.zeros_like(wavenumber)
    impulse[wavenumber.size // 2] = 1.0
    resolution = 0.5
    spacing = float(wavenumber[1] - wavenumber[0])
    sigma_pixels = resolution / (2.0 * np.sqrt(2.0 * np.log(2.0))) / spacing
    expected = np.convolve(
        impulse,
        _direct_gaussian_kernel(sigma_pixels),
        mode="same",
    )

    actual = calculator.apply_device_function(
        impulse,
        wavenumber,
        resolution,
        wavenumber_units="cm^-1",
        resolution_units="cm^-1",
        function_type="gaussian",
    )

    np.testing.assert_allclose(actual, expected, rtol=5.0e-14, atol=2.0e-17)
    np.testing.assert_allclose(np.sum(actual), 1.0, rtol=0.0, atol=2.0e-16)


@pytest.mark.parametrize("function_type", ["sinc", "sinc2"])
def test_sinc_device_functions_match_direct_normalized_convolution(
    function_type: str,
) -> None:
    calculator = _calculator()
    wavenumber = np.linspace(-5.0, 5.0, 1001)
    impulse = np.zeros_like(wavenumber)
    impulse[wavenumber.size // 2] = 1.0
    resolution = 0.5
    spacing = float(wavenumber[1] - wavenumber[0])
    offsets = np.arange(-wavenumber.size // 2, wavenumber.size // 2) * spacing
    kernel = np.sinc(2.0 * offsets / resolution)
    if function_type == "sinc2":
        kernel = kernel**2
    kernel /= np.sum(kernel)
    expected = np.convolve(impulse, kernel, mode="same")

    actual = calculator.apply_device_function(
        impulse,
        wavenumber,
        resolution,
        wavenumber_units="cm^-1",
        resolution_units="cm^-1",
        function_type=function_type,
    )

    np.testing.assert_allclose(actual, expected, rtol=5.0e-14, atol=2.0e-17)
    np.testing.assert_allclose(np.sum(actual), 1.0, rtol=0.0, atol=3.0e-16)
