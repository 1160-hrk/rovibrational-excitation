"""Independent analytic references for the spectroscopy boundary."""

from __future__ import annotations

import numpy as np

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


def _conditions() -> ExperimentalConditions:
    return ExperimentalConditions(
        temperature=300.0,
        temperature_units="K",
        pressure=3.0e4,
        pressure_units="Pa",
        optical_length=1.0e-3,
        optical_length_units="m",
        coherence_time=500.0,
        coherence_time_units="ps",
        molecular_mass=44.0e-3 / CONSTANTS.AVOGADRO,
        molecular_mass_units="kg",
    )


def _calculator() -> AbsorbanceCalculator:
    basis = TwoLevelBasis(
        energy_gap=ENERGY_GAP_J,
        input_units="J",
        output_units="J",
    )
    return AbsorbanceCalculator(
        basis,
        basis.generate_H0(),
        TwoLevelDipoleMatrix(basis, mu0=DIPOLE_C_M),
        _conditions(),
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


def test_two_level_absorbance_matches_direct_boltzmann_lorentzian() -> None:
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
        method="loop",
        wavenumber_units="cm^-1",
    )

    np.testing.assert_allclose(actual, expected, rtol=5.0e-14, atol=5.0e-11)
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
