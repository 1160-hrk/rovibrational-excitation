"""Characterize legacy unit conversion and diagnostic fallback boundaries."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from rovibrational_excitation.core.units import UnitValidator
from rovibrational_excitation.core.units.converters import converter

_SUPPORTED_UNITS = {
    "frequency": {
        "rad/fs",
        "THz",
        "GHz",
        "MHz",
        "kHz",
        "Hz",
        "cm^-1",
        "cm-1",
        "wavenumber",
        "PHz",
        "rad/s",
        "rad/ps",
    },
    "energy": {
        "J",
        "eV",
        "meV",
        "keV",
        "Ry",
        "Ha",
        "rad/fs",
        "rad/ps",
        "PHz",
        "THz",
        "cm^-1",
        "cm-1",
        "wavenumber",
        "kJ/mol",
        "kcal/mol",
    },
    "dipole": {
        "C*m",
        "C·m",
        "Cm",
        "D",
        "Debye",
        "ea0",
        "e*a0",
        "atomic",
        "rad/fs/(V/m)",
        "rad*PHz/(V/m)",
    },
    "field": {
        "V/m",
        "V/nm",
        "V/Å",
        "V/A",
        "kV/m",
        "kV/cm",
        "MV/m",
        "MV/cm",
        "GV/m",
        "TV/m",
        "atomic",
        "W/cm^2",
        "W/cm2",
        "W/m^2",
        "W/m2",
        "TW/cm^2",
        "TW/cm2",
        "GW/cm^2",
        "GW/cm2",
        "MW/cm^2",
        "MW/cm2",
    },
    "time": {"fs", "ps", "ns", "μs", "us", "ms", "s", "atomic"},
    "gdd": {"fs^2", "ps^2", "ns^2", "μs^2", "us^2", "ms^2", "s^2"},
    "tod": {"fs^3", "ps^3", "ns^3", "μs^3", "us^3", "ms^3", "s^3"},
}


@pytest.mark.parametrize("quantity", sorted(_SUPPORTED_UNITS))
def test_supported_unit_spellings_are_explicitly_frozen(quantity):
    assert set(converter.get_supported_units(quantity)) == _SUPPORTED_UNITS[quantity]


_ROUND_TRIP_CASES: tuple[tuple[str, str, Callable[[Any, str, str], Any]], ...] = (
    ("frequency", "rad/fs", converter.convert_frequency),
    ("energy", "J", converter.convert_energy),
    ("dipole", "C*m", converter.convert_dipole_moment),
    ("time", "fs", converter.convert_time),
    ("gdd", "fs^2", converter.convert_gdd),
    ("tod", "fs^3", converter.convert_tod),
)


@pytest.mark.parametrize(("quantity", "canonical_unit", "convert"), _ROUND_TRIP_CASES)
def test_every_directly_convertible_unit_round_trips_without_mutating_input(
    quantity,
    canonical_unit,
    convert,
):
    original = np.array([0.125, 1.5, 7.0], dtype=np.float64)
    snapshot = original.copy()

    for unit in sorted(_SUPPORTED_UNITS[quantity]):
        canonical = convert(original, unit, canonical_unit)
        restored = convert(canonical, canonical_unit, unit)
        np.testing.assert_allclose(restored, original, rtol=2e-15, atol=0.0)

    np.testing.assert_array_equal(original, snapshot)


def test_every_direct_field_unit_round_trips_without_mutating_input():
    intensity_units = {
        unit
        for unit in _SUPPORTED_UNITS["field"]
        if unit.startswith(("W/", "TW/", "GW/", "MW/"))
    }
    field_units = _SUPPORTED_UNITS["field"] - intensity_units
    original = np.array([0.125, 1.5, 7.0], dtype=np.float64)
    snapshot = original.copy()

    for unit in sorted(field_units):
        canonical = converter.convert_electric_field(original, unit, "V/m")
        restored = converter.convert_electric_field(canonical, "V/m", unit)
        np.testing.assert_allclose(restored, original, rtol=2e-15, atol=0.0)

    np.testing.assert_array_equal(original, snapshot)


def test_energy_and_angular_frequency_round_trip_across_hamiltonian_boundary():
    original = np.array([0.125, 1.5, 7.0], dtype=np.float64)
    frequency_units = _SUPPORTED_UNITS["frequency"]
    energy_units = _SUPPORTED_UNITS["energy"]

    for source_unit in sorted(frequency_units):
        for target_unit in sorted(energy_units):
            converted = converter.convert_hamiltonian(
                original, source_unit, target_unit
            )
            restored = converter.convert_hamiltonian(
                converted, target_unit, source_unit
            )
            np.testing.assert_allclose(restored, original, rtol=2e-15, atol=0.0)


@pytest.mark.parametrize(
    ("intensity", "unit"),
    [
        (1.0e12, "W/cm^2"),
        (1.0e12, "W/cm2"),
        (1.0e16, "W/m^2"),
        (1.0e16, "W/m2"),
        (1.0, "TW/cm^2"),
        (1.0, "TW/cm2"),
        (1.0e3, "GW/cm^2"),
        (1.0e3, "GW/cm2"),
        (1.0e6, "MW/cm^2"),
        (1.0e6, "MW/cm2"),
    ],
)
def test_all_intensity_aliases_use_the_same_current_peak_field_convention(
    intensity,
    unit,
):
    reference = converter.convert_electric_field(1.0e12, "W/cm^2", "V/m")
    assert converter.convert_electric_field(intensity, unit, "V/m") == pytest.approx(
        reference
    )


def test_propagation_validator_rejects_noncanonical_expected_units():
    class Hamiltonian:
        def get_matrix(self, units):
            assert units == "J"
            return np.diag([0.0, 1.0e-20])

    with pytest.raises(ValueError, match="expected_H0_units must be 'J'"):
        UnitValidator().validate_propagation_units(
            Hamiltonian(),
            object(),
            object(),
            expected_H0_units="not-a-unit",
        )


def test_propagation_validator_propagates_internal_accessor_error():
    class Hamiltonian:
        def get_matrix(self, _units):
            return np.diag([0.0, 1.0e-20])

    class BrokenDipole:
        def get_mu_x_SI(self):
            raise RuntimeError("broken SI accessor")

    with pytest.raises(RuntimeError, match="broken SI accessor"):
        UnitValidator().validate_propagation_units(
            Hamiltonian(),
            BrokenDipole(),
            object(),
        )


def test_propagation_validator_rejects_raw_dipole_attribute_fallback():
    raw = np.array([[0.0, 2.0], [2.0, 0.0]])

    class RawDipole:
        mu_x = raw

    class Hamiltonian:
        def get_matrix(self, _units):
            return np.diag([0.0, 1.0e-20])

    with pytest.raises(TypeError, match="get_mu_x_SI"):
        UnitValidator().validate_propagation_units(Hamiltonian(), RawDipole(), object())


def test_propagation_validator_accepts_canonical_structural_boundary():
    class Hamiltonian:
        def get_matrix(self, units):
            assert units == "J"
            return np.diag([0.0, 1.0e-20])

    class Dipole:
        def get_mu_x_SI(self):
            return np.zeros((2, 2))

        def get_mu_y_SI(self):
            return np.zeros((2, 2))

    class Field:
        dt = 0.5

        def get_time_SI(self):
            return np.array([0.0, 0.5, 1.0])

        def get_Efield_SI(self):
            return np.zeros((3, 2))

    assert (
        UnitValidator().validate_propagation_units(Hamiltonian(), Dipole(), Field())
        is None
    )


def test_propagation_validator_rejects_shape_mismatch():
    class Hamiltonian:
        def get_matrix(self, _units):
            return np.zeros((3, 3))

    class Dipole:
        def get_mu_x_SI(self):
            return np.zeros((2, 2))

        def get_mu_y_SI(self):
            return np.zeros((3, 3))

    with pytest.raises(ValueError, match=r"mu_x.*must have shape"):
        UnitValidator().validate_propagation_units(Hamiltonian(), Dipole(), object())
