"""Characterize legacy unit conversion and diagnostic fallback boundaries."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from rovibrational_excitation.core.units import ParameterProcessor, UnitValidator
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


def test_parameter_processor_currently_converts_values_but_keeps_input_labels():
    params = {
        "duration": 2.0,
        "duration_units": "ps",
        "amplitude": 1.0,
        "amplitude_units": "MV/cm",
        "mu0_Cm": 0.3,
        "mu0_Cm_units": "D",
    }

    converted = ParameterProcessor().auto_convert_parameters(params)

    assert converted["duration"] == 2.0e3
    assert converted["duration_units"] == "ps"
    assert converted["amplitude"] == 1.0e8
    assert converted["amplitude_units"] == "MV/cm"
    assert converted["mu0_Cm"] == pytest.approx(0.3 * 3.33564e-30)
    assert converted["mu0_Cm_units"] == "D"
    assert params["duration"] == 2.0
    assert params["amplitude"] == 1.0
    assert params["mu0_Cm"] == 0.3


def test_parameter_processor_strict_false_keeps_invalid_value_and_reports_warning():
    params = {"amplitude": 5.0, "amplitude_units": "not-a-unit"}

    converted = ParameterProcessor().auto_convert_parameters(params, strict=False)

    assert converted["amplitude"] == 5.0
    assert converted["amplitude_units"] == "not-a-unit"
    assert converted["_conversion_warnings"] == [
        "Unknown electric field/intensity unit: not-a-unit"
    ]


def test_parameter_processor_strict_true_rejects_invalid_known_parameter_unit():
    params = {"amplitude": 5.0, "amplitude_units": "not-a-unit"}

    with pytest.raises(ValueError, match="Failed to convert amplitude"):
        ParameterProcessor().auto_convert_parameters(params, strict=True)


def test_validator_unknown_context_currently_falls_back_and_warns():
    valid, warnings = UnitValidator().validate_frequency(
        1.0,
        "rad/fs",
        context="misspelled-context",
    )

    assert valid is False
    assert warnings == ["Unknown context 'misspelled-context', using 'molecular'"]


def test_propagation_validator_currently_uses_1000_fs_for_unknown_h0_units():
    class Hamiltonian:
        def get_matrix(self, units):
            assert units == "J"
            return np.diag([0.0, 1.0e-20])

    class RawDipole:
        mu_x = np.array([[0.0, 1.0e-30], [1.0e-30, 0.0]])
        mu_y = np.zeros((2, 2))

    class Field:
        dt = 300.0
        Efield = np.full((3, 2), 1.0e8)

    warnings = UnitValidator().validate_propagation_units(
        Hamiltonian(),
        RawDipole(),
        Field(),
        expected_H0_units="not-a-unit",
    )

    assert "Unknown energy unit: not-a-unit" in warnings
    assert any("1000.000 fs" in warning for warning in warnings)


def test_propagation_validator_currently_downgrades_internal_error_to_warning():
    class Hamiltonian:
        def get_matrix(self, _units):
            return np.diag([0.0, 1.0e-20])

    class BrokenDipole:
        def get_mu_x_SI(self):
            raise RuntimeError("broken SI accessor")

    warnings = UnitValidator().validate_propagation_units(
        Hamiltonian(),
        BrokenDipole(),
        object(),
    )

    assert warnings == ["単位検証中にエラーが発生しました: broken SI accessor"]


def test_propagation_validator_currently_falls_back_to_raw_dipole_attribute():
    raw = np.array([[0.0, 2.0], [2.0, 0.0]])

    class RawDipole:
        mu_x = raw

    extracted = UnitValidator()._get_dipole_component(RawDipole(), "x")

    assert extracted is raw
