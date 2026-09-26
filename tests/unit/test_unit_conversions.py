"""
Tests for unit conversion utilities.

This module tests the automatic unit conversion functionality
to ensure physical quantities are correctly converted to standard units.
"""

import numpy as np
import pytest

from rovibrational_excitation.core.units import (
    DipoleMoment,
    ElectricFieldAmplitude,
    Frequency,
    GroupDelayDispersion,
    KrotovPenalty,
    LocalControlGain,
    ThirdOrderDispersion,
    TimeQuantity,
)
from rovibrational_excitation.core.units.constants import CONSTANTS
from rovibrational_excitation.core.units.converters import converter

# 現在のAPIに合わせてマッピング
convert_frequency = converter.convert_frequency
convert_dipole_moment = converter.convert_dipole_moment
convert_electric_field = converter.convert_electric_field
convert_energy = converter.convert_energy
convert_time = converter.convert_time

# Physical constants for validation
_c = 2.99792458e10  # speed of light [cm/s]
_debye = 3.33564e-30  # Debye unit [C·m]
_e = 1.602176634e-19  # elementary charge [C]
_a0 = 5.29177210903e-11  # Bohr radius [m]


class TestFrequencyConversions:
    """Test frequency/angular frequency conversions."""

    def test_identity_conversion(self):
        """Test that rad/fs stays unchanged."""
        value = 100.0
        result = convert_frequency(value, "rad/fs")
        assert result == value

    def test_thz_conversion(self):
        """Test THz to rad/fs conversion."""
        value = 100.0  # THz
        result = convert_frequency(value, "THz")
        expected = value * 2 * np.pi * 1e-3  # THz → rad/fs
        assert np.isclose(result, expected)

    def test_wavenumber_conversion(self):
        """Test cm⁻¹ to rad/fs conversion."""
        value = 2350.0  # cm⁻¹ (CO2 ν3 mode)
        result = convert_frequency(value, "cm^-1")
        expected = value * 2 * np.pi * _c * 1e-15
        assert np.isclose(result, expected)

    def test_alternative_wavenumber_notation(self):
        """Test alternative cm-1 notation."""
        value = 1000.0
        result1 = convert_frequency(value, "cm^-1")
        result2 = convert_frequency(value, "cm-1")
        result3 = convert_frequency(value, "wavenumber")
        assert np.isclose(result1, result2)
        assert np.isclose(result1, result3)

    def test_array_input(self):
        """Test conversion with numpy array input."""
        values = np.array([100, 200, 300])  # THz
        results = convert_frequency(values, "THz")
        expected = values * 2 * np.pi * 1e-3
        assert np.allclose(results, expected)

    def test_invalid_unit(self):
        """Test error handling for invalid units."""
        with pytest.raises(ValueError, match="Unknown frequency unit"):
            convert_frequency(100, "invalid_unit")


class TestFrequencyValue:
    """Test the strict scalar frequency boundary."""

    @pytest.mark.parametrize(
        ("value", "unit"),
        [
            (0.1, "PHz"),
            (100.0, "THz"),
            (0.2 * np.pi, "rad/fs"),
            (1.0e14, "Hz"),
            (0.1 / (2.99792458e8 * 1.0e-13), "cm^-1"),
            (0.1 / (2.99792458e8 * 1.0e-13), "wavenumber"),
        ],
    )
    def test_equivalent_units_have_one_canonical_value(self, value, unit):
        frequency = Frequency(value, unit)

        assert frequency.angular_rad_per_fs == pytest.approx(0.2 * np.pi)
        assert frequency.cycles_per_fs == pytest.approx(0.1)

    @pytest.mark.parametrize("value", [True, np.bool_(False)])
    def test_boolean_value_is_rejected(self, value):
        with pytest.raises(TypeError, match="finite scalar"):
            Frequency(value, "PHz")

    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_nonfinite_value_is_rejected(self, value):
        with pytest.raises(ValueError, match="finite"):
            Frequency(value, "PHz")

    def test_unknown_unit_is_rejected(self):
        with pytest.raises(ValueError, match="invalid frequency unit"):
            Frequency(0.1, "cycles_per_second")


class TestDipoleConversions:
    """Test dipole moment conversions."""

    def test_identity_conversion(self):
        """Test that C·m stays unchanged."""
        value = 1e-30
        result = convert_dipole_moment(value, "C·m")
        assert result == value

    def test_debye_conversion(self):
        """Test Debye to C·m conversion."""
        value = 1.0  # Debye
        result = convert_dipole_moment(value, "D")
        expected = value * _debye
        assert np.isclose(result, expected)

        # Test alternative notation
        result2 = convert_dipole_moment(value, "Debye")
        assert np.isclose(result, result2)

    def test_atomic_units_conversion(self):
        """Test atomic units (ea0) to C·m conversion."""
        value = 1.0  # ea0
        result = convert_dipole_moment(value, "ea0")
        expected = value * _e * _a0
        assert np.isclose(result, expected)

        # Test alternative notations
        result2 = convert_dipole_moment(value, "e*a0")
        result3 = convert_dipole_moment(value, "atomic")
        assert np.isclose(result, result2)
        assert np.isclose(result, result3)

    def test_typical_values(self):
        """Test typical molecular dipole moments."""
        # CO2 ν3 mode: ~0.3 Debye
        co2_debye = 0.3
        co2_cm = convert_dipole_moment(co2_debye, "D")
        assert co2_cm > 0
        assert co2_cm < 1e-29  # Reasonable range

    def test_invalid_unit(self):
        """Test error handling for invalid units."""
        with pytest.raises(ValueError, match="Unknown dipole unit"):
            convert_dipole_moment(1.0, "invalid_unit")


class TestElectricFieldConversions:
    """Test electric field conversions."""

    def test_identity_conversion(self):
        """Test that V/m stays unchanged."""
        value = 1e8
        result = convert_electric_field(value, "V/m")
        assert result == value

    def test_field_unit_conversions(self):
        """Test direct field unit conversions."""
        value = 100  # MV/cm
        result = convert_electric_field(value, "MV/cm")
        expected = value * 1e8  # MV/cm → V/m
        assert np.isclose(result, expected)

        value2 = 10  # kV/cm
        result2 = convert_electric_field(value2, "kV/cm")
        expected2 = value2 * 1e5  # kV/cm → V/m
        assert np.isclose(result2, expected2)

    def test_intensity_conversions(self):
        """Test intensity to electric field conversions."""
        # Test W/cm² conversion
        intensity = 1e12  # W/cm²
        result = convert_electric_field(intensity, "W/cm^2")

        # Verify with physics: E = sqrt(2*I*μ0*c*conversion_factors)
        assert result > 0
        assert isinstance(result, float)

        # Test alternative notation
        result2 = convert_electric_field(intensity, "W/cm2")
        assert np.isclose(result, result2)

    def test_high_intensity_conversions(self):
        """Test high-intensity laser conversions."""
        intensities = {
            "TW/cm^2": 1.0,  # 1 TW/cm²
            "GW/cm^2": 1.0,  # 1 GW/cm²
            "MW/cm^2": 1.0,  # 1 MW/cm²
        }

        results = {}
        for unit, value in intensities.items():
            results[unit] = convert_electric_field(value, unit)

        # Higher intensity should give higher field
        assert results["TW/cm^2"] > results["GW/cm^2"]
        assert results["GW/cm^2"] > results["MW/cm^2"]

    def test_invalid_unit(self):
        """Test error handling for invalid units."""
        with pytest.raises(ValueError, match="Unknown electric field/intensity unit"):
            convert_electric_field(100, "invalid_unit")


class TestEnergyConversions:
    """Test energy conversions."""

    def test_identity_conversion(self):
        """Test that J stays unchanged."""
        value = 1e-20
        result = convert_energy(value, "J")
        assert result == value

    def test_ev_conversion(self):
        """Test eV to J conversion."""
        value = 1.0  # eV
        result = convert_energy(value, "eV")
        expected = value * _e
        assert np.isclose(result, expected)

    def test_wavenumber_energy_conversion(self):
        """Test cm⁻¹ to J conversion for energy."""
        value = 2000.0  # cm⁻¹
        result = convert_energy(value, "cm^-1")
        # E = hcν where ν is in cm⁻¹
        expected = value * 6.62607015e-34 * _c * 1e2
        assert np.isclose(result, expected)

    def test_typical_molecular_energies(self):
        """Test typical molecular energy scales."""
        # Thermal energy at room temperature: ~25 meV
        thermal_mev = 25.0
        thermal_j = convert_energy(thermal_mev, "meV")

        # Should be around kT ≈ 4e-21 J at 300K
        assert 1e-22 < thermal_j < 1e-20


class TestTimeConversions:
    """Test time conversions."""

    def test_identity_conversion(self):
        """Test that fs stays unchanged."""
        value = 100.0
        result = convert_time(value, "fs")
        assert result == value

    def test_common_time_conversions(self):
        """Test common time unit conversions."""
        value = 1.0

        # ps → fs
        result_ps = convert_time(value, "ps")
        assert np.isclose(result_ps, 1000.0)

        # ns → fs
        result_ns = convert_time(value, "ns")
        assert np.isclose(result_ns, 1e6)

        # s → fs
        result_s = convert_time(value, "s")
        assert np.isclose(result_s, 1e15)


class TestPhysicalConsistency:
    """Test physical consistency of conversions."""

    def test_co2_parameters(self):
        """Test realistic CO2 molecule parameters."""
        omega_rad_fs = Frequency(2349.1, "cm^-1").angular_rad_per_fs
        rotational_rad_fs = Frequency(0.39021, "cm^-1").angular_rad_per_fs
        mu_cm = convert_dipole_moment(0.3, "D")

        # Vibrational frequency should be much larger than rotational
        assert omega_rad_fs > rotational_rad_fs * 1000

        # Dipole moment should be in reasonable range
        assert 1e-31 < mu_cm < 1e-29

    def test_energy_frequency_consistency(self):
        """Test consistency between energy and frequency conversions."""
        # Same physical quantity in different units
        freq_thz = 100.0  # THz
        energy_j = freq_thz * 1e12 * 6.62607015e-34  # E = hf

        freq_rad_fs = convert_frequency(freq_thz, "THz")
        energy_from_freq = freq_rad_fs * 6.62607015e-034 / (2 * np.pi) * 1e15

        # Should be approximately equal (within numerical precision)
        assert np.isclose(energy_j, energy_from_freq, rtol=1e-10)

    def test_field_intensity_consistency(self):
        """Test consistency between field and intensity."""
        # Convert intensity to field and back
        intensity = 1e12  # W/cm²
        field = convert_electric_field(intensity, "W/cm^2")
        expected_peak = np.sqrt(2.0 * intensity * 1.0e4 * CONSTANTS.MU0 * CONSTANTS.C)
        assert field == pytest.approx(expected_peak)

        # Field should be positive and reasonable
        assert field > 0
        assert 1e6 < field < 1e12  # Reasonable range for strong fields


def test_time_quantity_converts_once_to_femtoseconds():
    quantity = TimeQuantity(0.013, "ps")
    assert quantity.femtoseconds == pytest.approx(13.0)


def test_time_quantity_rejects_unknown_unit():
    with pytest.raises(ValueError, match="invalid time unit"):
        TimeQuantity(1.0, "fortnight")


def test_scalar_quantity_types_convert_once_to_canonical_units():
    dipole = DipoleMoment(0.3, "D")
    amplitude = ElectricFieldAmplitude(1.0, "MV/cm")
    gdd = GroupDelayDispersion(2.0, "ps^2")
    tod = ThirdOrderDispersion(3.0, "ps^3")
    gain = LocalControlGain(1000.0, "(GV/m)^2 fs")
    penalty = KrotovPenalty(0.01, "1 / ((GV/m)^2 fs)")

    assert dipole.coulomb_meters == pytest.approx(0.3 * _debye)
    assert amplitude.volts_per_meter == pytest.approx(1.0e8)
    assert gdd.femtoseconds_squared == pytest.approx(2.0e6)
    assert tod.femtoseconds_cubed == pytest.approx(3.0e9)
    assert gain.volts_per_meter_squared_femtoseconds == pytest.approx(1.0e21)
    assert penalty.inverse_volts_per_meter_squared_femtoseconds == pytest.approx(
        1.0e-20
    )


@pytest.mark.parametrize(
    ("value", "unit", "expected"),
    [
        (1.0e18, "(V/m)^2 fs", 1.0e18),
        (1.0e6, "(MV/m)^2 fs", 1.0e18),
        (1.0, "(GV/m)^2 fs", 1.0e18),
        (1.0e-6, "(TV/m)^2 fs", 1.0e18),
    ],
)
def test_local_control_gain_units_have_one_canonical_value(value, unit, expected):
    gain = LocalControlGain(value, unit)

    assert gain.volts_per_meter_squared_femtoseconds == pytest.approx(expected)


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_local_control_gain_must_be_positive(value):
    with pytest.raises(ValueError, match="local control gain must be positive"):
        LocalControlGain(value, "(GV/m)^2 fs")


@pytest.mark.parametrize(
    ("value", "unit", "expected"),
    [
        (1.0e-20, "1 / ((V/m)^2 fs)", 1.0e-20),
        (1.0e-8, "1 / ((MV/m)^2 fs)", 1.0e-20),
        (1.0e-2, "1 / ((GV/m)^2 fs)", 1.0e-20),
        (1.0e4, "1 / ((TV/m)^2 fs)", 1.0e-20),
    ],
)
def test_krotov_penalty_units_have_one_canonical_value(value, unit, expected):
    penalty = KrotovPenalty(value, unit)

    assert penalty.inverse_volts_per_meter_squared_femtoseconds == pytest.approx(
        expected
    )


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_krotov_penalty_must_be_positive(value):
    with pytest.raises(ValueError, match="Krotov penalty must be positive"):
        KrotovPenalty(value, "1 / ((GV/m)^2 fs)")


def test_intensity_quantity_uses_cycle_average_to_peak_field_contract():
    quantity = ElectricFieldAmplitude(1.0e12, "W/cm^2")
    expected = np.sqrt(2.0 * 1.0e16 * CONSTANTS.MU0 * CONSTANTS.C)

    assert quantity.volts_per_meter == pytest.approx(expected)


@pytest.mark.parametrize(
    ("factory", "unit"),
    [
        (DipoleMoment, "not-a-dipole-unit"),
        (ElectricFieldAmplitude, "not-a-field-unit"),
        (GroupDelayDispersion, "not-a-gdd-unit"),
        (ThirdOrderDispersion, "not-a-tod-unit"),
        (LocalControlGain, "not-a-gain-unit"),
        (KrotovPenalty, "not-a-penalty-unit"),
    ],
)
def test_scalar_quantity_types_reject_unknown_units(factory, unit):
    with pytest.raises(ValueError, match="invalid"):
        factory(1.0, unit)


@pytest.mark.parametrize(
    "factory",
    [
        DipoleMoment,
        ElectricFieldAmplitude,
        GroupDelayDispersion,
        ThirdOrderDispersion,
        LocalControlGain,
        KrotovPenalty,
    ],
)
def test_scalar_quantity_types_reject_nonfinite_values(factory):
    with pytest.raises(ValueError, match="finite"):
        if factory is LocalControlGain:
            unit = "(V/m)^2 fs"
        elif factory is KrotovPenalty:
            unit = "1 / ((V/m)^2 fs)"
        else:
            unit = "C*m"
        factory(np.nan, unit)


if __name__ == "__main__":
    pytest.main([__file__])
