"""Contracts for the current public unit-system guide."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from rovibrational_excitation.core.units import (
    ElectricFieldAmplitude,
    Frequency,
    KrotovPenalty,
    LocalControlGain,
    TimeQuantity,
)
from rovibrational_excitation.core.units.converters import converter
from rovibrational_excitation.fields import ElectricField

ROOT = Path(__file__).resolve().parents[2]
GUIDE = ROOT / "docs" / "UNIT_SYSTEM.md"
LOCAL_LINK = re.compile(r"\[[^]]+\]\((?P<target>[^)]+)\)")

SUPPORTED_UNITS = {
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
    "field_amplitude": {
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
    },
    "time": {"fs", "ps", "ns", "μs", "us", "ms", "s", "atomic"},
    "gdd": {"fs^2", "ps^2", "ns^2", "μs^2", "us^2", "ms^2", "s^2"},
    "tod": {"fs^3", "ps^3", "ns^3", "μs^3", "us^3", "ms^3", "s^3"},
    "local_control_gain": {
        "(V/m)^2 fs",
        "(MV/m)^2 fs",
        "(GV/m)^2 fs",
        "(TV/m)^2 fs",
    },
    "krotov_penalty": {
        "1 / ((V/m)^2 fs)",
        "1 / ((MV/m)^2 fs)",
        "1 / ((GV/m)^2 fs)",
        "1 / ((TV/m)^2 fs)",
    },
}
INTENSITY_UNITS = {
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
}


def _guide() -> str:
    return GUIDE.read_text()


@pytest.mark.parametrize("quantity", sorted(SUPPORTED_UNITS))
def test_guide_lists_every_supported_unit_spelling(quantity: str) -> None:
    expected = SUPPORTED_UNITS[quantity]
    assert set(converter.get_supported_units(quantity)) == expected
    text = _guide()
    for unit in expected:
        assert f"`{unit}`".replace("`", chr(96)) in text


def test_guide_lists_intensity_aliases_separately_from_direct_field_units() -> None:
    assert set(converter.get_supported_units("field")) == (
        SUPPORTED_UNITS["field_amplitude"] | INTENSITY_UNITS
    )
    text = _guide()
    for unit in INTENSITY_UNITS:
        assert f"`{unit}`".replace("`", chr(96)) in text
    assert "array 入力へ\nintensity label を認めない" in text


def test_frequency_examples_are_physically_equivalent_without_double_2pi() -> None:
    thz = Frequency(100.0, "THz")
    phz = Frequency(0.1, "PHz")
    angular = Frequency(2 * np.pi * 0.1, "rad/fs")

    assert thz.value == 100.0
    assert thz.unit == "THz"
    assert thz.angular_rad_per_fs == pytest.approx(phz.angular_rad_per_fs)
    assert thz.angular_rad_per_fs == pytest.approx(angular.angular_rad_per_fs)
    assert thz.cycles_per_fs == pytest.approx(0.1)


def test_scalar_quantity_boundaries_retain_input_and_expose_canonical_values() -> None:
    time = TimeQuantity(1.5, "ps")
    generated_amplitude = ElectricFieldAmplitude(1.0, "W/m^2")
    gain = LocalControlGain(2.0, "(GV/m)^2 fs")
    penalty = KrotovPenalty(3.0, "1 / ((GV/m)^2 fs)")

    assert (time.value, time.unit, time.femtoseconds) == (1.5, "ps", 1500.0)
    assert generated_amplitude.unit == "W/m^2"
    assert np.isfinite(generated_amplitude.volts_per_meter)
    assert gain.volts_per_meter_squared_femtoseconds == pytest.approx(2.0e18)
    assert penalty.inverse_volts_per_meter_squared_femtoseconds == pytest.approx(
        3.0e-18
    )


def test_signed_field_array_rejects_intensity_while_generated_scalar_accepts_it() -> (
    None
):
    field = ElectricField(np.array([0.0, 0.5, 1.0]), time_units="fs")

    with pytest.raises(ValueError, match="supported electric-field amplitude unit"):
        field.add_arbitrary_Efield(np.zeros((3, 2)), field_units="W/m^2")


def test_guide_records_provenance_optimizer_and_spectroscopy_boundaries() -> None:
    text = _guide()

    assert "caller が指定した\n値と unit label を変更せず保存" in text
    assert "gain が大きいほど local update の電場変更を強める" in text
    assert "standard-Krotov penalty が\n大きいほど更新を抑える" in text
    assert "GRAPE と `legacy_batch_overlap` の `lambda_a`".replace("`", chr(96)) in text
    assert '`wavenumber_units="cm^-1"`'.replace("`", chr(96)) in text
    assert "user data を規格化、対称化、clip、resample" in text


def test_unit_guide_local_links_resolve() -> None:
    missing: list[str] = []
    for match in LOCAL_LINK.finditer(_guide()):
        target = match.group("target").split("#", 1)[0]
        if target and not (GUIDE.parent / target).resolve().exists():
            missing.append(target)

    assert missing == []
