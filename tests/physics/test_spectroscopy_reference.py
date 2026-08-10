"""Spectroscopy calculation-mode and broadening contracts."""

from __future__ import annotations

import numpy as np
import pytest

from rovibrational_excitation.core.basis import TwoLevelBasis
from rovibrational_excitation.core.units.constants import CONSTANTS
from rovibrational_excitation.dipole import TwoLevelDipoleMatrix
from rovibrational_excitation.spectroscopy import (
    AbsorbanceCalculator,
    ExperimentalConditions,
)


@pytest.fixture
def two_level_case():
    basis = TwoLevelBasis(
        energy_gap=1.0e-20,
        input_units="J",
        output_units="J",
    )
    hamiltonian = basis.generate_H0()
    dipole = TwoLevelDipoleMatrix(basis, mu0=1.0e-30)
    conditions = ExperimentalConditions(
        temperature=300.0,
        pressure=3.0e4,
        optical_length=1.0e-3,
        T2=500.0,
        molecular_mass=44.0e-3 / CONSTANTS.AVOGADRO,
    )
    calculator = AbsorbanceCalculator(
        basis,
        hamiltonian,
        dipole,
        conditions,
        axes="xy",
        pol_int=np.array([1.0, 0.0]),
        pol_det=np.array([1.0, 0.0]),
        use_v_mask=False,
    )
    rho = np.diag([1.0, 0.0]).astype(np.complex128)
    wavenumber = np.linspace(400.0, 600.0, 17)
    return calculator, rho, wavenumber


def test_experimental_conditions_are_explicit_and_positive():
    with pytest.raises(TypeError):
        ExperimentalConditions()

    valid = {
        "temperature": 300.0,
        "pressure": 3.0e4,
        "optical_length": 1.0e-3,
        "T2": 500.0,
        "molecular_mass": 44.0e-3 / CONSTANTS.AVOGADRO,
    }
    for name in valid:
        invalid = valid | {name: 0.0}
        with pytest.raises(ValueError, match=f"{name} must be finite and positive"):
            ExperimentalConditions(**invalid)


def test_exact_methods_retain_physical_scale_transitions(two_level_case):
    calculator, rho, wavenumber = two_level_case

    loop = calculator.calculate(rho, wavenumber, method="loop")
    matrix = calculator.calculate(rho, wavenumber, method="matrix")
    two_dimensional = calculator.calculate(rho, wavenumber, method="2d")
    chunked = calculator.calculate(
        rho,
        wavenumber,
        method="chunked",
        chunk_size=5,
    )

    assert np.max(np.abs(loop)) > 1.0e-3
    np.testing.assert_allclose(matrix, loop, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(two_dimensional, loop, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(chunked, loop, rtol=2e-14, atol=2e-14)
    assert calculator.last_calculation_report.executed_method == "chunked"
    assert calculator.last_calculation_report.relative_threshold is None
    assert calculator.last_calculation_report.discarded_commutator_l2_fraction == 0.0


def test_method_is_required_and_controls_are_mode_specific(two_level_case):
    calculator, rho, wavenumber = two_level_case

    with pytest.raises(TypeError):
        calculator.calculate(rho, wavenumber)

    with pytest.raises(ValueError, match="relative_threshold.*approximate_sparse"):
        calculator.calculate(
            rho,
            wavenumber,
            method="loop",
            relative_threshold=1.0e-3,
        )

    with pytest.raises(ValueError, match="memory_budget_bytes.*auto"):
        calculator.calculate(
            rho,
            wavenumber,
            method="loop",
            memory_budget_bytes=1024,
        )

    with pytest.raises(ValueError, match="chunk_size.*chunked"):
        calculator.calculate(
            rho,
            wavenumber,
            method="loop",
            chunk_size=8,
        )

    with pytest.raises(ValueError, match="Unknown method"):
        calculator.calculate(rho, wavenumber, method="optimized")


def test_auto_requires_budget_and_records_executed_method(two_level_case):
    calculator, rho, wavenumber = two_level_case

    with pytest.raises(ValueError, match="memory_budget_bytes is required"):
        calculator.calculate(
            rho,
            wavenumber,
            method="auto",
            chunk_size=5,
        )

    calculator.calculate(
        rho,
        wavenumber,
        method="auto",
        memory_budget_bytes=10**9,
        chunk_size=5,
    )
    report = calculator.last_calculation_report
    assert report.requested_method == "auto"
    assert report.executed_method == "2d"
    assert report.memory_budget_bytes == 10**9
    assert report.estimated_2d_bytes < report.memory_budget_bytes

    calculator.calculate(
        rho,
        wavenumber,
        method="auto",
        memory_budget_bytes=1,
        chunk_size=5,
    )
    report = calculator.last_calculation_report
    assert report.executed_method == "chunked"
    assert report.estimated_2d_bytes > report.memory_budget_bytes


class _ThreeLevelBasis:
    V_array = np.array([0, 1, 2])


class _ThreeLevelHamiltonian:
    def get_eigenvalues(self, units):
        assert units == "J"
        return np.array([0.0, 1.0e-20, 2.0e-20])


class _ThreeLevelDipole:
    def __init__(self):
        self._x = np.array(
            [
                [0.0, 1.0e-30, 1.0e-34],
                [1.0e-30, 0.0, 2.0e-32],
                [1.0e-34, 2.0e-32, 0.0],
            ],
            dtype=np.complex128,
        )

    def get_mu_x_SI(self):
        return self._x.copy()

    def get_mu_y_SI(self):
        return np.zeros_like(self._x)

    def get_mu_z_SI(self):
        return np.zeros_like(self._x)


class _OrthogonalTransitionDipole:
    def __init__(self):
        self._x = np.array(
            [
                [0.0, 1.0e-30, 0.0],
                [1.0e-30, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.complex128,
        )
        self._y = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0e-30],
                [0.0, 1.0e-30, 0.0],
            ],
            dtype=np.complex128,
        )

    def get_mu_x_SI(self):
        return self._x.copy()

    def get_mu_y_SI(self):
        return self._y.copy()

    def get_mu_z_SI(self):
        return np.zeros_like(self._x)


def test_exact_methods_enumerate_detection_transition_support():
    conditions = ExperimentalConditions(
        temperature=300.0,
        pressure=3.0e4,
        optical_length=1.0e-3,
        T2=500.0,
        molecular_mass=44.0e-3 / CONSTANTS.AVOGADRO,
    )
    calculator = AbsorbanceCalculator(
        _ThreeLevelBasis(),
        _ThreeLevelHamiltonian(),
        _OrthogonalTransitionDipole(),
        conditions,
        axes="xy",
        pol_int=np.array([1.0, 0.0]),
        pol_det=np.array([0.0, 1.0]),
        use_v_mask=False,
    )
    state = np.array([1.0, 0.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    rho = np.outer(state, state.conj())
    wavenumber = np.linspace(400.0, 1200.0, 19)

    loop = calculator.calculate(rho, wavenumber, method="loop")
    matrix = calculator.calculate(rho, wavenumber, method="matrix")
    two_dimensional = calculator.calculate(rho, wavenumber, method="2d")
    chunked = calculator.calculate(
        rho,
        wavenumber,
        method="chunked",
        chunk_size=7,
    )

    assert np.max(np.abs(loop)) > 1.0e-3
    np.testing.assert_allclose(matrix, loop, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(two_dimensional, loop, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(chunked, loop, rtol=2e-14, atol=2e-14)


def test_approximate_sparse_requires_relative_threshold_and_reports_discarding():
    conditions = ExperimentalConditions(
        temperature=300.0,
        pressure=3.0e4,
        optical_length=1.0e-3,
        T2=500.0,
        molecular_mass=44.0e-3 / CONSTANTS.AVOGADRO,
    )
    calculator = AbsorbanceCalculator(
        _ThreeLevelBasis(),
        _ThreeLevelHamiltonian(),
        _ThreeLevelDipole(),
        conditions,
        axes="xy",
        pol_int=np.array([1.0, 0.0]),
        pol_det=np.array([1.0, 0.0]),
        use_v_mask=False,
    )
    rho = np.diag([1.0, 0.5, 0.0]).astype(np.complex128)
    wavenumber = np.linspace(400.0, 1200.0, 19)

    with pytest.raises(ValueError, match="relative_threshold is required"):
        calculator.calculate(
            rho,
            wavenumber,
            method="approximate_sparse",
            chunk_size=7,
        )

    for invalid in (0.0, -1.0e-3, 1.1):
        with pytest.raises(ValueError, match="0 < relative_threshold <= 1"):
            calculator.calculate(
                rho,
                wavenumber,
                method="approximate_sparse",
                chunk_size=7,
                relative_threshold=invalid,
            )

    calculator.calculate(
        rho,
        wavenumber,
        method="approximate_sparse",
        chunk_size=7,
        relative_threshold=0.1,
    )
    report = calculator.last_calculation_report
    assert report.requested_method == "approximate_sparse"
    assert report.executed_method == "chunked"
    assert report.relative_threshold == 0.1
    assert 0.0 < report.discarded_commutator_l2_fraction < 1.0


def test_device_function_is_applied_only_when_explicit(two_level_case, monkeypatch):
    calculator, rho, wavenumber = two_level_case
    calls = []

    def fake_device(spectrum, supplied_wavenumber, resolution, function_type="sinc2"):
        calls.append((supplied_wavenumber.copy(), resolution, function_type))
        return spectrum + 2.0

    monkeypatch.setattr(calculator, "apply_device_function", fake_device)

    with pytest.raises(ValueError, match="device_resolution.*apply_device_function"):
        calculator.calculate(
            rho,
            wavenumber,
            method="loop",
            device_resolution=2.0,
        )

    baseline = calculator.calculate(rho, wavenumber, method="loop")
    broadened = calculator.calculate(
        rho,
        wavenumber,
        method="loop",
        apply_device_function=True,
        device_resolution=2.0,
    )

    np.testing.assert_allclose(broadened, baseline + 2.0)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], wavenumber)
    assert calls[0][1:] == (2.0, "sinc2")
    assert calculator.last_calculation_report.device_function_applied is True

    with pytest.raises(ValueError, match="unknown device function"):
        AbsorbanceCalculator.apply_device_function(
            calculator,
            baseline,
            wavenumber,
            resolution=2.0,
            function_type="lorentzian",
        )


def test_doppler_uses_grid_resolution_without_fixed_cutoff(
    two_level_case,
    monkeypatch,
):
    calculator, _rho, wavenumber = two_level_case
    calculator.conditions.molecular_mass = 1.0e-15
    response = np.linspace(0.0, 1.0, len(wavenumber)).astype(np.complex128)
    sigma_values = []

    def fake_gaussian(values, sigma, mode):
        assert mode == "reflect"
        sigma_values.append(sigma)
        return values.copy()

    monkeypatch.setattr(
        "rovibrational_excitation.spectroscopy.absorbance_calculator.ndimage.gaussian_filter1d",
        fake_gaussian,
    )

    calculator._apply_doppler_broadening_full(wavenumber, response)

    assert len(sigma_values) == 2
    assert 0.0 < sigma_values[0] < 0.01 / np.diff(wavenumber)[0]
    assert sigma_values[0] == sigma_values[1]


def test_doppler_rejects_nonuniform_grid(two_level_case):
    calculator, rho, _wavenumber = two_level_case

    with pytest.raises(ValueError, match="uniformly spaced"):
        calculator.calculate(
            rho,
            np.array([400.0, 401.0, 403.0]),
            method="loop",
            apply_doppler=True,
        )


def test_spectroscopy_uses_authoritative_constants(two_level_case):
    calculator, _rho, _wavenumber = two_level_case

    expected_density = calculator.conditions.pressure / (
        CONSTANTS.BOLTZMANN * calculator.conditions.temperature
    )
    assert calculator.conditions.number_density == expected_density
