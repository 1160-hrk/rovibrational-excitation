"""Spectroscopy calculation-mode and broadening contracts."""

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
    create_calculator_from_params,
)

WAVENUMBER_UNITS = "cm^-1"


def _experimental_conditions() -> ExperimentalConditions:
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


@pytest.fixture
def two_level_case():
    basis = TwoLevelBasis(
        energy_gap=1.0e-20,
        input_units="J",
        output_units="J",
    )
    hamiltonian = basis.generate_H0()
    dipole = TwoLevelDipoleMatrix(basis, mu0=1.0e-30)
    conditions = _experimental_conditions()
    calculator = AbsorbanceCalculator(
        basis,
        hamiltonian,
        dipole,
        conditions,
        phase_matching="unfiltered",
        axes="xy",
        pol_int=np.array([1.0, 0.0]),
        pol_det=np.array([1.0, 0.0]),
    )
    rho = np.diag([1.0, 0.0]).astype(np.complex128)
    wavenumber = np.linspace(400.0, 600.0, 17)
    return calculator, rho, wavenumber


def test_experimental_conditions_are_explicit_and_positive():
    with pytest.raises(TypeError):
        ExperimentalConditions()

    valid = {
        "temperature": 300.0,
        "temperature_units": "K",
        "pressure": 3.0e4,
        "pressure_units": "Pa",
        "optical_length": 1.0e-3,
        "optical_length_units": "m",
        "coherence_time": 500.0,
        "coherence_time_units": "ps",
        "molecular_mass": 44.0e-3 / CONSTANTS.AVOGADRO,
        "molecular_mass_units": "kg",
    }
    for name in (
        "temperature",
        "pressure",
        "optical_length",
        "coherence_time",
        "molecular_mass",
    ):
        invalid = valid | {name: 0.0}
        with pytest.raises(ValueError, match=f"{name} must be finite and positive"):
            ExperimentalConditions(**invalid)


@pytest.mark.parametrize(
    ("unit_name", "invalid_unit"),
    [
        ("temperature_units", "degC"),
        ("pressure_units", "bar"),
        ("optical_length_units", "cm"),
        ("coherence_time_units", "fs"),
        ("molecular_mass_units", "g/mol"),
    ],
)
def test_experimental_condition_units_are_required_and_exact(
    unit_name,
    invalid_unit,
):
    values = {
        "temperature": 300.0,
        "temperature_units": "K",
        "pressure": 3.0e4,
        "pressure_units": "Pa",
        "optical_length": 1.0e-3,
        "optical_length_units": "m",
        "coherence_time": 500.0,
        "coherence_time_units": "ps",
        "molecular_mass": 44.0e-3 / CONSTANTS.AVOGADRO,
        "molecular_mass_units": "kg",
    }
    missing = values.copy()
    del missing[unit_name]
    with pytest.raises(TypeError):
        ExperimentalConditions(**missing)

    invalid = values | {unit_name: invalid_unit}
    with pytest.raises(ValueError, match=unit_name):
        ExperimentalConditions(**invalid)


def test_calculator_factory_requires_and_forwards_condition_units(two_level_case):
    reference, _rho, _wavenumber = two_level_case
    calculator = create_calculator_from_params(
        reference.basis,
        reference.hamiltonian,
        reference.dipole_matrix,
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
        phase_matching="unfiltered",
        axes="xy",
        pol_int=np.array([1.0, 0.0]),
        pol_det=np.array([1.0, 0.0]),
    )

    assert calculator.conditions == reference.conditions


def test_exact_methods_retain_physical_scale_transitions(two_level_case):
    calculator, rho, wavenumber = two_level_case

    loop = calculator.calculate(
        rho, wavenumber, method="loop", wavenumber_units=WAVENUMBER_UNITS
    )
    matrix = calculator.calculate(
        rho, wavenumber, method="matrix", wavenumber_units=WAVENUMBER_UNITS
    )
    two_dimensional = calculator.calculate(
        rho, wavenumber, method="2d", wavenumber_units=WAVENUMBER_UNITS
    )
    chunked = calculator.calculate(
        rho,
        wavenumber,
        method="chunked",
        wavenumber_units=WAVENUMBER_UNITS,
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
        calculator.calculate(
            rho,
            wavenumber,
            wavenumber_units=WAVENUMBER_UNITS,
        )

    with pytest.raises(TypeError, match="wavenumber_units"):
        calculator.calculate(rho, wavenumber, method="loop")

    with pytest.raises(ValueError, match="wavenumber_units"):
        calculator.calculate(
            rho,
            wavenumber,
            method="loop",
            wavenumber_units="m^-1",
        )

    with pytest.raises(ValueError, match="relative_threshold.*approximate_sparse"):
        calculator.calculate(
            rho,
            wavenumber,
            method="loop",
            wavenumber_units=WAVENUMBER_UNITS,
            relative_threshold=1.0e-3,
        )

    with pytest.raises(ValueError, match="memory_budget_bytes.*auto"):
        calculator.calculate(
            rho,
            wavenumber,
            method="loop",
            wavenumber_units=WAVENUMBER_UNITS,
            memory_budget_bytes=1024,
        )

    with pytest.raises(ValueError, match="chunk_size.*chunked"):
        calculator.calculate(
            rho,
            wavenumber,
            method="loop",
            wavenumber_units=WAVENUMBER_UNITS,
            chunk_size=8,
        )

    with pytest.raises(ValueError, match="Unknown method"):
        calculator.calculate(
            rho, wavenumber, method="optimized", wavenumber_units=WAVENUMBER_UNITS
        )


def test_all_public_spectral_entry_points_require_units(two_level_case):
    calculator, rho, wavenumber = two_level_case
    spectrum = np.zeros_like(wavenumber)

    with pytest.raises(TypeError, match="wavenumber_units"):
        calculator.prepare_2d_calculation(wavenumber)
    with pytest.raises(ValueError, match="wavenumber_units"):
        calculator.prepare_2d_calculation(wavenumber, wavenumber_units="m^-1")
    with pytest.raises(TypeError, match="wavenumber_units"):
        calculator.calculate_radiation_spectrum(rho, wavenumber)
    with pytest.raises(TypeError, match="wavenumber_units"):
        calculator.calculate_pfid_spectrum(rho, wavenumber)
    with pytest.raises(TypeError, match="wavenumber_units"):
        calculator.apply_device_function(spectrum, wavenumber, resolution=2.0)


def test_auto_requires_budget_and_records_executed_method(two_level_case):
    calculator, rho, wavenumber = two_level_case

    with pytest.raises(ValueError, match="memory_budget_bytes is required"):
        calculator.calculate(
            rho,
            wavenumber,
            method="auto",
            wavenumber_units=WAVENUMBER_UNITS,
            chunk_size=5,
        )

    calculator.calculate(
        rho,
        wavenumber,
        method="auto",
        wavenumber_units=WAVENUMBER_UNITS,
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
        wavenumber_units=WAVENUMBER_UNITS,
        memory_budget_bytes=1,
        chunk_size=5,
    )
    report = calculator.last_calculation_report
    assert report.executed_method == "chunked"
    assert report.estimated_2d_bytes > report.memory_budget_bytes


class _ThreeLevelBasis:
    V_array = np.array([0, 1, 2])


class _SameVBlockBasis:
    V_array = np.array([0, 0, 1])


class _BasisWithoutV:
    pass


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
    conditions = _experimental_conditions()
    calculator = AbsorbanceCalculator(
        _ThreeLevelBasis(),
        _ThreeLevelHamiltonian(),
        _OrthogonalTransitionDipole(),
        conditions,
        phase_matching="unfiltered",
        axes="xy",
        pol_int=np.array([1.0, 0.0]),
        pol_det=np.array([0.0, 1.0]),
    )
    state = np.array([1.0, 0.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    rho = np.outer(state, state.conj())
    wavenumber = np.linspace(400.0, 1200.0, 19)

    loop = calculator.calculate(
        rho, wavenumber, method="loop", wavenumber_units=WAVENUMBER_UNITS
    )
    matrix = calculator.calculate(
        rho, wavenumber, method="matrix", wavenumber_units=WAVENUMBER_UNITS
    )
    two_dimensional = calculator.calculate(
        rho, wavenumber, method="2d", wavenumber_units=WAVENUMBER_UNITS
    )
    chunked = calculator.calculate(
        rho,
        wavenumber,
        method="chunked",
        wavenumber_units=WAVENUMBER_UNITS,
        chunk_size=7,
    )

    assert np.max(np.abs(loop)) > 1.0e-3
    np.testing.assert_allclose(matrix, loop, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(two_dimensional, loop, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(chunked, loop, rtol=2e-14, atol=2e-14)


def test_approximate_sparse_requires_relative_threshold_and_reports_discarding():
    conditions = _experimental_conditions()
    calculator = AbsorbanceCalculator(
        _ThreeLevelBasis(),
        _ThreeLevelHamiltonian(),
        _ThreeLevelDipole(),
        conditions,
        phase_matching="unfiltered",
        axes="xy",
        pol_int=np.array([1.0, 0.0]),
        pol_det=np.array([1.0, 0.0]),
    )
    rho = np.diag([1.0, 0.5, 0.0]).astype(np.complex128)
    wavenumber = np.linspace(400.0, 1200.0, 19)

    with pytest.raises(ValueError, match="relative_threshold is required"):
        calculator.calculate(
            rho,
            wavenumber,
            method="approximate_sparse",
            wavenumber_units=WAVENUMBER_UNITS,
            chunk_size=7,
        )

    for invalid in (0.0, -1.0e-3, 1.1):
        with pytest.raises(ValueError, match="0 < relative_threshold <= 1"):
            calculator.calculate(
                rho,
                wavenumber,
                method="approximate_sparse",
                wavenumber_units=WAVENUMBER_UNITS,
                chunk_size=7,
                relative_threshold=invalid,
            )

    calculator.calculate(
        rho,
        wavenumber,
        method="approximate_sparse",
        wavenumber_units=WAVENUMBER_UNITS,
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

    def fake_device(
        spectrum,
        supplied_wavenumber,
        resolution,
        *,
        wavenumber_units,
        resolution_units,
        function_type="sinc2",
    ):
        calls.append(
            (
                supplied_wavenumber.copy(),
                resolution,
                wavenumber_units,
                resolution_units,
                function_type,
            )
        )
        return spectrum + 2.0

    monkeypatch.setattr(calculator, "apply_device_function", fake_device)

    with pytest.raises(ValueError, match="device_resolution.*apply_device_function"):
        calculator.calculate(
            rho,
            wavenumber,
            method="loop",
            wavenumber_units=WAVENUMBER_UNITS,
            device_resolution=2.0,
        )

    with pytest.raises(ValueError, match="device_resolution_units"):
        calculator.calculate(
            rho,
            wavenumber,
            method="loop",
            wavenumber_units=WAVENUMBER_UNITS,
            apply_device_function=True,
            device_resolution=2.0,
            device_resolution_units="m^-1",
        )

    with pytest.raises(ValueError, match="applicable only"):
        calculator.calculate(
            rho,
            wavenumber,
            method="loop",
            wavenumber_units=WAVENUMBER_UNITS,
            device_resolution_units="cm^-1",
        )

    baseline = calculator.calculate(
        rho, wavenumber, method="loop", wavenumber_units=WAVENUMBER_UNITS
    )
    broadened = calculator.calculate(
        rho,
        wavenumber,
        method="loop",
        wavenumber_units=WAVENUMBER_UNITS,
        apply_device_function=True,
        device_resolution=2.0,
        device_resolution_units="cm^-1",
    )

    np.testing.assert_allclose(broadened, baseline + 2.0)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], wavenumber)
    assert calls[0][1:] == (2.0, "cm^-1", "cm^-1", "sinc2")
    assert calculator.last_calculation_report.device_function_applied is True

    with pytest.raises(ValueError, match="unknown device function"):
        AbsorbanceCalculator.apply_device_function(
            calculator,
            baseline,
            wavenumber,
            resolution=2.0,
            wavenumber_units="cm^-1",
            resolution_units="cm^-1",
            function_type="lorentzian",
        )


def test_doppler_rejects_nonuniform_grid(two_level_case):
    calculator, rho, _wavenumber = two_level_case

    with pytest.raises(ValueError, match="uniformly spaced"):
        calculator.calculate(
            rho,
            np.array([400.0, 401.0, 403.0]),
            method="loop",
            wavenumber_units=WAVENUMBER_UNITS,
            apply_doppler=True,
        )


def test_spectroscopy_uses_authoritative_constants(two_level_case):
    calculator, _rho, _wavenumber = two_level_case

    expected_density = calculator.conditions.pressure_pa / (
        CONSTANTS.BOLTZMANN * calculator.conditions.temperature_k
    )
    assert calculator.conditions.number_density == expected_density


class _DegenerateCircularHamiltonian:
    def get_eigenvalues(self, units):
        assert units == "J"
        return np.array([0.0, 1.0e-20, 1.0e-20])


class _DegenerateCircularDipole:
    def __init__(self):
        amplitude = 1.0e-30 / np.sqrt(2.0)
        self._x = np.array(
            [
                [0.0, amplitude, amplitude],
                [amplitude, 0.0, 0.0],
                [amplitude, 0.0, 0.0],
            ],
            dtype=np.complex128,
        )
        self._y = np.array(
            [
                [0.0, -1j * amplitude, 1j * amplitude],
                [1j * amplitude, 0.0, 0.0],
                [-1j * amplitude, 0.0, 0.0],
            ],
            dtype=np.complex128,
        )
        self._z = np.zeros((3, 3), dtype=np.complex128)
        self._z[0, 2] = 2.0e-30
        self._z[2, 0] = 2.0e-30

    def get_mu_x_SI(self):
        return self._x.copy()

    def get_mu_y_SI(self):
        return self._y.copy()

    def get_mu_z_SI(self):
        return self._z.copy()


def _spectroscopy_conditions():
    return _experimental_conditions()


def _circular_calculator(polarization, *, axes="xy", pol_det=None):
    return AbsorbanceCalculator(
        _ThreeLevelBasis(),
        _DegenerateCircularHamiltonian(),
        _DegenerateCircularDipole(),
        _spectroscopy_conditions(),
        phase_matching="unfiltered",
        axes=axes,
        pol_int=polarization,
        pol_det=pol_det,
    )


def test_circular_detection_is_adjoint_and_global_phase_invariant():
    polarization = np.array([1.0, 1.0j]) / np.sqrt(2.0)
    reference = _circular_calculator(polarization)
    phase_shifted = _circular_calculator(polarization * np.exp(0.73j))
    rho = np.diag([1.0, 0.0, 0.0]).astype(np.complex128)
    wavenumber = np.linspace(450.0, 550.0, 101)

    np.testing.assert_allclose(reference.mu_det, reference.mu_int.conj().T)
    for method, options in (
        ("loop", {}),
        ("matrix", {}),
        ("2d", {}),
        ("chunked", {"chunk_size": 17}),
    ):
        np.testing.assert_allclose(
            phase_shifted.calculate(
                rho,
                wavenumber,
                method=method,
                wavenumber_units=WAVENUMBER_UNITS,
                **options,
            ),
            reference.calculate(
                rho,
                wavenumber,
                method=method,
                wavenumber_units=WAVENUMBER_UNITS,
                **options,
            ),
            rtol=2.0e-14,
            atol=2.0e-14,
            err_msg=method,
        )


def test_circular_helicity_selection_and_m_symmetric_response():
    plus = np.array([1.0, 1.0j]) / np.sqrt(2.0)
    minus = plus.conj()
    plus_calculator = _circular_calculator(plus)
    minus_calculator = _circular_calculator(minus)
    rho = np.diag([1.0, 0.0, 0.0]).astype(np.complex128)
    wavenumber = np.linspace(450.0, 550.0, 101)

    assert abs(plus_calculator.mu_int[0, 1]) > 0.0
    assert abs(plus_calculator.mu_int[0, 2]) < 1.0e-45
    assert abs(minus_calculator.mu_int[0, 1]) < 1.0e-45
    assert abs(minus_calculator.mu_int[0, 2]) > 0.0
    np.testing.assert_allclose(
        plus_calculator.calculate(
            rho, wavenumber, method="loop", wavenumber_units=WAVENUMBER_UNITS
        ),
        minus_calculator.calculate(
            rho, wavenumber, method="loop", wavenumber_units=WAVENUMBER_UNITS
        ),
        rtol=2.0e-14,
        atol=2.0e-14,
    )


def test_reversing_m_orientation_reverses_circular_dichroism():
    plus = np.array([1.0, 1.0j]) / np.sqrt(2.0)
    minus = plus.conj()
    wavenumber = np.linspace(450.0, 550.0, 101)
    oriented = np.diag([0.8, 0.2, 0.0]).astype(np.complex128)
    reversed_orientation = np.diag([0.8, 0.0, 0.2]).astype(np.complex128)

    difference = _circular_calculator(plus).calculate(
        oriented, wavenumber, method="loop", wavenumber_units=WAVENUMBER_UNITS
    ) - _circular_calculator(minus).calculate(
        oriented, wavenumber, method="loop", wavenumber_units=WAVENUMBER_UNITS
    )
    reversed_difference = _circular_calculator(plus).calculate(
        reversed_orientation,
        wavenumber,
        method="loop",
        wavenumber_units=WAVENUMBER_UNITS,
    ) - _circular_calculator(minus).calculate(
        reversed_orientation,
        wavenumber,
        method="loop",
        wavenumber_units=WAVENUMBER_UNITS,
    )

    assert np.max(np.abs(difference)) > 1.0
    np.testing.assert_allclose(
        reversed_difference,
        -difference,
        rtol=2.0e-14,
        atol=2.0e-14,
    )


def test_linear_polarization_is_unchanged_and_third_axis_contributes():
    linear = _circular_calculator(np.array([3.0, 4.0]))
    expected_linear = (
        3.0 * linear.mu_components["x"] + 4.0 * linear.mu_components["y"]
    ) / 5.0
    np.testing.assert_allclose(linear.mu_int, expected_linear)
    np.testing.assert_allclose(linear.mu_det, expected_linear)

    z_polarized = _circular_calculator(
        np.array([0.0, 0.0, 1.0]),
        axes="xyz",
    )
    np.testing.assert_allclose(z_polarized.mu_int, z_polarized.mu_components["z"])
    np.testing.assert_allclose(z_polarized.mu_det, z_polarized.mu_components["z"])


def test_polarization_vectors_are_strictly_validated():
    for axes in ("", "xx", "xyzz"):
        with pytest.raises(ValueError):
            _circular_calculator(np.array([1.0, 0.0]), axes=axes)

    for invalid in (
        np.array([1.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, np.nan, 0.0]),
    ):
        with pytest.raises(ValueError):
            _circular_calculator(invalid, axes="xyz")

    with pytest.raises(ValueError, match="pol_det.*shape"):
        _circular_calculator(
            np.array([1.0, 0.0]),
            pol_det=[1.0, 0.0, 0.0],
        )


def test_response_conversion_has_no_implicit_orientational_third():
    calculator = _circular_calculator(np.array([1.0, 0.0]))
    omega = np.array([1.2e14])
    density = calculator.conditions.number_density
    response = np.array([1.0j * CONSTANTS.EPSILON0 / density * 1.0e-4])
    refractive_index = np.sqrt(1.0 + response * density / CONSTANTS.EPSILON0)
    expected = (
        2.0
        * calculator.conditions.optical_length_m
        * omega
        / CONSTANTS.C
        * refractive_index.imag
        * np.log10(np.exp(1.0))
        * 1000.0
    )

    np.testing.assert_allclose(
        calculator._response_to_absorbance(omega, response),
        expected,
        rtol=2.0e-14,
        atol=2.0e-14,
    )


@pytest.mark.parametrize(
    ("method", "options"),
    [
        ("2d", {}),
        ("chunked", {"chunk_size": 5}),
        (
            "approximate_sparse",
            {"chunk_size": 5, "relative_threshold": 0.1},
        ),
        ("auto", {"chunk_size": 5, "memory_budget_bytes": 10**6}),
    ],
)
def test_doppler_rejects_routes_without_transition_specific_widths(
    two_level_case,
    method,
    options,
):
    calculator, rho, wavenumber = two_level_case
    with pytest.raises(ValueError, match="Doppler broadening currently requires"):
        calculator.calculate(
            rho,
            wavenumber,
            method=method,
            wavenumber_units=WAVENUMBER_UNITS,
            apply_doppler=True,
            **options,
        )


def test_matrix_and_loop_share_transition_specific_doppler(two_level_case):
    calculator, rho, wavenumber = two_level_case
    loop = calculator.calculate(
        rho,
        wavenumber,
        method="loop",
        wavenumber_units=WAVENUMBER_UNITS,
        apply_doppler=True,
    )
    matrix = calculator.calculate(
        rho,
        wavenumber,
        method="matrix",
        wavenumber_units=WAVENUMBER_UNITS,
        apply_doppler=True,
    )

    np.testing.assert_allclose(matrix, loop, rtol=2.0e-14, atol=2.0e-14)


def _pathway_calculator(basis, phase_matching):
    return AbsorbanceCalculator(
        basis,
        _ThreeLevelHamiltonian(),
        _ThreeLevelDipole(),
        _spectroscopy_conditions(),
        phase_matching=phase_matching,
        axes="x",
        pol_int=np.array([1.0]),
        pol_det=np.array([1.0]),
    )


def test_pump_probe_selects_same_v_blocks_and_reports_discarded_norm():
    calculator = _pathway_calculator(_SameVBlockBasis(), "pump_probe")
    state = np.array([1.0, 2.0, 3.0j], dtype=np.complex128) / np.sqrt(14.0)
    rho = np.outer(state, state.conj())
    selected = calculator._select_pre_probe_density(rho)
    expected_mask = np.equal.outer(
        _SameVBlockBasis.V_array,
        _SameVBlockBasis.V_array,
    )

    np.testing.assert_allclose(selected, np.where(expected_mask, rho, 0.0))
    assert selected[0, 1] == rho[0, 1]  # same-V J/M-like coherence survives
    assert selected[0, 2] == 0.0  # cross-V optical-order coherence is removed

    wavenumber = np.linspace(400.0, 1200.0, 19)
    loop = calculator.calculate(
        rho, wavenumber, method="loop", wavenumber_units=WAVENUMBER_UNITS
    )
    for method, options in (
        ("matrix", {}),
        ("2d", {}),
        ("chunked", {"chunk_size": 7}),
    ):
        np.testing.assert_allclose(
            calculator.calculate(
                rho,
                wavenumber,
                method=method,
                wavenumber_units=WAVENUMBER_UNITS,
                **options,
            ),
            loop,
            rtol=2.0e-14,
            atol=2.0e-14,
            err_msg=method,
        )

    report = calculator.last_calculation_report
    discarded = np.where(expected_mask, 0.0, rho)
    expected_fraction = np.linalg.norm(discarded) / np.linalg.norm(rho)
    assert report.phase_matching == "pump_probe"
    assert report.discarded_density_l2_fraction == pytest.approx(expected_fraction)

    calculator.calculate(
        np.zeros_like(rho), wavenumber, method="loop", wavenumber_units=WAVENUMBER_UNITS
    )
    assert calculator.last_calculation_report.discarded_density_l2_fraction == 0.0


def test_unfiltered_and_pump_probe_pathways_are_observably_distinct():
    state = np.array([1.0, 2.0, 3.0j], dtype=np.complex128) / np.sqrt(14.0)
    rho = np.outer(state, state.conj())
    wavenumber = np.linspace(400.0, 1200.0, 101)
    unfiltered = _pathway_calculator(_SameVBlockBasis(), "unfiltered")
    pump_probe = _pathway_calculator(_SameVBlockBasis(), "pump_probe")

    unfiltered_spectrum = unfiltered.calculate(
        rho, wavenumber, method="loop", wavenumber_units=WAVENUMBER_UNITS
    )
    selected_spectrum = pump_probe.calculate(
        rho, wavenumber, method="loop", wavenumber_units=WAVENUMBER_UNITS
    )

    assert np.max(np.abs(unfiltered_spectrum - selected_spectrum)) > 1.0e-6
    assert unfiltered.last_calculation_report.phase_matching == "unfiltered"
    assert unfiltered.last_calculation_report.discarded_density_l2_fraction == 0.0


def test_phase_matching_mode_is_required_and_never_falls_back():
    args = (
        _ThreeLevelBasis(),
        _ThreeLevelHamiltonian(),
        _ThreeLevelDipole(),
        _spectroscopy_conditions(),
    )
    with pytest.raises(TypeError, match="phase_matching"):
        AbsorbanceCalculator(*args)
    with pytest.raises(ValueError, match="phase_matching must be"):
        AbsorbanceCalculator(*args, phase_matching="automatic")
    with pytest.raises(ValueError, match="requires basis.V_array"):
        _pathway_calculator(_BasisWithoutV(), "pump_probe")
    malformed_shape = _SameVBlockBasis()
    malformed_shape.V_array = np.array([0, 1])
    with pytest.raises(ValueError, match="V_array must have shape"):
        _pathway_calculator(malformed_shape, "pump_probe")
    nonfinite_labels = _SameVBlockBasis()
    nonfinite_labels.V_array = np.array([0.0, np.nan, 1.0])
    with pytest.raises(
        ValueError,
        match="finite vibrational quantum numbers",
    ):
        _pathway_calculator(nonfinite_labels, "pump_probe")


@pytest.mark.parametrize(
    "invalid_rho",
    [
        np.zeros((2, 2), dtype=np.complex128),
        np.full((3, 3), np.nan, dtype=np.complex128),
    ],
)
def test_density_matrix_is_strictly_validated(invalid_rho):
    calculator = _pathway_calculator(_ThreeLevelBasis(), "unfiltered")
    with pytest.raises(ValueError, match="rho must"):
        calculator.calculate(
            invalid_rho,
            np.linspace(400.0, 1200.0, 19),
            method="loop",
            wavenumber_units=WAVENUMBER_UNITS,
        )


def test_radiation_and_pfid_keep_post_probe_optical_coherence():
    calculator = _pathway_calculator(_ThreeLevelBasis(), "pump_probe")
    rho_after_probe = np.zeros((3, 3), dtype=np.complex128)
    rho_after_probe[1, 0] = 1.0
    wavenumber = np.linspace(450.0, 550.0, 101)

    # The same cross-V entry is excluded only from the pre-probe absorption input.
    absorption = calculator.calculate(
        rho_after_probe,
        wavenumber,
        method="loop",
        wavenumber_units=WAVENUMBER_UNITS,
    )
    radiation = calculator.calculate_radiation_spectrum(
        rho_after_probe,
        wavenumber,
        wavenumber_units=WAVENUMBER_UNITS,
    )
    pfid = calculator.calculate_pfid_spectrum(
        rho_after_probe,
        wavenumber,
        wavenumber_units=WAVENUMBER_UNITS,
    )

    np.testing.assert_allclose(absorption, 0.0)
    assert np.max(np.abs(radiation)) > 1.0e-6
    np.testing.assert_allclose(pfid, radiation)
