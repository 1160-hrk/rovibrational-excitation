"""Contracts for explicit spectroscopy projection and standard detection."""

from __future__ import annotations

import numpy as np
import pytest

from rovibrational_excitation.core.model import (
    Axis,
    CouplingSpec,
    SystemModel,
)
from rovibrational_excitation.core.units.constants import CONSTANTS
from rovibrational_excitation.models.two_level import (
    TwoLevelBasis,
    TwoLevelDipoleMatrix,
)
from rovibrational_excitation.spectroscopy import (
    AbsorbanceCalculator,
    CartesianAnalyzerProjection,
    CartesianProjection,
    ComplexResponseSpectrum,
    ExperimentalConditions,
)


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


def _two_level_model(*, coupling: CouplingSpec) -> SystemModel:
    basis = TwoLevelBasis(
        energy_gap=1.0e-20,
        input_units="J",
        output_units="J",
    )
    return SystemModel(
        name="twolevel-test",
        basis=basis,
        hamiltonian=basis.generate_H0(),
        dipole=TwoLevelDipoleMatrix(basis, mu0=1.0e-30),
        coupling=coupling,
        metadata={},
    )


def test_scalar_standard_absorption_is_bitwise_identical_to_typed_direct() -> None:
    model = _two_level_model(coupling=CouplingSpec.scalar(Axis.X))
    direct = AbsorbanceCalculator(
        model.basis,
        model.hamiltonian,
        model.dipole,
        _conditions(),
        phase_matching="unfiltered",
        projection=CartesianProjection(
            axes=(Axis.X,),
            interaction=np.array([1.0]),
        ),
    )
    standard = AbsorbanceCalculator.standard_absorption(
        model,
        _conditions(),
        phase_matching="unfiltered",
    )
    rho = np.diag([1.0, 0.0]).astype(np.complex128)
    wavenumber = np.linspace(400.0, 600.0, 17)

    assert standard.axes == "x"
    assert np.array_equal(standard.mu_int, direct.mu_int)
    assert np.array_equal(standard.mu_det, direct.mu_det)
    for method, options in (
        ("loop", {}),
        ("matrix", {}),
        ("2d", {}),
        ("chunked", {"chunk_size": 5}),
    ):
        expected = direct.calculate(
            rho,
            wavenumber,
            method=method,
            wavenumber_units="cm^-1",
            **options,
        )
        actual = standard.calculate(
            rho,
            wavenumber,
            method=method,
            wavenumber_units="cm^-1",
            **options,
        )
        assert np.array_equal(actual, expected), method


def test_cartesian_standard_absorption_uses_analyzer_bra_of_probe_ket() -> None:
    model = _two_level_model(coupling=CouplingSpec.cartesian("xy"))
    probe = CartesianProjection(
        axes=(Axis.X, Axis.Y),
        interaction=np.array([1.0, 1.0j]) / np.sqrt(2.0),
    )

    calculator = AbsorbanceCalculator.standard_absorption(
        model,
        _conditions(),
        phase_matching="unfiltered",
        projection=probe,
    )

    assert calculator.axes == "xy"
    assert calculator.projection is probe
    np.testing.assert_allclose(calculator.mu_det, calculator.mu_int.conj().T)


def test_projection_is_conditional_on_model_coupling() -> None:
    scalar = _two_level_model(coupling=CouplingSpec.scalar(Axis.X))
    cartesian = _two_level_model(coupling=CouplingSpec.cartesian("xy"))
    probe = CartesianProjection(
        axes=(Axis.X, Axis.Y),
        interaction=np.array([1.0, 0.0]),
    )

    with pytest.raises(ValueError, match="not applicable to scalar"):
        AbsorbanceCalculator.standard_absorption(
            scalar,
            _conditions(),
            phase_matching="unfiltered",
            projection=probe,
        )
    with pytest.raises(ValueError, match="required for Cartesian"):
        AbsorbanceCalculator.standard_absorption(
            cartesian,
            _conditions(),
            phase_matching="unfiltered",
        )


def test_direct_constructor_requires_one_typed_projection() -> None:
    model = _two_level_model(coupling=CouplingSpec.scalar(Axis.X))
    with pytest.raises(TypeError, match="projection"):
        AbsorbanceCalculator(
            model.basis,
            model.hamiltonian,
            model.dipole,
            _conditions(),
            phase_matching="unfiltered",
        )
    with pytest.raises(TypeError, match="projection must be"):
        AbsorbanceCalculator(
            model.basis,
            model.hamiltonian,
            model.dipole,
            _conditions(),
            phase_matching="unfiltered",
            projection=np.array([1.0]),  # type: ignore[arg-type]
        )


def test_cartesian_projection_requires_typed_unique_axes_and_finite_nonzero_ket() -> (
    None
):
    with pytest.raises(ValueError, match="unique"):
        CartesianProjection(
            axes=(Axis.X, Axis.X),
            interaction=np.array([1.0, 0.0]),
        )
    with pytest.raises(ValueError, match="shape"):
        CartesianProjection(
            axes=(Axis.X, Axis.Y),
            interaction=np.array([1.0]),
        )
    with pytest.raises(ValueError, match="finite nonzero"):
        CartesianProjection(
            axes=(Axis.X, Axis.Y),
            interaction=np.array([0.0, 0.0]),
        )
    with pytest.raises(TypeError, match="typed Axis"):
        CartesianProjection(
            axes=("x", "y"),  # type: ignore[arg-type]
            interaction=np.array([1.0, 0.0]),
        )


def test_complex_response_is_a_typed_immutable_snapshot() -> None:
    model = _two_level_model(coupling=CouplingSpec.scalar(Axis.X))
    calculator = AbsorbanceCalculator.standard_absorption(
        model,
        _conditions(),
        phase_matching="unfiltered",
    )
    rho = np.diag([1.0, 0.0]).astype(np.complex128)
    wavenumber = np.linspace(400.0, 600.0, 17)

    result = calculator.calculate_complex_response(
        rho,
        wavenumber,
        method="loop",
        wavenumber_units="cm^-1",
    )

    assert isinstance(result, ComplexResponseSpectrum)
    np.testing.assert_array_equal(result.wavenumber_cm_inverse, wavenumber)
    assert np.iscomplexobj(result.molecular_response_c2_m2_per_j)
    assert result.calculation_report is calculator.last_calculation_report
    assert not result.wavenumber_cm_inverse.flags.writeable
    assert not result.molecular_response_c2_m2_per_j.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        result.molecular_response_c2_m2_per_j[0] = 0.0


def test_complex_response_returns_each_existing_route_before_mod_conversion() -> None:
    model = _two_level_model(coupling=CouplingSpec.scalar(Axis.X))
    calculator = AbsorbanceCalculator.standard_absorption(
        model,
        _conditions(),
        phase_matching="unfiltered",
    )
    rho = np.diag([1.0, 0.0]).astype(np.complex128)
    wavenumber = np.linspace(400.0, 600.0, 17)

    route_calls = (
        ("loop", {}, lambda: calculator._calculate_loop(rho, wavenumber)),
        ("matrix", {}, lambda: calculator._calculate_matrix(rho, wavenumber)),
        ("2d", {}, lambda: calculator._calculate_2d(rho, wavenumber)),
        (
            "chunked",
            {"chunk_size": 5},
            lambda: calculator._calculate_chunked(rho, wavenumber, chunk_size=5),
        ),
    )
    for method, options, raw_route in route_calls:
        _omega, expected = raw_route()
        actual = calculator.calculate_complex_response(
            rho,
            wavenumber,
            method=method,
            wavenumber_units="cm^-1",
            **options,
        )
        np.testing.assert_array_equal(
            actual.molecular_response_c2_m2_per_j,
            expected,
            err_msg=method,
        )
        assert actual.calculation_report.executed_method == method
        assert actual.calculation_report.device_function_applied is False


def test_typed_analyzer_returns_complex_response_and_rejects_mod() -> None:
    model = _two_level_model(coupling=CouplingSpec.cartesian("xy"))
    projection = CartesianAnalyzerProjection(
        axes=(Axis.X, Axis.Y),
        interaction=np.array([1.0, 1.0j]),
        analyzer=np.array([1.0, 0.0]),
    )
    calculator = AbsorbanceCalculator.analyzer_complex_response(
        model,
        _conditions(),
        phase_matching="unfiltered",
        projection=projection,
    )
    rho = np.diag([1.0, 0.0]).astype(np.complex128)
    wavenumber = np.linspace(400.0, 600.0, 17)

    response = calculator.calculate_complex_response(
        rho,
        wavenumber,
        method="loop",
        wavenumber_units="cm^-1",
    )

    assert isinstance(response, ComplexResponseSpectrum)
    assert calculator.projection is projection
    with pytest.raises(ValueError, match="analyzer.*complex response"):
        calculator.calculate(
            rho,
            wavenumber,
            method="loop",
            wavenumber_units="cm^-1",
        )
    with pytest.raises(ValueError, match="analyzer complex response"):
        calculator.calculate_radiation_spectrum(
            rho,
            wavenumber,
            wavenumber_units="cm^-1",
        )
    with pytest.raises(ValueError, match="analyzer complex response"):
        calculator.calculate_pfid_spectrum(
            rho,
            wavenumber,
            wavenumber_units="cm^-1",
        )


def test_analyzer_projection_is_strict_and_model_coupling_conditional() -> None:
    scalar = _two_level_model(coupling=CouplingSpec.scalar(Axis.X))
    cartesian = _two_level_model(coupling=CouplingSpec.cartesian("xy"))
    analyzer = CartesianAnalyzerProjection(
        axes=(Axis.X, Axis.Y),
        interaction=np.array([1.0, 0.0]),
        analyzer=np.array([0.0, 1.0]),
    )
    mismatched = CartesianAnalyzerProjection(
        axes=(Axis.X, Axis.Z),
        interaction=np.array([1.0, 0.0]),
        analyzer=np.array([0.0, 1.0]),
    )

    assert not analyzer.interaction.flags.writeable
    assert not analyzer.analyzer.flags.writeable
    with pytest.raises(ValueError, match="not applicable to scalar"):
        AbsorbanceCalculator.analyzer_complex_response(
            scalar,
            _conditions(),
            phase_matching="unfiltered",
            projection=analyzer,
        )
    with pytest.raises(ValueError, match="must exactly match model coupling axes"):
        AbsorbanceCalculator.analyzer_complex_response(
            cartesian,
            _conditions(),
            phase_matching="unfiltered",
            projection=mismatched,
        )
    with pytest.raises(ValueError, match="must exactly match model coupling axes"):
        AbsorbanceCalculator.standard_absorption(
            cartesian,
            _conditions(),
            phase_matching="unfiltered",
            projection=CartesianProjection(
                axes=(Axis.X, Axis.Z),
                interaction=np.array([1.0, 0.0]),
            ),
        )
