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
    CartesianProjection,
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


def test_scalar_standard_absorption_is_bitwise_identical_to_explicit_legacy() -> None:
    model = _two_level_model(coupling=CouplingSpec.scalar(Axis.X))
    legacy = AbsorbanceCalculator(
        model.basis,
        model.hamiltonian,
        model.dipole,
        _conditions(),
        phase_matching="unfiltered",
        axes="x",
        pol_int=np.array([1.0]),
        pol_det=np.array([1.0]),
    )
    standard = AbsorbanceCalculator.standard_absorption(
        model,
        _conditions(),
        phase_matching="unfiltered",
    )
    rho = np.diag([1.0, 0.0]).astype(np.complex128)
    wavenumber = np.linspace(400.0, 600.0, 17)

    assert standard.axes == "x"
    assert np.array_equal(standard.mu_int, legacy.mu_int)
    assert np.array_equal(standard.mu_det, legacy.mu_det)
    for method, options in (
        ("loop", {}),
        ("matrix", {}),
        ("2d", {}),
        ("chunked", {"chunk_size": 5}),
    ):
        expected = legacy.calculate(
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
    np.testing.assert_array_equal(calculator.pol_int, probe.interaction)
    np.testing.assert_array_equal(calculator.pol_det, probe.interaction)
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
