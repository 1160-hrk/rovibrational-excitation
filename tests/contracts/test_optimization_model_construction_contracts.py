"""Shared production/optimization model-construction contracts."""

from __future__ import annotations

import numpy as np
import pytest

from rovibrational_excitation.core.basis import (
    LinMolBasis,
    TwoLevelBasis,
    VibLadderBasis,
)
from rovibrational_excitation.dipole.factory import create_dipole_matrix
from rovibrational_excitation.optimization.model import (
    OptimizationModelConfigurationError,
    build_optimization_model,
    validate_optimization_state,
)


def _dense(value):
    return value.toarray() if hasattr(value, "toarray") else np.asarray(value)


def _assert_legacy_operator_parity(legacy_basis, legacy_dipole, actual) -> None:
    np.testing.assert_array_equal(actual.basis.basis, legacy_basis.basis)
    np.testing.assert_allclose(
        actual.hamiltonian.get_matrix("rad/fs"),
        legacy_basis.generate_H0().get_matrix("rad/fs"),
        rtol=2.0e-16,
        atol=0.0,
    )
    for axis in "xyz":
        np.testing.assert_allclose(
            _dense(actual.dipole.get_mu_in_units(axis, "C*m")),
            _dense(legacy_dipole.get_mu_in_units(axis, "C*m")),
            rtol=5.0e-16,
            atol=0.0,
        )


def test_vibladder_optimization_uses_production_frozen_schema_with_parity() -> None:
    legacy_basis = VibLadderBasis(
        V_max=3,
        omega=2349.1,
        delta_omega=25.0,
        input_units="cm^-1",
        output_units="rad/fs",
    )
    legacy_dipole = create_dipole_matrix(
        legacy_basis,
        mu0=0.3,
        potential_type="harmonic",
        backend="numpy",
        dense=False,
        units="D",
        units_input="D",
    )
    actual = build_optimization_model(
        {
            "type": "vibladder",
            "params": {
                "V_max": 3,
                "vibrational_frequency": 2349.1,
                "vibrational_frequency_units": "cm^-1",
                "anharmonic_shift": 25.0,
                "anharmonic_shift_units": "cm^-1",
                "dipole_scale": 0.3,
                "dipole_scale_units": "D",
                "potential_type": "harmonic",
            },
        }
    )
    _assert_legacy_operator_parity(legacy_basis, legacy_dipole, actual)


def test_linmol_optimization_uses_m_resolved_production_order_with_parity() -> None:
    legacy_basis = LinMolBasis(
        V_max=1,
        J_max=2,
        use_M=True,
        omega=2349.1,
        delta_omega=25.0,
        B=0.39,
        alpha=0.0037,
        input_units="cm^-1",
        output_units="rad/fs",
    )
    legacy_dipole = create_dipole_matrix(
        legacy_basis,
        mu0=0.3,
        potential_type="harmonic",
        backend="numpy",
        dense=False,
        units="D",
        units_input="D",
    )
    actual = build_optimization_model(
        {
            "type": "linmol",
            "params": {
                "V_max": 1,
                "J_max": 2,
                "representation": "m_resolved",
                "vibrational_frequency": 2349.1,
                "vibrational_frequency_units": "cm^-1",
                "anharmonic_shift": 25.0,
                "anharmonic_shift_units": "cm^-1",
                "rotational_constant": 0.39,
                "rotational_constant_units": "cm^-1",
                "vibration_rotation_coupling": 0.0037,
                "vibration_rotation_coupling_units": "cm^-1",
                "dipole_scale": 0.3,
                "dipole_scale_units": "D",
                "potential_type": "harmonic",
            },
        }
    )
    _assert_legacy_operator_parity(legacy_basis, legacy_dipole, actual)


def test_twolevel_optimization_uses_production_frozen_schema_with_parity() -> None:
    legacy_basis = TwoLevelBasis(
        energy_gap=2300.0,
        input_units="cm^-1",
        output_units="rad/fs",
    )
    legacy_dipole = create_dipole_matrix(
        legacy_basis,
        mu0=0.3,
        backend="numpy",
        dense=False,
        units="D",
        units_input="D",
    )
    actual = build_optimization_model(
        {
            "type": "twolevel",
            "params": {
                "energy_gap": 2300.0,
                "energy_gap_units": "cm^-1",
                "dipole_scale": 0.3,
                "dipole_scale_units": "D",
            },
        }
    )
    _assert_legacy_operator_parity(legacy_basis, legacy_dipole, actual)


@pytest.mark.parametrize(
    ("system", "match"),
    [
        (
            {"type": "viblad", "params": {}},
            "system.type=viblad was removed.*vibladder",
        ),
        (
            {"type": "vibladder", "params": {"omega_cm": 1.0}},
            "omega_cm was removed.*vibrational_frequency",
        ),
        (
            {"type": "twolevel", "params": {"potential_type": "harmonic"}},
            "not applicable.*potential_type",
        ),
        (
            {"type": "twolevel", "params": {"mystery": 1}},
            "Unknown optimization model parameters: mystery",
        ),
    ],
)
def test_optimization_model_schema_rejects_legacy_and_unknown_inputs(
    system,
    match,
) -> None:
    with pytest.raises(OptimizationModelConfigurationError, match=match):
        build_optimization_model(system)


def test_optimization_rejects_m_incoherent_average_until_multiblock_support() -> None:
    with pytest.raises(
        OptimizationModelConfigurationError,
        match="requires representation=m_resolved.*separate multi-block",
    ):
        build_optimization_model(
            {
                "type": "linmol",
                "params": {
                    "representation": "m_incoherent_average",
                    "V_max": 0,
                    "J_max": 0,
                },
            }
        )


def test_optimization_state_validation_never_adds_or_drops_m() -> None:
    basis = LinMolBasis(
        V_max=0,
        J_max=0,
        use_M=True,
        omega=1.0,
        delta_omega=0.0,
        B=0.1,
        alpha=0.0,
    )
    assert validate_optimization_state(basis, [0, 0, 0], label="initial") == (
        0,
        0,
        0,
    )
    with pytest.raises(
        OptimizationModelConfigurationError,
        match=r"states.initial=\(0, 0\) is not present",
    ):
        validate_optimization_state(basis, [0, 0], label="initial")
