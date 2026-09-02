"""Strict construction and execution contracts for the production SymTop model."""

from __future__ import annotations

import numpy as np
import pytest

from rovibrational_excitation.core.execution import (
    ArrayBackend,
    ExecutionPolicy,
    MatrixStorage,
)
from rovibrational_excitation.models import build_model
from rovibrational_excitation.models.parameters import SymmetricTopParameters
from rovibrational_excitation.models.validation import (
    ModelConfigurationError,
    validate_model_parameters,
)
from rovibrational_excitation.simulation.optimize_runner import _build_basis
from rovibrational_excitation.simulation.runner import _run_one
from rovibrational_excitation.simulation.validation import (
    SimulationConfigurationError,
)


def _model_params(**overrides) -> dict:
    params = {
        "basis_type": "symtop",
        "molecule": "CH3F",
        "nuclear_spin_isomer": "ortho",
        "axes": "xy",
        "V_max": 1,
        "J_max": 1,
        "vibrational_frequency": 0.37,
        "vibrational_frequency_units": "rad/fs",
        "anharmonic_shift": 0.015,
        "anharmonic_shift_units": "rad/fs",
        "rotational_constant_perpendicular": 0.004,
        "rotational_constant_perpendicular_units": "rad/fs",
        "rotational_constant_parallel": 0.006,
        "rotational_constant_parallel_units": "rad/fs",
        "vibration_rotation_coupling_perpendicular": 0.0002,
        "vibration_rotation_coupling_perpendicular_units": "rad/fs",
        "vibration_rotation_coupling_parallel": 0.0003,
        "vibration_rotation_coupling_parallel_units": "rad/fs",
        "dipole_scale": 2.0e-29,
        "dipole_scale_units": "C*m",
        "potential_type": "harmonic",
        "initial_states": [0],
    }
    params.update(overrides)
    return params


def _simulation_params(**overrides) -> dict:
    params = {
        **_model_params(),
        "t_start": 0.0,
        "t_start_units": "fs",
        "t_end": 0.04,
        "t_end_units": "fs",
        "dt": 0.001,
        "dt_units": "fs",
        "duration": 0.03,
        "duration_units": "fs",
        "t_center": 0.02,
        "t_center_units": "fs",
        "envelope_kind": "gaussian_fwhm",
        "modulation_kind": "none",
        "carrier_frequency": 0.37,
        "carrier_frequency_units": "rad/fs",
        "amplitude": 4.0e8,
        "amplitude_units": "V/m",
        "polarization": [1.0, 0.0],
        "backend": "numpy",
        "algorithm": "rk4",
        "storage": "dense",
        "return_traj": True,
        "sample_stride": 1,
        "nondimensional": False,
        "renorm": False,
        "validate_units": False,
        "save": False,
    }
    params.update(overrides)
    return params


def test_named_molecule_builds_new_symtop_model_with_canonical_metadata() -> None:
    policy = ExecutionPolicy(ArrayBackend.NUMPY, MatrixStorage.DENSE)
    components = build_model(
        _model_params(molecule="methyl fluoride"), execution_policy=policy
    )

    assert isinstance(
        SymmetricTopParameters.from_mapping(_model_params()),
        SymmetricTopParameters,
    )
    assert components.name == "symtop"
    assert components.basis.quantum_number_order == ("v", "J", "K", "M")
    assert components.to_system_model().metadata["canonical_id"] == "CH3F"
    assert components.to_system_model().metadata["nuclear_spin_isomer"] == "ortho"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"molecule": "unknown"}, "Unknown molecule symmetry preset"),
        ({"molecule": "H2"}, "requires a symmetric_top molecule preset"),
        ({"nuclear_spin_isomer": []}, "must be a string"),
        ({"nuclear_spin_isomer": "all"}, "must be ortho or para"),
        ({"nuclear_spin_isomer": "invalid"}, "must be ortho or para"),
    ],
)
def test_symtop_rejects_unresolved_or_inapplicable_symmetry(overrides, message) -> None:
    with pytest.raises(ModelConfigurationError, match=message):
        validate_model_parameters(_model_params(**overrides))


@pytest.mark.parametrize(
    "missing",
    [
        "molecule",
        "nuclear_spin_isomer",
        "rotational_constant_perpendicular",
        "rotational_constant_perpendicular_units",
        "rotational_constant_parallel",
        "rotational_constant_parallel_units",
        "vibration_rotation_coupling_perpendicular",
        "vibration_rotation_coupling_perpendicular_units",
        "vibration_rotation_coupling_parallel",
        "vibration_rotation_coupling_parallel_units",
    ],
)
def test_symtop_physics_inputs_are_required_before_allocation(missing: str) -> None:
    params = _model_params()
    params.pop(missing)
    with pytest.raises(
        ModelConfigurationError, match="Missing required model parameters"
    ):
        validate_model_parameters(params)


def test_morse_bound_levels_fail_during_model_construction() -> None:
    params = _model_params(
        potential_type="morse",
        vibrational_frequency=1.0,
        anharmonic_shift=0.1,
        V_max=10,
    )
    policy = ExecutionPolicy(ArrayBackend.NUMPY, MatrixStorage.DENSE)
    with pytest.raises(ValueError, match="V_max=10 exceeds the Morse limit 9"):
        build_model(params, execution_policy=policy)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"backend": "cupy"}, "SymTop currently supports only backend='numpy'"),
        (
            {"algorithm": "split_operator", "split_interaction": "cartesian"},
            "SymTop currently supports only algorithm='rk4'",
        ),
    ],
)
def test_unverified_symtop_execution_routes_raise_explicitly(
    overrides, message
) -> None:
    with pytest.raises(SimulationConfigurationError, match=message):
        _run_one(_simulation_params(**overrides))


def test_dense_and_csr_rk4_populations_agree() -> None:
    dense = _run_one(_simulation_params(storage="dense"))
    csr = _run_one(_simulation_params(storage="csr"))

    assert dense.shape == csr.shape
    np.testing.assert_allclose(csr, dense, rtol=0.0, atol=2.0e-14)


def test_dimensional_and_nondimensional_symtop_rk4_populations_agree() -> None:
    dimensional = _run_one(_simulation_params(nondimensional=False))
    nondimensional = _run_one(_simulation_params(nondimensional=True))

    np.testing.assert_allclose(
        nondimensional,
        dimensional,
        rtol=0.0,
        atol=2.0e-12,
    )


def test_optimization_rejects_legacy_symtop_before_basis_construction() -> None:
    with pytest.raises(
        ValueError,
        match="SymTop optimization is not supported yet.*normal simulation runner",
    ):
        _build_basis({"type": "symtop", "params": {}})
