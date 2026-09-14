"""Simulation contracts for one explicit construction/propagation policy."""

import numpy as np
import pytest
import scipy.sparse as sp

from rovibrational_excitation.core.basis import VibLadderBasis
from rovibrational_excitation.core.execution import (
    ArrayBackend,
    ExecutionPolicy,
    MatrixStorage,
)
from rovibrational_excitation.dipole.factory import create_dipole_matrix
from rovibrational_excitation.dynamics.capabilities import (
    PropagationAlgorithm,
)
from rovibrational_excitation.models import build_model
from rovibrational_excitation.models.two_level import TwoLevelBasis
from rovibrational_excitation.simulation.validation import (
    SimulationConfigurationError,
    validate_simulation_case,
)


def _twolevel_case(**overrides):
    params = {
        "basis_type": "twolevel",
        "energy_gap": 0.2,
        "energy_gap_units": "rad/fs",
        "dipole_scale": 3.0e-30,
        "dipole_scale_units": "C*m",
        "t_start": -0.5,
        "t_start_units": "fs",
        "t_end": 0.5,
        "t_end_units": "fs",
        "dt": 0.05,
        "dt_units": "fs",
        "duration": 0.3,
        "duration_units": "fs",
        "t_center": 0.0,
        "t_center_units": "fs",
        "envelope_kind": "gaussian_fwhm",
        "modulation_kind": "none",
        "carrier_frequency": 0.1,
        "carrier_frequency_units": "PHz",
        "amplitude": 1.0e8,
        "amplitude_units": "V/m",
        "initial_states": [0],
        "backend": "numpy",
        "storage": "dense",
        "algorithm": "rk4",
        "return_traj": True,
        "sample_stride": 1,
        "nondimensional": False,
        "renorm": False,
    }
    params.update(overrides)
    return params


@pytest.mark.parametrize(
    "missing",
    [
        "backend",
        "storage",
        "algorithm",
        "return_traj",
        "sample_stride",
        "nondimensional",
        "renorm",
    ],
)
def test_simulation_requires_each_execution_choice(missing):
    params = _twolevel_case()
    del params[missing]

    with pytest.raises(SimulationConfigurationError, match=missing):
        validate_simulation_case(params)


@pytest.mark.parametrize("removed", ["dense", "sparse"])
def test_simulation_rejects_legacy_storage_booleans(removed):
    params = _twolevel_case(**{removed: True})

    with pytest.raises(SimulationConfigurationError, match="use required storage"):
        validate_simulation_case(params)


def test_validation_returns_the_only_execution_choices_used_by_runner():
    options = validate_simulation_case(
        _twolevel_case(storage="csr", algorithm="split_operator")
    )

    assert options.execution == ExecutionPolicy(
        backend=ArrayBackend.NUMPY,
        storage=MatrixStorage.CSR,
    )
    assert options.algorithm is PropagationAlgorithm.SPLIT_OPERATOR
    assert options.return_trajectory is True
    assert options.sample_stride == 1
    assert options.nondimensional is False
    assert options.renorm is False


def test_structurally_unsupported_policy_fails_during_simulation_preflight():
    with pytest.raises(SimulationConfigurationError, match="CuPy CSR"):
        validate_simulation_case(_twolevel_case(backend="cupy", storage="csr"))


@pytest.mark.parametrize(
    "params",
    [
        {
            "basis_type": "twolevel",
            "energy_gap": 1.0,
            "energy_gap_units": "rad/fs",
            "dipole_scale": 2.0e-30,
            "dipole_scale_units": "C*m",
            "initial_states": [0],
        },
        {
            "basis_type": "vibladder",
            "V_max": 3,
            "vibrational_frequency": 1.0,
            "vibrational_frequency_units": "rad/fs",
            "anharmonic_shift": 0.01,
            "anharmonic_shift_units": "rad/fs",
            "potential_type": "harmonic",
            "dipole_scale": 2.0e-30,
            "dipole_scale_units": "C*m",
            "initial_states": [0],
        },
    ],
)
def test_scalar_model_numpy_csr_is_real_csr_with_dense_element_parity(params):
    dense_model = build_model(
        params,
        execution_policy=ExecutionPolicy(
            backend=ArrayBackend.NUMPY,
            storage=MatrixStorage.DENSE,
        ),
    )
    csr_model = build_model(
        params,
        execution_policy=ExecutionPolicy(
            backend=ArrayBackend.NUMPY,
            storage=MatrixStorage.CSR,
        ),
    )

    axis = csr_model.coupling.scalar_axis
    assert axis is not None
    axis = axis.value
    dense_mu = dense_model.dipole.mu(axis)
    csr_mu = csr_model.dipole.mu(axis)

    assert sp.isspmatrix_csr(csr_mu)
    np.testing.assert_array_equal(csr_mu.toarray(), dense_mu)
    assert csr_model.dipole.backend == "numpy"
    assert csr_model.dipole.dense is False


def test_generic_dipole_factory_preserves_scalar_model_csr_choice():
    cases = [
        (
            TwoLevelBasis(energy_gap=1.0, input_units="rad/fs", output_units="J"),
            {},
            "x",
        ),
        (
            VibLadderBasis(V_max=3, omega=1.0, delta_omega=0.01),
            {"potential_type": "harmonic"},
            "z",
        ),
    ]

    for basis, extra, axis in cases:
        dipole = create_dipole_matrix(basis, mu0=2.0e-30, dense=False, **extra)

        assert dipole.dense is False
        assert sp.isspmatrix_csr(dipole.mu(axis))


def test_scalar_dipole_rejects_cupy_csr_before_backend_allocation():
    dipole = create_dipole_matrix(
        TwoLevelBasis(energy_gap=1.0, input_units="rad/fs", output_units="J"),
        mu0=2.0e-30,
        backend="cupy",
        dense=False,
    )

    with pytest.raises(ValueError, match="CuPy CSR"):
        dipole.mu("x")
