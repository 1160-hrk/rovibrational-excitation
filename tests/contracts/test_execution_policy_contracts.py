"""Contracts for explicit backend/storage policy and capability preflight."""

import pytest

from rovibrational_excitation.core.execution import (
    ArrayBackend,
    ExecutionPolicy,
    MatrixStorage,
)
from rovibrational_excitation.core.propagation.capabilities import (
    PropagationAlgorithm,
    StatePath,
    UnsupportedExecutionPolicyError,
    validate_execution_capability,
)


def test_execution_policy_requires_typed_explicit_choices():
    with pytest.raises(TypeError):
        ExecutionPolicy()  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="ArrayBackend"):
        ExecutionPolicy(backend="numpy", storage=MatrixStorage.DENSE)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="MatrixStorage"):
        ExecutionPolicy(backend=ArrayBackend.NUMPY, storage="dense")  # type: ignore[arg-type]


def test_execution_policy_parses_strings_without_inference():
    policy = ExecutionPolicy.from_strings(backend="numpy", storage="csr")

    assert policy.backend is ArrayBackend.NUMPY
    assert policy.storage is MatrixStorage.CSR
    assert policy.sparse is True
    assert policy.dense is False

    with pytest.raises(ValueError, match="backend must be"):
        ExecutionPolicy.from_strings(backend="auto", storage="dense")
    with pytest.raises(ValueError, match="storage must be"):
        ExecutionPolicy.from_strings(backend="numpy", storage="auto")


@pytest.mark.parametrize(
    ("state_path", "algorithm", "backend", "storage", "supported"),
    [
        (state_path, algorithm, backend, storage, supported)
        for state_path in (StatePath.PURE, StatePath.INCOHERENT_ENSEMBLE)
        for algorithm in PropagationAlgorithm
        for backend in ArrayBackend
        for storage in MatrixStorage
        for supported in [
            not (backend is ArrayBackend.CUPY and storage is MatrixStorage.CSR)
        ]
    ]
    + [
        (
            StatePath.DENSITY,
            algorithm,
            backend,
            storage,
            algorithm is PropagationAlgorithm.RK4
            and backend is ArrayBackend.NUMPY
            and storage is MatrixStorage.DENSE,
        )
        for algorithm in PropagationAlgorithm
        for backend in ArrayBackend
        for storage in MatrixStorage
    ],
)
def test_capability_matrix_is_explicit(
    state_path, algorithm, backend, storage, supported
):
    policy = ExecutionPolicy(backend=backend, storage=storage)

    if supported:
        validate_execution_capability(
            state_path=state_path,
            algorithm=algorithm,
            policy=policy,
            cupy_available=True,
        )
    else:
        with pytest.raises(UnsupportedExecutionPolicyError):
            validate_execution_capability(
                state_path=state_path,
                algorithm=algorithm,
                policy=policy,
                cupy_available=True,
            )


def test_unsupported_cupy_csr_is_rejected_before_availability_check():
    policy = ExecutionPolicy(
        backend=ArrayBackend.CUPY,
        storage=MatrixStorage.CSR,
    )

    with pytest.raises(UnsupportedExecutionPolicyError, match="CuPy CSR"):
        validate_execution_capability(
            state_path=StatePath.PURE,
            algorithm=PropagationAlgorithm.RK4,
            policy=policy,
            cupy_available=False,
        )


def test_unavailable_cupy_never_falls_back_to_numpy():
    policy = ExecutionPolicy(
        backend=ArrayBackend.CUPY,
        storage=MatrixStorage.DENSE,
    )

    with pytest.raises(RuntimeError, match="CuPy backend requested"):
        validate_execution_capability(
            state_path=StatePath.PURE,
            algorithm=PropagationAlgorithm.RK4,
            policy=policy,
            cupy_available=False,
        )
