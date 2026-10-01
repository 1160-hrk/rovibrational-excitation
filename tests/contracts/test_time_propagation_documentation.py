"""Contracts for the current time-propagation guide."""

from __future__ import annotations

from pathlib import Path

import pytest

from rovibrational_excitation.core.execution import (
    ArrayBackend,
    ExecutionPolicy,
    MatrixStorage,
)
from rovibrational_excitation.dynamics.capabilities import (
    PropagationAlgorithm,
    StatePath,
    validate_execution_capability,
)

ROOT = Path(__file__).resolve().parents[2]
GUIDE = ROOT / "docs" / "TIME_PROPAGATION.md"


def _guide() -> str:
    return GUIDE.read_text()


@pytest.mark.parametrize(
    ("state_path", "algorithm", "backend", "storage"),
    [
        (
            StatePath.PURE,
            PropagationAlgorithm.RK4,
            ArrayBackend.NUMPY,
            MatrixStorage.DENSE,
        ),
        (
            StatePath.PURE,
            PropagationAlgorithm.RK4,
            ArrayBackend.NUMPY,
            MatrixStorage.CSR,
        ),
        (
            StatePath.PURE,
            PropagationAlgorithm.SPLIT_OPERATOR,
            ArrayBackend.NUMPY,
            MatrixStorage.DENSE,
        ),
        (
            StatePath.INCOHERENT_ENSEMBLE,
            PropagationAlgorithm.SPLIT_OPERATOR,
            ArrayBackend.NUMPY,
            MatrixStorage.CSR,
        ),
    ],
)
def test_documented_cpu_capability_rows_are_accepted(
    state_path: StatePath,
    algorithm: PropagationAlgorithm,
    backend: ArrayBackend,
    storage: MatrixStorage,
) -> None:
    validate_execution_capability(
        state_path=state_path,
        algorithm=algorithm,
        policy=ExecutionPolicy(backend=backend, storage=storage),
        cupy_available=False,
    )


@pytest.mark.parametrize(
    ("state_path", "algorithm", "backend", "storage", "message"),
    [
        (
            StatePath.DENSITY,
            PropagationAlgorithm.SPLIT_OPERATOR,
            ArrayBackend.NUMPY,
            MatrixStorage.DENSE,
            "only algorithm='rk4'",
        ),
        (
            StatePath.DENSITY,
            PropagationAlgorithm.RK4,
            ArrayBackend.CUPY,
            MatrixStorage.DENSE,
            "only backend='numpy'",
        ),
        (
            StatePath.DENSITY,
            PropagationAlgorithm.RK4,
            ArrayBackend.NUMPY,
            MatrixStorage.CSR,
            "only dense storage",
        ),
        (
            StatePath.PURE,
            PropagationAlgorithm.RK4,
            ArrayBackend.CUPY,
            MatrixStorage.CSR,
            "CuPy CSR",
        ),
    ],
)
def test_documented_unsupported_capabilities_raise(
    state_path: StatePath,
    algorithm: PropagationAlgorithm,
    backend: ArrayBackend,
    storage: MatrixStorage,
    message: str,
) -> None:
    with pytest.raises((ValueError, RuntimeError), match=message):
        validate_execution_capability(
            state_path=state_path,
            algorithm=algorithm,
            policy=ExecutionPolicy(backend=backend, storage=storage),
            cupy_available=True,
        )


def test_guide_records_frozen_timing_and_output_contracts() -> None:
    text = _guide()

    assert r"propagation\_dt\_fs}=2" in text
    assert "$2N+1$" in text
    assert "sample_stride" in text
    assert "出力間引きだけ" in text
    assert "正確な終了点" in text
    assert "resample は行わない" in text


def test_guide_distinguishes_exact_cartesian_and_explicit_approximation() -> None:
    text = _guide()

    assert "`cartesian`（標準・厳密な物理 Hamiltonian）" in text
    assert "`helicity_projected`（明示的な近似）" in text
    assert 'split_interaction="helicity_projected"' in text
    assert "CSR operator を渡すことはできるが" in text
    assert "実空間 FFT split、4次 Suzuki split、adaptive split は実装していない" in text


def test_guide_records_cuda_acceptance_without_silent_repair() -> None:
    text = _guide()

    assert "実 CUDA 数値受入れ" in text
    assert "NumPy へ fallback しない" in text
    assert "入力を規格化・対称化・clip しない" in text
    assert "GPU test がskipされたことだけを" in text
    assert "CUDA対応の検証根拠にはしない" in text
