"""CPU-verifiable contracts for the device-native CuPy RK4 graph."""

from __future__ import annotations

import numpy as np
import pytest

try:
    import cupy as cp
except ImportError:  # pragma: no cover - optional GPU dependency
    cp = None

from rovibrational_excitation.dynamics.algorithms.rk4 import schrodinger_cupy
from rovibrational_excitation.dynamics.algorithms.rk4.schrodinger import (
    rk4_schrodinger,
)


class _NumPyCuPyDouble:
    complex128 = np.complex128
    float64 = np.float64
    asarray = staticmethod(np.asarray)
    empty = staticmethod(np.empty)
    isfinite = staticmethod(np.isfinite)
    sqrt = staticmethod(np.sqrt)
    vdot = staticmethod(np.vdot)


def _problem() -> tuple[np.ndarray, ...]:
    h0 = np.diag([0.1, 0.5, 1.2]).astype(np.complex128)
    mu_x = np.array(
        [[0.0, 0.7, 0.0], [0.7, 0.0, 0.4], [0.0, 0.4, 0.0]],
        dtype=np.complex128,
    )
    mu_y = np.array(
        [[0.0, -0.2j, 0.0], [0.2j, 0.0, -0.3j], [0.0, 0.3j, 0.0]],
        dtype=np.complex128,
    )
    field_x = np.array([0.2, -0.1, 0.4, 0.3, -0.2, 0.5, 0.1])
    field_y = np.array([-0.3, 0.2, 0.1, -0.4, 0.25, 0.15, -0.2])
    initial = np.array([np.sqrt(0.6), 0.2j, np.sqrt(0.36)], dtype=np.complex128)
    return h0, mu_x, mu_y, field_x, field_y, initial


@pytest.mark.parametrize(
    ("return_traj", "stride", "renorm"),
    [(True, 1, False), (True, 2, False), (True, 1, True), (False, 3, False)],
)
def test_cupy_graph_matches_the_cpu_rk4_contract_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
    return_traj: bool,
    stride: int,
    renorm: bool,
) -> None:
    problem = _problem()
    expected = rk4_schrodinger(
        *problem,
        dt=0.03,
        return_traj=return_traj,
        stride=stride,
        renorm=renorm,
        backend="numpy",
    )
    monkeypatch.setattr(schrodinger_cupy, "cp", _NumPyCuPyDouble)

    actual = schrodinger_cupy.rk4_schrodinger_cupy(
        *problem,
        dt=0.03,
        return_traj=return_traj,
        stride=stride,
        renorm=renorm,
    )

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=3.0e-16, atol=3.0e-16)


def test_cupy_graph_rejects_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(schrodinger_cupy, "cp", None)

    with pytest.raises(RuntimeError, match="CuPy is not installed"):
        schrodinger_cupy.rk4_schrodinger_cupy(
            *_problem(),
            dt=0.03,
            return_traj=False,
            stride=1,
            renorm=False,
        )


@pytest.mark.gpu
@pytest.mark.skipif(cp is None, reason="CuPy not available")
@pytest.mark.parametrize(
    ("return_traj", "stride", "renorm"),
    [(True, 1, False), (True, 2, True), (False, 3, False)],
)
def test_real_cupy_rk4_matches_cpu_and_stays_on_device(
    return_traj: bool,
    stride: int,
    renorm: bool,
) -> None:
    problem = _problem()
    expected = rk4_schrodinger(
        *problem,
        dt=0.03,
        return_traj=return_traj,
        stride=stride,
        renorm=renorm,
        backend="numpy",
    )

    actual = rk4_schrodinger(
        *problem,
        dt=0.03,
        return_traj=return_traj,
        stride=stride,
        renorm=renorm,
        backend="cupy",
    )

    assert isinstance(actual, cp.ndarray)
    assert actual.dtype == cp.complex128
    assert actual.shape == expected.shape
    np.testing.assert_allclose(cp.asnumpy(actual), expected, rtol=2.0e-13, atol=2.0e-13)
    if renorm:
        np.testing.assert_allclose(
            cp.asnumpy(cp.linalg.norm(actual, axis=1)),
            np.ones(actual.shape[0]),
            rtol=0.0,
            atol=2.0e-13,
        )
