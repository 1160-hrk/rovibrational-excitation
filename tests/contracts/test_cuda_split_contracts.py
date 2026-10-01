"""CPU-verifiable and real-GPU contracts for device-native split propagation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from rovibrational_excitation.dynamics.algorithms.split_operator import (
    schrodinger_cupy,
)
from rovibrational_excitation.dynamics.algorithms.split_operator.schrodinger import (
    splitop_schrodinger,
)

try:
    import cupy as cp
except ImportError:  # pragma: no cover - optional GPU dependency
    cp = None


class _NumPyCuPyDouble:
    complex128 = np.complex128
    float64 = np.float64
    inf = np.inf
    pi = np.pi
    linalg = np.linalg
    abs = staticmethod(np.abs)
    all = staticmethod(np.all)
    any = staticmethod(np.any)
    arctan2 = staticmethod(np.arctan2)
    argmax = staticmethod(np.argmax)
    asarray = staticmethod(np.asarray)
    column_stack = staticmethod(np.column_stack)
    diag = staticmethod(np.diag)
    empty = staticmethod(np.empty)
    exp = staticmethod(np.exp)
    finfo = staticmethod(np.finfo)
    hypot = staticmethod(np.hypot)
    isclose = staticmethod(np.isclose)
    isfinite = staticmethod(np.isfinite)
    imag = staticmethod(np.imag)
    max = staticmethod(np.max)
    real = staticmethod(np.real)
    sqrt = staticmethod(np.sqrt)
    triu = staticmethod(np.triu)
    zeros_like = staticmethod(np.zeros_like)


def _base_problem() -> tuple[np.ndarray, ...]:
    h0 = np.diag([0.1, 0.7]).astype(np.complex128)
    mu_x = np.array([[0.0, 0.4], [0.4, 0.0]], dtype=np.complex128)
    mu_y = np.array([[0.0, -0.4j], [0.4j, 0.0]], dtype=np.complex128)
    initial = np.array([np.sqrt(0.65), 1j * np.sqrt(0.35)], dtype=np.complex128)
    return h0, mu_x, mu_y, initial


def _case(mode: str) -> tuple[tuple[Any, ...], dict[str, Any]]:
    h0, mu_x, mu_y, initial = _base_problem()
    if mode == "static":
        field_x = np.array([0.2, -0.1, 0.3, 0.4, -0.2, 0.1, 0.25])
        field_y = 0.5 * field_x
        kwargs: dict[str, Any] = {
            "return_traj": True,
            "sample_stride": 2,
            "renorm": True,
        }
    elif mode == "rotating":
        field_x = np.array([0.3, 0.2, 0.0, -0.2, -0.3])
        field_y = np.array([0.0, 0.2, 0.3, 0.2, 0.0])
        kwargs = {
            "return_traj": False,
            "sample_stride": 1,
            "magnetic_quantum_numbers": np.array([0.0, 1.0]),
        }
    elif mode == "helicity_projected":
        field_x = np.zeros(7)
        field_y = np.zeros(7)
        kwargs = {
            "return_traj": True,
            "sample_stride": 1,
            "interaction_mode": "helicity_projected",
            "polarization": np.array([1.0, 1.0j]) / np.sqrt(2.0),
            "scalar_field": np.array([0.2, -0.1, 0.3, 0.4, -0.2, 0.1, 0.25]),
        }
    else:
        raise AssertionError(f"unknown test mode: {mode}")
    return (h0, mu_x, mu_y, field_x, field_y, initial, 0.03), kwargs


@pytest.mark.parametrize("mode", ["static", "rotating", "helicity_projected"])
def test_cupy_split_graph_matches_numpy_without_cuda(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    arguments, kwargs = _case(mode)
    expected = splitop_schrodinger(*arguments, backend="numpy", **kwargs)
    monkeypatch.setattr(schrodinger_cupy, "cp", _NumPyCuPyDouble)

    actual = splitop_schrodinger(*arguments, backend="cupy", **kwargs)

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=2.0e-15, atol=2.0e-15)


@pytest.mark.parametrize(
    ("mode", "mutation", "message"),
    [
        ("static", "nonhermitian", "mu_x must be Hermitian"),
        ("rotating", "bad_covariance", "rotation covariance"),
        ("helicity_projected", "bad_polarization", "must be normalized"),
    ],
)
def test_cupy_split_preserves_explicit_validation_failures(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    mutation: str,
    message: str,
) -> None:
    arguments, kwargs = _case(mode)
    mutable = list(arguments)
    if mutation == "nonhermitian":
        mu_x = mutable[1].copy()
        mu_x[0, 1] += 0.1j
        mutable[1] = mu_x
    elif mutation == "bad_covariance":
        mutable[2] = np.zeros_like(mutable[2])
    elif mutation == "bad_polarization":
        kwargs["polarization"] = np.array([1.0, 1.0j])
    else:
        raise AssertionError(f"unknown mutation: {mutation}")

    with pytest.raises(ValueError, match=message):
        splitop_schrodinger(*mutable, backend="numpy", **kwargs)

    monkeypatch.setattr(schrodinger_cupy, "cp", _NumPyCuPyDouble)
    with pytest.raises(ValueError, match=message):
        splitop_schrodinger(*mutable, backend="cupy", **kwargs)


def test_cupy_split_rejects_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, kwargs = _case("static")
    monkeypatch.setattr(schrodinger_cupy, "cp", None)

    with pytest.raises(RuntimeError, match="CuPy is not installed"):
        splitop_schrodinger(*arguments, backend="cupy", **kwargs)


@pytest.mark.gpu
@pytest.mark.skipif(cp is None, reason="CuPy not available")
@pytest.mark.parametrize("mode", ["static", "rotating", "helicity_projected"])
def test_real_cupy_split_matches_numpy_and_stays_on_device(mode: str) -> None:
    arguments, kwargs = _case(mode)
    expected = splitop_schrodinger(*arguments, backend="numpy", **kwargs)

    actual = splitop_schrodinger(*arguments, backend="cupy", **kwargs)

    assert isinstance(actual, cp.ndarray)
    assert actual.dtype == cp.complex128
    assert actual.shape == expected.shape
    np.testing.assert_allclose(cp.asnumpy(actual), expected, rtol=2.0e-12, atol=2.0e-13)
    np.testing.assert_allclose(
        cp.asnumpy(cp.linalg.norm(actual, axis=1)),
        np.ones(actual.shape[0]),
        rtol=0.0,
        atol=2.0e-12,
    )
