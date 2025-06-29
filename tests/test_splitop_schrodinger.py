import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpy as np
import pytest

from rovibrational_excitation.core._splitop_schrodinger import splitop_schrodinger

# CuPyが利用可能か判定
try:
    import cupy as cp  # noqa: F401

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


def make_simple_case():
    H0 = np.diag([0.0, 1.0])
    mu_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    mu_y = np.zeros((2, 2), dtype=np.complex128)
    pol = np.array([1.0, 0.0], dtype=np.complex128)
    Efield = np.linspace(0, 1, 3)  # 2ステップ分
    psi0 = np.array([1.0, 0.0], dtype=np.complex128)
    dt = 0.1
    steps = 1
    return H0, mu_x, mu_y, pol, Efield, psi0, dt, steps


def test_splitop_schrodinger_norm():
    H0, mu_x, mu_y, pol, Efield, psi0, dt, steps = make_simple_case()
    traj = splitop_schrodinger(H0, mu_x, mu_y, pol, Efield, psi0, dt, steps)
    assert traj.shape == (2, 2, 1)
    for i in range(traj.shape[0]):
        norm = np.linalg.norm(traj[i, :, 0])
        np.testing.assert_allclose(norm, 1.0, atol=1e-12)


def test_splitop_schrodinger_stride():
    H0, mu_x, mu_y, pol, Efield, psi0, dt, steps = make_simple_case()
    traj = splitop_schrodinger(
        H0, mu_x, mu_y, pol, Efield, psi0, dt, steps, sample_stride=1
    )
    assert traj.shape == (2, 2, 1)
    traj2 = splitop_schrodinger(
        H0, mu_x, mu_y, pol, Efield, psi0, dt, steps, sample_stride=2
    )
    assert traj2.shape == (1, 2, 1)


def test_splitop_schrodinger_hbar():
    H0, mu_x, mu_y, pol, Efield, psi0, dt, steps = make_simple_case()
    # hbarを変えてもエラーにならないこと
    traj = splitop_schrodinger(H0, mu_x, mu_y, pol, Efield, psi0, dt, steps, hbar=2.0)
    assert traj.shape[1] == 2


@pytest.mark.skipif(not HAS_CUPY, reason="CuPyがインストールされていないためスキップ")
def test_splitop_schrodinger_cupy():
    H0, mu_x, mu_y, pol, Efield, psi0, dt, steps = make_simple_case()
    traj = splitop_schrodinger(
        H0, mu_x, mu_y, pol, Efield, psi0, dt, steps, backend="cupy"
    )
    assert traj.shape == (2, 2, 1)
    for i in range(traj.shape[0]):
        norm = np.linalg.norm(traj[i, :, 0])
        np.testing.assert_allclose(norm, 1.0, atol=1e-12)


def test_splitop_schrodinger_backend_error():
    H0, mu_x, mu_y, pol, Efield, psi0, dt, steps = make_simple_case()
    # CuPy未インストール時にcupy指定でエラー
    import importlib

    sys_modules_backup = sys.modules.copy()
    if "cupy" in sys.modules:
        del sys.modules["cupy"]
    importlib.reload(
        __import__(
            "rovibrational_excitation.core._splitop_schrodinger",
            fromlist=["splitop_schrodinger"],
        )
    )
    from rovibrational_excitation.core._splitop_schrodinger import (
        splitop_schrodinger as splitop_reload,
    )

    try:
        with pytest.raises(RuntimeError):
            splitop_reload(H0, mu_x, mu_y, pol, Efield, psi0, dt, steps, backend="cupy")
    finally:
        sys.modules = sys_modules_backup
