import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpy as np
import pytest

from rovibrational_excitation.core.propagation.algorithms.split_operator.schrodinger import splitop_schrodinger

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
    pol = np.array([1.0, 0.0], dtype=np.float64)  # 実数型
    Efield = np.concatenate([np.zeros(5), np.linspace(0, 1, 11), np.zeros(5)])  # 奇数長
    psi0 = np.array([1.0, 0.0], dtype=np.complex128)
    dt = 0.1
    return H0, mu_x, mu_y, pol, Efield, psi0, dt


def test_splitop_schrodinger_norm():
    H0, mu_x, mu_y, pol, Efield, psi0, dt = make_simple_case()
    traj = splitop_schrodinger(H0, mu_x, mu_y, pol, Efield, psi0, dt, return_traj=True)
    assert len(traj.shape) == 2  # (n_samples, dim)
    for i in range(traj.shape[0]):
        norm = np.linalg.norm(traj[i])
        np.testing.assert_allclose(norm, 1.0, atol=1e-12)


def test_splitop_schrodinger_stride():
    H0, mu_x, mu_y, pol, Efield, psi0, dt = make_simple_case()
    traj = splitop_schrodinger(
        H0, mu_x, mu_y, pol, Efield, psi0, dt, return_traj=True, sample_stride=1
    )
    traj2 = splitop_schrodinger(
        H0, mu_x, mu_y, pol, Efield, psi0, dt, return_traj=True, sample_stride=2
    )
    # stride=2の方がサンプル数が少ないはず
    assert traj2.shape[0] < traj.shape[0]


def test_splitop_schrodinger_return_final():
    H0, mu_x, mu_y, pol, Efield, psi0, dt = make_simple_case()
    # 最終状態のみ返す場合
    final = splitop_schrodinger(H0, mu_x, mu_y, pol, Efield, psi0, dt, return_traj=False)
    assert final.shape == (1, 2)  # (1, dim)


@pytest.mark.skipif(not HAS_CUPY, reason="CuPyがインストールされていないためスキップ")
def test_splitop_schrodinger_cupy():
    H0, mu_x, mu_y, pol, Efield, psi0, dt = make_simple_case()
    traj = splitop_schrodinger(
        H0, mu_x, mu_y, pol, Efield, psi0, dt, return_traj=True, backend="cupy"
    )
    assert len(traj.shape) == 2
    for i in range(traj.shape[0]):
        norm = np.linalg.norm(traj[i])
        np.testing.assert_allclose(norm, 1.0, atol=1e-12)


def test_splitop_schrodinger_backend_error():
    H0, mu_x, mu_y, pol, Efield, psi0, dt = make_simple_case()
    # CuPy未インストール時にcupy指定でエラー
    import importlib

    sys_modules_backup = sys.modules.copy()
    if "cupy" in sys.modules:
        del sys.modules["cupy"]
    importlib.reload(
        __import__(
            "rovibrational_excitation.core.propagation.algorithms.split_operator.schrodinger",
            fromlist=["splitop_schrodinger"],
        )
    )
    from rovibrational_excitation.core.propagation.algorithms.split_operator.schrodinger import (
        splitop_schrodinger as splitop_reload,
    )

    try:
        with pytest.raises(RuntimeError):
            splitop_reload(H0, mu_x, mu_y, pol, Efield, psi0, dt, return_traj=True, backend="cupy")
    finally:
        sys.modules = sys_modules_backup
