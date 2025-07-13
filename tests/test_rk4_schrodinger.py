import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
import numpy as np
import pytest

from rovibrational_excitation.core.propagation.algorithms.rk4.schrodinger import rk4_schrodinger


def make_simple_case():
    H0 = np.diag([0.0, 1.0])
    mux = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    muy = np.zeros((2, 2), dtype=np.complex128)
    Ex = np.linspace(0, 1, 3)
    Ey = np.zeros(3)
    psi0 = np.array([1.0, 0.0], dtype=np.complex128)
    dt = 0.1
    return H0, mux, muy, Ex, Ey, psi0, dt


def test_rk4_schrodinger_norm():
    H0, mux, muy, Ex, Ey, psi0, dt = make_simple_case()
    traj = rk4_schrodinger(H0, mux, muy, Ex, Ey, psi0, dt, return_traj=True)
    assert traj.shape[1] == 2
    for i in range(traj.shape[0]):
        norm = np.linalg.norm(traj[i])
        np.testing.assert_allclose(norm, 1.0, atol=1e-6)


def test_rk4_schrodinger_stride():
    H0, mux, muy, Ex, Ey, psi0, dt = make_simple_case()
    traj = rk4_schrodinger(H0, mux, muy, Ex, Ey, psi0, dt, return_traj=True, stride=2)
    assert traj.shape[0] == 1


def test_rk4_schrodinger_renorm():
    H0, mux, muy, Ex, Ey, psi0, dt = make_simple_case()
    traj = rk4_schrodinger(
        H0, mux, muy, Ex, Ey, psi0, dt, return_traj=True, renorm=True
    )
    for i in range(traj.shape[0]):
        norm = np.linalg.norm(traj[i])
        np.testing.assert_allclose(norm, 1.0, atol=1e-6)


def test_rk4_schrodinger_error():
    H0, mux, muy, Ex, Ey, psi0, dt = make_simple_case()
    # 空の電場配列でValueErrorが発生することを確認
    with pytest.raises(ValueError):
        rk4_schrodinger(H0, mux, muy, np.array([]), np.array([]), psi0, dt)
