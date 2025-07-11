"""RK4基本テスト"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpy as np
import pytest

from rovibrational_excitation.core.propagation.algorithms.rk4.lvne import rk4_lvne
from rovibrational_excitation.core.propagation.algorithms.rk4.schrodinger import rk4_schrodinger


def setup_system():
    """テストシステムセットアップ"""
    H0 = np.diag([0.0, 1.0])
    mu_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    mu_y = np.zeros((2, 2), dtype=np.complex128)
    return H0, mu_x, mu_y


class TestRK4Basic:
    """RK4基本機能テスト"""

    def test_schrodinger_norm_conservation(self):
        """Schrödingerノルム保存"""
        H0, mu_x, mu_y = setup_system()
        Ex = np.array([0, 0.1, 0])
        Ey = np.zeros(3)
        psi0 = np.array([1, 0], dtype=complex)

        result = rk4_schrodinger(H0, mu_x, mu_y, Ex, Ey, psi0, dt=0.1)
        norm = np.linalg.norm(result[0])
        np.testing.assert_allclose(norm, 1.0, atol=1e-5)

    def test_liouville_trace_conservation(self):
        """Liouvilleトレース保存"""
        H0, mu_x, mu_y = setup_system()
        Ex = np.array([0, 0.1, 0])
        Ey = np.zeros(3)
        rho0 = np.eye(2, dtype=complex) / 2

        result = rk4_lvne(H0, mu_x, mu_y, Ex, Ey, rho0, dt=0.1, steps=1)
        trace = np.trace(result)
        np.testing.assert_allclose(trace, 1.0, atol=1e-12)

    def test_field_error(self):
        """電場エラーハンドリング"""
        H0, mu_x, mu_y = setup_system()
        psi0 = np.array([1, 0], dtype=complex)

        # 空の電場配列でValueErrorが発生することを確認
        with pytest.raises(ValueError):
            rk4_schrodinger(
                H0, mu_x, mu_y, np.array([]), np.array([]), psi0, dt=0.1
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
