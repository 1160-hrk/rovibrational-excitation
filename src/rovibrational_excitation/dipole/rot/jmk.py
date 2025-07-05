from __future__ import annotations

import numpy as np
from numba import complex128, int64, njit
from sympy.physics.wigner import wigner_3j

SQRT2 = np.sqrt(2.0)


def _wigner_3j_as_float(j1: int, j2: int, j3: int, m1: int, m2: int, m3: int) -> float:
    """sympy.physics.wigner.wigner_3j の戻り値 (SymPy 型) を float へ変換"""
    return float(wigner_3j(j1, j2, j3, m1, m2, m3))


# --------------------------------------------------------------------
# 球面成分 μ_q
# --------------------------------------------------------------------

def _tdm_spherical(
    J1: int,
    M1: int,
    K1: int,
    J2: int,
    M2: int,
    K2: int,
    q: int,
):
    """球面テンソル成分 μ_q を返す (complex128)。"""
    # 量子数が半整数の場合にも対応したいが，当面は整数のみ想定。
    prefactor = (-1) ** (M2 - K2) * np.sqrt((2 * J1 + 1) * (2 * J2 + 1))
    three_j1 = _wigner_3j_as_float(J1, 1, J2, -M1, q, M2)
    three_j2 = _wigner_3j_as_float(J1, 1, J2, -K1, 0, K2)
    return prefactor * three_j1 * three_j2 + 0.0j


# --------------------------------------------------------------------
# カルテシアン成分
# --------------------------------------------------------------------

def tdm_jmk_x(J1, M1, K1, J2, M2, K2):
    """μx 成分 (complex128)"""
    t_p1 = _tdm_spherical(J1, M1, K1, J2, M2, K2, q=+1)
    t_m1 = _tdm_spherical(J1, M1, K1, J2, M2, K2, q=-1)
    return -(t_p1 - t_m1) / SQRT2


def tdm_jmk_y(J1, M1, K1, J2, M2, K2):
    """μy 成分 (complex128)"""
    t_p1 = _tdm_spherical(J1, M1, K1, J2, M2, K2, q=+1)
    t_m1 = _tdm_spherical(J1, M1, K1, J2, M2, K2, q=-1)
    return 1j * (t_p1 + t_m1) / SQRT2


def tdm_jmk_z(J1, M1, K1, J2, M2, K2):
    """μz 成分 (complex128, 実数値)"""
    return _tdm_spherical(J1, M1, K1, J2, M2, K2, q=0)


# --------------------------------------------------------------------
# 軸 → 関数マッピング
# --------------------------------------------------------------------

tdm_jmk_dict = {"x": tdm_jmk_x, "y": tdm_jmk_y, "z": tdm_jmk_z}