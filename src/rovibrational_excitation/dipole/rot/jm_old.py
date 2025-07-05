"""
rovibrational_excitation.dipole.rot/jm.py  ―  rigid-rotor TDM matrix elements |J M⟩ ↔ |J' M'⟩
===================================================================

* 参照式（Hönl–London，線形分子）をそのまま実装
* ΔJ = ±1，J=J'=0 禁制
* μx, μy : ΔM = ±1 (q = ±1 成分)
* μz     : ΔM = 0  (q = 0 成分)
  ─ μz の数式は **μx/μy と係数が異なる**点に注意。

戻り値
------
complex128   （μx, μy は複素位相を含む。μz は実数だが complex128 で統一）

Numba 0.59+ 用：`@njit(cache=True, fastmath=True)` で JIT コンパイル。
"""

from __future__ import annotations

import numpy as np
from numba import complex128, int64, njit


# --------------------------------------------------------------------
# μx : ΔM = ±1  (対称 q = ±1 組み合わせ)
# --------------------------------------------------------------------
@njit(complex128(int64, int64, int64, int64), cache=True, fastmath=True)
def tdm_jm_x(J1, M1, J2, M2):
    dJ = J2 - J1
    dM = M2 - M1
    if dJ == 1 and dM in (1, -1):
        return (
            (-dM)
            * 0.5
            * np.sqrt(
                (J1 + dM * M1 + 1) * (J1 + dM * M1 + 2) / ((2 * J1 + 1) * (2 * J1 + 3))
            )
        )
    elif dJ == -1 and dM in (1, -1):
        return (
            dM
            * 0.5
            * np.sqrt(
                (J2 - dM * M2 + 1) * (J2 - dM * M2 + 2) / ((2 * J2 + 1) * (2 * J2 + 3))
            )
        )
    return 0.0 + 0.0j


# --------------------------------------------------------------------
# μy : ΔM = ±1  (反対称 q = ±1 組み合わせ)  位相 ±i
# --------------------------------------------------------------------
@njit(complex128(int64, int64, int64, int64), cache=True, fastmath=True)
def tdm_jm_y(J1, M1, J2, M2):
    dJ = J2 - J1
    dM = M2 - M1
    if dJ == 1 and dM in (1, -1):
        return (
            (-1j)
            * 0.5
            * np.sqrt(
                (J1 + dM * M1 + 1) * (J1 + dM * M1 + 2) / ((2 * J1 + 1) * (2 * J1 + 3))
            )
        )
    elif dJ == -1 and dM in (1, -1):
        return (
            (1j)
            * 0.5
            * np.sqrt(
                (J2 - dM * M2 + 1) * (J2 - dM * M2 + 2) / ((2 * J2 + 1) * (2 * J2 + 3))
            )
        )
    return 0.0 + 0.0j


# --------------------------------------------------------------------
# μz : ΔM = 0   **係数が μx/μy と異なる**   実値を返す
# --------------------------------------------------------------------
@njit(complex128(int64, int64, int64, int64), cache=True, fastmath=True)
def tdm_jm_z(J1, M1, J2, M2):
    dJ = J2 - J1
    if M1 != M2:  # ΔM ≠ 0 → 禁制
        return 0.0 + 0.0j

    if dJ == 1:  # J → J+1
        val = np.sqrt((J1 + 1 - M1) * (J1 + 1 + M1) / ((2 * J1 + 1) * (2 * J1 + 3)))
        return val + 0.0j
    elif dJ == -1:  # J → J-1
        val = np.sqrt((J2 + 1 - M2) * (J2 + 1 + M2) / ((2 * J2 + 1) * (2 * J2 + 3)))
        return val + 0.0j
    return 0.0 + 0.0j


# --------------------------------------------------------------------
# 軸 → 関数マッピング
# --------------------------------------------------------------------
tdm_jm_dict = {"x": tdm_jm_x, "y": tdm_jm_y, "z": tdm_jm_z}
