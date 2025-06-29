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


@njit(complex128(int64, int64), cache=True, fastmath=True)
def tdm_j(J1: int, J2: int):
    dJ = J2 - J1
    if dJ == 1:
        return 0.5 * np.sqrt(J1 / (2 * J1 + 1)) + 0.0j
    if dJ == -1:
        return 0.5 * np.sqrt((J1 + 1) / (2 * J1 + 1)) + 0.0j
    return 0.0 + 0.0j
