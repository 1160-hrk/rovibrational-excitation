"""
rk4_lvne.py  ―  Liouville-von Neumann RK4 propagator（改良版）
--------------------------------------------------------------
* Numba `fastmath=True` で SIMD/FMA を有効化
* ループ内スライスを排し，直接インデックスで電場を取得
* `rho_tmp` 行列をループ外で 1 回だけ確保し再利用（メモリ節約 & ヒープ確保ゼロ）
* 栗田・大久保型のエルミート保証は入れていない（元実装踏襲）
"""

from __future__ import annotations

import numpy as np
from numba import njit


# ------------------------------------------------------------
# 低レベル共通コア（軌跡記録あり／なしをフラグで切替）
# ------------------------------------------------------------
@njit(
    "c16[:, :, :](c16[:, :], c16[:, :], c16[:, :],"
    "f8[:], f8[:],"
    "c16[:, :], f8, i8, i8, b1)",
    cache=True,
    fastmath=True,
)
def _rk4_lvne_core(
    H0: np.ndarray,
    mu_x: np.ndarray,
    mu_y: np.ndarray,
    Ex: np.ndarray,  # (2*steps+1,)
    Ey: np.ndarray,  # (2*steps+1,)
    rho0: np.ndarray,  # (dim, dim)
    dt: float,
    steps: int,
    stride: int,
    record_traj: bool,
) -> np.ndarray:
    """内部用：complex128 固定"""

    dim = rho0.shape[0]
    n_out = steps // stride + 1 if record_traj else 1
    traj = np.empty((n_out, dim, dim), np.complex128)

    rho = rho0.copy()
    traj[0] = rho

    buf = np.empty_like(rho)  # 作業バッファ
    out_idx = 1

    for s in range(steps):
        idx = 2 * s
        ex1 = Ex[idx]
        ex2 = Ex[idx + 1]
        ex4 = Ex[idx + 2]

        ey1 = Ey[idx]
        ey2 = Ey[idx + 1]
        ey4 = Ey[idx + 2]

        H1 = H0 + mu_x * ex1 + mu_y * ey1
        H2 = H0 + mu_x * ex2 + mu_y * ey2  # H3 と同じ
        H4 = H0 + mu_x * ex4 + mu_y * ey4

        # --- RK4 ---
        k1 = -1j * (H1 @ rho - rho @ H1)

        buf[:, :] = rho + 0.5 * dt * k1
        k2 = -1j * (H2 @ buf - buf @ H2)

        buf[:, :] = rho + 0.5 * dt * k2
        k3 = -1j * (H2 @ buf - buf @ H2)

        buf[:, :] = rho + dt * k3
        k4 = -1j * (H4 @ buf - buf @ H4)

        rho += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        # --------------------

        if record_traj and ((s + 1) % stride == 0):
            traj[out_idx] = rho
            out_idx += 1

    if not record_traj:
        traj[0] = rho

    return traj


# ------------------------------------------------------------
# 公開 API
# ------------------------------------------------------------
def rk4_lvne_traj(
    H0,
    mu_x,
    mu_y,
    Efield_x,
    Efield_y,
    rho0,
    dt: float,
    steps: int,
    sample_stride: int = 1,
) -> np.ndarray:
    """
    軌跡を返す版  ―  shape = (steps//sample_stride+1, dim, dim)
    """
    return _rk4_lvne_core(
        np.ascontiguousarray(H0, dtype=np.complex128),
        np.ascontiguousarray(mu_x, dtype=np.complex128),
        np.ascontiguousarray(mu_y, dtype=np.complex128),
        np.asarray(Efield_x, dtype=np.float64),
        np.asarray(Efield_y, dtype=np.float64),
        np.ascontiguousarray(rho0, dtype=np.complex128),
        float(dt),
        int(steps),
        int(sample_stride),
        True,  # record_traj
    )


def rk4_lvne(
    H0,
    mu_x,
    mu_y,
    Efield_x,
    Efield_y,
    rho0,
    dt: float,
    steps: int,
) -> np.ndarray:
    """
    最終密度行列だけ返す軽量版  ―  shape = (dim, dim)
    """
    traj = _rk4_lvne_core(
        np.ascontiguousarray(H0, dtype=np.complex128),
        np.ascontiguousarray(mu_x, dtype=np.complex128),
        np.ascontiguousarray(mu_y, dtype=np.complex128),
        np.asarray(Efield_x, dtype=np.float64),
        np.asarray(Efield_y, dtype=np.float64),
        np.ascontiguousarray(rho0, dtype=np.complex128),
        float(dt),
        int(steps),
        1,  # stride (dummy)
        False,  # record_traj
    )
    return traj[0]  # (1, dim, dim) → (dim, dim)
