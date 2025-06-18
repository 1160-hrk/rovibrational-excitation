"""
高速版 RK4-Schrödinger propagator
--------------------------------
* **Numba @njit(fastmath=True)** で SIMD／FMA を有効化  
* 電場を (steps, 3) に reshape 済みで渡してインデックス計算を排除  
* ψ の正規化をオプションで切り替え可能  
* トラジェクトリ記録の有無をフラグで切り替えつつ，
  返り値は常に 2-D 配列（後段処理をシンプルにするため）  
  -- `record_traj=False` のときは shape が (1, dim) になる  
"""

from __future__ import annotations
import numpy as np
from numba import njit


# ============================================================
# 低レベル本体（Numba JIT）
# ============================================================

@njit(
    "c16[:, :](c16[:, :], c16[:, :], c16[:, :],"
    "f8[:, :], f8[:, :],"
    "c16[:], f8, i8, i8, b1, b1)",
    cache=True,
    fastmath=True,
)
def _rk4_core(
    H0: np.ndarray,
    mu_x: np.ndarray,
    mu_y: np.ndarray,
    Ex3: np.ndarray,            # (steps, 3)  0, dt/2, dt の電場
    Ey3: np.ndarray,            # (steps, 3)
    psi0: np.ndarray,           # (dim,)
    dt: float,
    steps: int,
    stride: int,
    record_traj: bool,
    renorm: bool,
) -> np.ndarray:
    """内部用：Runge–Kutta 4 本計算（complex128 固定）"""

    psi = psi0.copy()           # 1-D (dim,)
    dim = psi.size
    n_out = steps // stride + 1 if record_traj else 1
    out = np.empty((n_out, dim), np.complex128)
    out[0] = psi

    buf = np.empty_like(psi)    # 作業バッファ
    out_idx = 1

    for s in range(steps):
        ex1, ex2, ex4 = Ex3[s]      # t, t+dt/2, t+dt
        ey1, ey2, ey4 = Ey3[s]

        # ハミルトニアンを 3 つ作成
        H1 = H0 + mu_x * ex1 + mu_y * ey1
        H2 = H0 + mu_x * ex2 + mu_y * ey2   # (H3 と同じ)
        H4 = H0 + mu_x * ex4 + mu_y * ey4

        # --- RK4 ---
        k1 = -1j * (H1 @ psi)

        buf[:] = psi + 0.5 * dt * k1
        k2 = -1j * (H2 @ buf)

        buf[:] = psi + 0.5 * dt * k2
        k3 = -1j * (H2 @ buf)

        buf[:] = psi + dt * k3
        k4 = -1j * (H4 @ buf)

        psi += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        # ----------------------

        if renorm:
            # ─ 確率振幅のドリフトを抑制（強電場・長時間用）─
            norm = (psi.conj() @ psi).real
            psi *= 1.0 / np.sqrt(norm)

        if record_traj and ((s + 1) % stride == 0):
            out[out_idx] = psi
            out_idx += 1
    if not record_traj:
        # 1 ステップだけ記録する場合は最後の行を返す
        out[0] = psi
    # 非記録モードでも shape を (1, dim) にそろえて返す
    return out


# ============================================================
# 公開 API
# ============================================================

def _prepare_field(field: np.ndarray, steps: int) -> np.ndarray:
    """
    1-D 電場配列 (2*steps+1,) → (steps,3) へ変換
    [ t, t+dt/2, t+dt ] を 1 行にまとめて **連続メモリ化**。
    """
    # 安全チェック (Python 側で行い，JIT 前にエラーにする)
    expected_len = 2 * steps + 1
    if field.size != expected_len:
        raise ValueError(f"field length {field.size} != 2*steps+1 ({expected_len})")

    # ビュー同士を column_stack で連結 → 新しい連続配列を生成
    return np.column_stack(
        (field[0:-2:2], field[1:-1:2], field[2::2])
    ).astype(np.float64, copy=False)


def rk4_schrodinger_traj(
    H0: np.ndarray,
    mu_x: np.ndarray,
    mu_y: np.ndarray,
    Efield_x: np.ndarray,
    Efield_y: np.ndarray,
    psi0: np.ndarray,
    dt: float,
    steps: int,
    sample_stride: int = 1,
    renormalize: bool = False,
) -> np.ndarray:
    """
    output the trajectry (spacing with 'stride') of the state (ψ(t))
    ------------------------------------------------------------
    return shape = (steps//sample_stride + 1, dim)
    """
    Ex3 = _prepare_field(Efield_x, steps)
    Ey3 = _prepare_field(Efield_y, steps)

    psi0 = np.asarray(psi0, dtype=np.complex128).ravel()

    return _rk4_core(
        np.ascontiguousarray(H0, dtype=np.complex128),
        np.ascontiguousarray(mu_x, dtype=np.complex128),
        np.ascontiguousarray(mu_y, dtype=np.complex128),
        Ex3,
        Ey3,
        psi0,
        float(dt),
        int(steps),
        int(sample_stride),
        True,           # record_traj
        bool(renormalize),
    )


def rk4_schrodinger(
    H0: np.ndarray,
    mu_x: np.ndarray,
    mu_y: np.ndarray,
    Efield_x: np.ndarray,
    Efield_y: np.ndarray,
    psi0: np.ndarray,
    dt: float,
    steps: int,
    renormalize: bool = False,
) -> np.ndarray:
    """
    output only the final state (ψ(t_final)) 
    ------------------------------------------------------------
    return shape = (dim,)  （内部では (1,dim) を使い最後の行を返す）
    """
    Ex3 = _prepare_field(Efield_x, steps)
    Ey3 = _prepare_field(Efield_y, steps)

    psi0 = np.asarray(psi0, dtype=np.complex128).ravel()

    traj = _rk4_core(
        np.ascontiguousarray(H0, dtype=np.complex128),
        np.ascontiguousarray(mu_x, dtype=np.complex128),
        np.ascontiguousarray(mu_y, dtype=np.complex128),
        Ex3,
        Ey3,
        psi0,
        float(dt),
        int(steps),
        1,              # stride (dummy)
        False,          # record_traj
        bool(renormalize),
    )
    return traj[0]      # (1,dim) → (dim,)
