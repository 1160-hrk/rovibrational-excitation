import numpy as np
from numba import njit

@njit(cache=True)
def rk4_schrodinger_traj(
    H0: np.ndarray,
    mu_x: np.ndarray,
    mu_y: np.ndarray,
    Efield_x: np.ndarray,
    Efield_y: np.ndarray,
    psi: np.ndarray,
    dt: float,
    steps: int,
    sample_stride: int = 1,
) -> np.ndarray:
    """
    Runge–Kutta 4 次法で時間依存シュレディンガー方程式を伝搬し、
    ウェーブファンクションの軌跡を返す。

    Parameters
    ----------
    H0 : (dim, dim) complex ndarray
        時間に依存しないハミルトニアン
    mu_x, mu_y : (dim, dim) complex ndarray
        x, y 方向の双極子行列
    Efield_x, Efield_y : (2*steps+1,) real ndarray
        x, y 方向の電場配列（t = 0, dt/2, dt, … , steps*dt の順）
    psi : (dim,) または (dim,1) complex ndarray
        初期状態ベクトル
    dt : float
        時間ステップ
    steps : int
        伝搬ステップ数
    sample_stride : int, optional
        何ステップごとに状態を記録するか（既定 1）

    Returns
    -------
    traj : (steps//sample_stride + 1, dim) complex ndarray
        各サンプル時刻のウェーブファンクション
    """

    # 1 次元ベクトルで処理
    psi = psi.ravel()
    dim = psi.size

    n_samples = steps // sample_stride + 1
    traj = np.zeros((n_samples, dim), dtype=np.complex128)
    traj[0, :] = psi

    # ループ内で再利用する作業バッファ
    psi_tmp = np.empty_like(psi)

    sample_idx = 1
    for step in range(steps):
        idx = 2 * step          # 電場配列の基準インデックス

        ex1 = Efield_x[idx]
        ex2 = Efield_x[idx + 1]
        ex4 = Efield_x[idx + 2]

        ey1 = Efield_y[idx]
        ey2 = Efield_y[idx + 1]
        ey4 = Efield_y[idx + 2]

        # 各副ステップのハミルトニアン
        H1 = H0 + mu_x * ex1 + mu_y * ey1
        H2 = H0 + mu_x * ex2 + mu_y * ey2   # H3 と同じ
        H4 = H0 + mu_x * ex4 + mu_y * ey4

        # --- Runge–Kutta 4 ---
        k1 = -1j * (H1 @ psi)

        psi_tmp[:] = psi
        psi_tmp += 0.5 * dt * k1
        k2 = -1j * (H2 @ psi_tmp)

        psi_tmp[:] = psi
        psi_tmp += 0.5 * dt * k2
        k3 = -1j * (H2 @ psi_tmp)           # H3 = H2

        psi_tmp[:] = psi
        psi_tmp += dt * k3
        k4 = -1j * (H4 @ psi_tmp)

        psi += (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        # --------------------

        if (step + 1) % sample_stride == 0:
            traj[sample_idx, :] = psi
            sample_idx += 1

    return traj


@njit(cache=True)
def rk4_schrodinger(
    H0: np.ndarray,
    mu_x: np.ndarray,
    mu_y: np.ndarray,
    Efield_x: np.ndarray,
    Efield_y: np.ndarray,
    psi: np.ndarray,
    dt: float,
    steps: int,
) -> np.ndarray:
    """
    Runge–Kutta 4 次法で時間依存シュレディンガー方程式を伝搬し、
    波動関数の最終値を返す。

    Parameters
    ----------
    H0 : (dim, dim) complex ndarray
        時間に依存しないハミルトニアン
    mu_x, mu_y : (dim, dim) complex ndarray
        x, y 方向の双極子行列
    Efield_x, Efield_y : (2*steps+1,) real ndarray
        x, y 方向の電場配列（t = 0, dt/2, dt, … , steps*dt の順）
    psi : (dim,) または (dim,1) complex ndarray
        初期状態ベクトル
    dt : float
        時間ステップ
    steps : int
        伝搬ステップ数

    Returns
    -------
    traj : (dim,) complex ndarray
        最終時刻の波動関数
    """

    # 1 次元ベクトルで処理
    psi = psi.ravel()

    # ループ内で再利用する作業バッファ
    psi_tmp = np.empty_like(psi)

    sample_idx = 1
    for step in range(steps):
        idx = 2 * step          # 電場配列の基準インデックス

        ex1 = Efield_x[idx]
        ex2 = Efield_x[idx + 1]
        ex4 = Efield_x[idx + 2]

        ey1 = Efield_y[idx]
        ey2 = Efield_y[idx + 1]
        ey4 = Efield_y[idx + 2]

        # 各副ステップのハミルトニアン
        H1 = H0 + mu_x * ex1 + mu_y * ey1
        H2 = H0 + mu_x * ex2 + mu_y * ey2   # H3 と同じ
        H4 = H0 + mu_x * ex4 + mu_y * ey4

        # --- Runge–Kutta 4 ---
        k1 = -1j * (H1 @ psi)

        psi_tmp[:] = psi
        psi_tmp += 0.5 * dt * k1
        k2 = -1j * (H2 @ psi_tmp)

        psi_tmp[:] = psi
        psi_tmp += 0.5 * dt * k2
        k3 = -1j * (H2 @ psi_tmp)           # H3 = H2

        psi_tmp[:] = psi
        psi_tmp += dt * k3
        k4 = -1j * (H4 @ psi_tmp)

        psi += (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        # --------------------

    return psi