import numpy as np
from numba import njit

@njit(cache=True)
def rk4_lvne_traj(
    H0: np.ndarray,
    mu_x: np.ndarray,
    mu_y: np.ndarray,
    Efield_x: np.ndarray,
    Efield_y: np.ndarray,
    rho0: np.ndarray,
    dt: float,
    steps: int,
    sample_stride: int = 1,
) -> np.ndarray:
    """
    Runge-Kutta 4th order method for Liouville-von Neumann equation.

    Parameters
    ----------
    H0 : np.ndarray
        Hamiltonian matrix (dim x dim)
    mu_x : np.ndarray
        Dipole moment matrix in x direction (dim x dim)
    mu_y : np.ndarray
        Dipole moment matrix in y direction (dim x dim)
    Efield_x : np.ndarray
        Electric field in x direction. Must have length 2 * steps + 1.
    Efield_y : np.ndarray
        Electric field in y direction. Must have length 2 * steps + 1.
    rho0 : np.ndarray
        Initial density matrix (dim x dim)
    dt : float
        Time step size
    steps : int
        Number of time steps to propagate
    sample_stride : int, optional
        Stride for sampling the trajectory (default is 1, meaning every step is sampled)

    Returns
    -------
    np.ndarray
        Trajectory of the density matrix with shape (n_samples, dim, dim), where
        n_samples = steps // sample_stride + 1
    """
    dim = rho0.shape[0]

    # 取り出し回数を 1 回にするためステップ長を明示
    n_samples = steps // sample_stride + 1
    rho_traj = np.zeros((n_samples, dim, dim), dtype=np.complex128)

    rho = rho0.copy()
    rho_traj[0] = rho          # t = 0
    sample_idx = 1             # 次に書き込むインデクス

    for step in range(steps):
        # Ex/Ey の 3 点をまとめてスライス（ビューなので追加メモリ不要）
        ex1, ex2, ex4 = Efield_x[2 * step : 2 * step + 3]
        ey1, ey2, ey4 = Efield_y[2 * step : 2 * step + 3]

        # Hamiltonians（H3 は H2 と等しいので用意しない）
        H1 = H0 + mu_x * ex1 + mu_y * ey1
        H2 = H0 + mu_x * ex2 + mu_y * ey2
        H4 = H0 + mu_x * ex4 + mu_y * ey4

        # ---- Runge–Kutta 4 ----
        k1 = -1j * (H1 @ rho - rho @ H1)

        rho_tmp = rho + 0.5 * dt * k1
        k2 = -1j * (H2 @ rho_tmp - rho_tmp @ H2)

        rho_tmp = rho + 0.5 * dt * k2
        k3 = -1j * (H2 @ rho_tmp - rho_tmp @ H2)  # H3 == H2

        rho_tmp = rho + dt * k3
        k4 = -1j * (H4 @ rho_tmp - rho_tmp @ H4)

        rho += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # -----------------------

        # stride 間隔でサンプリング
        if (step + 1) % sample_stride == 0:
            rho_traj[sample_idx] = rho
            sample_idx += 1

    return rho_traj

@njit(cache=True)
def rk4_lvne(
    H0: np.ndarray,
    mu_x: np.ndarray,
    mu_y: np.ndarray,
    Efield_x: np.ndarray,
    Efield_y: np.ndarray,
    rho0: np.ndarray,
    dt: float,
    steps: int,
) -> np.ndarray:
    """
    Runge-Kutta 4th order method for Liouville-von Neumann equation.

    Parameters
    ----------
    H0 : np.ndarray
        Hamiltonian matrix (dim x dim)
    mu_x : np.ndarray
        Dipole moment matrix in x direction (dim x dim)
    mu_y : np.ndarray
        Dipole moment matrix in y direction (dim x dim)
    Efield_x : np.ndarray
        Electric field in x direction. Must have length 2 * steps + 1.
    Efield_y : np.ndarray
        Electric field in y direction. Must have length 2 * steps + 1.
    rho0 : np.ndarray
        Initial density matrix (dim x dim)
    dt : float
        Time step size
    steps : int
        Number of time steps to propagate
    
    Returns
    -------
    np.ndarray
        final density matrix with shape (n_samples, dim, dim)
    """
    dim = rho0.shape[0]

    
    rho = rho0.copy()
    
    for step in range(steps):
        # Ex/Ey の 3 点をまとめてスライス（ビューなので追加メモリ不要）
        ex1, ex2, ex4 = Efield_x[2 * step : 2 * step + 3]
        ey1, ey2, ey4 = Efield_y[2 * step : 2 * step + 3]

        # Hamiltonians（H3 は H2 と等しいので用意しない）
        H1 = H0 + mu_x * ex1 + mu_y * ey1
        H2 = H0 + mu_x * ex2 + mu_y * ey2
        H4 = H0 + mu_x * ex4 + mu_y * ey4

        # ---- Runge–Kutta 4 ----
        k1 = -1j * (H1 @ rho - rho @ H1)

        rho_tmp = rho + 0.5 * dt * k1
        k2 = -1j * (H2 @ rho_tmp - rho_tmp @ H2)

        rho_tmp = rho + 0.5 * dt * k2
        k3 = -1j * (H2 @ rho_tmp - rho_tmp @ H2)  # H3 == H2

        rho_tmp = rho + dt * k3
        k4 = -1j * (H4 @ rho_tmp - rho_tmp @ H4)

        rho += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # -----------------------

    return rho