import numpy as np
from numba import njit

@njit(
    # 'c16[:, :, ::1](c16[:,::1], c16[:,::1], c16[:,::1], c16[::1], c16[::1], c16[:, ::1], c16, i8, i8)',
    cache=True
    )
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
    Runge-Kutta 4th order method for time-dependent Schrödinger equation.

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
    psi : np.ndarray
        Initial wavefunction (dim,)
    dt : float
        Time step size
    steps : int
        Number of time steps to propagate
    sample_stride : int, optional
        Stride for sampling the trajectory (default is 1, meaning every step is sampled)

    Returns
    -------
    np.ndarray
        Trajectory of the wavefunction with shape (n_samples, dim), where
        n_samples = steps // sample_stride + 1
    """
    dim = psi.shape[0]

    # 取り出し回数を 1 回にするためステップ長を明示
    n_samples = steps // sample_stride + 1
    psi_traj = np.zeros((n_samples, dim, 1), dtype=np.complex128)

    psi = psi.copy()
    psi_traj[0] = psi          # t = 0
    sample_idx = 1             # 次に書き込むインデクス

    for step in range(steps):
        ex1, ex2, ex4 = Efield_x[2 * step : 2 * step + 3]
        ey1, ey2, ey4 = Efield_y[2 * step : 2 * step + 3]

        H1 = H0 + mu_x * ex1 + mu_y * ey1
        H2 = H0 + mu_x * ex2 + mu_y * ey2
        H4 = H0 + mu_x * ex4 + mu_y * ey4

        # ---- Runge–Kutta 4 ----
        k1 = -1j * H1 @ psi

        psi_tmp = psi + 0.5 * dt * k1
        k2 = -1j * H2 @ psi_tmp

        psi_tmp = psi + 0.5 * dt * k2
        k3 = -1j * H2 @ psi_tmp  # H3 == H2

        psi_tmp = psi + dt * k3
        k4 = -1j * H4 @ psi_tmp

        psi += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # -----------------------

        # stride 間隔でサンプリング
        if (step + 1) % sample_stride == 0:
            psi_traj[sample_idx] = psi
            sample_idx += 1

    return psi_traj


@njit(
    # 'c16[:, ::1](c16[:,::1], c16[:,::1], c16[:,::1], c16[::1], c16[::1], c16[:, ::1], c16, i8, i8)',
    cache=True
    )
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
    Runge-Kutta 4th order method for time-dependent Schrödinger equation.

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
    psi : np.ndarray
        Initial wavefunction (dim,)
    dt : float
        Time step size
    steps : int
        Number of time steps to propagate
    Returns
    -------
    np.ndarray
        final wavefunction with shape (dim)
    """
    dim = psi.shape[0]

    # 取り出し回数を 1 回にするためステップ長を明示
    psi = psi.copy()
    
    for step in range(steps):
        ex1, ex2, ex4 = Efield_x[2 * step : 2 * step + 3]
        ey1, ey2, ey4 = Efield_y[2 * step : 2 * step + 3]

        H1 = H0 + mu_x * ex1 + mu_y * ey1
        H2 = H0 + mu_x * ex2 + mu_y * ey2
        H4 = H0 + mu_x * ex4 + mu_y * ey4

        # ---- Runge–Kutta 4 ----
        k1 = -1j * H1 @ psi

        psi_tmp = psi + 0.5 * dt * k1
        k2 = -1j * H2 @ psi_tmp

        psi_tmp = psi + 0.5 * dt * k2
        k3 = -1j * H2 @ psi_tmp  # H3 == H2

        psi_tmp = psi + dt * k3
        k4 = -1j * H4 @ psi_tmp

        psi += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return psi