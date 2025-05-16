# hamiltonian.py
import numpy as np
from .basis import LinMolBasis
import numpy as np

def generate_H0_LinMol(basis: LinMolBasis, omega_rad_phz=1.0, delta_omega_rad_phz=0.0, B_rad_phz=1.0, alpha_rad_phz=0.0):
    """
    分子の自由ハミルトニアン H0 を生成（単位：rad * PHz）
    E(V, J) = ω*(V+1/2) - Δω*(V+1/2)**2 + (B - α*(V+1/2))*J*(J+1)

    Parameters
    ----------
    omega_phz : float
        振動固有周波数（rad/fs）
    delta_omega_phz : float
        振動の非調和性補正項（rad/fs）
    B_phz : float
        回転定数（rad/fs）
    alpha_phz : float
        振動-回転相互作用定数（rad/fs）
    """
    vterm = basis.V_array + 0.5
    jterm = basis.J_array * (basis.J_array + 1)
    energy = omega_rad_phz * vterm - delta_omega_rad_phz * vterm**2
    energy += (B_rad_phz - alpha_rad_phz * vterm) * jterm
    H0 = np.diag(energy)
    return H0