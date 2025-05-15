# hamiltonian.py
import numpy as np
from .basis import LinMolBasis
import numpy as np

def generate_H0_LinMol(basis: LinMolBasis, omega_phz=1.0, delta_omega_phz=0.0, B_phz=1.0, alpha_phz=0.0):
    """
    分子の自由ハミルトニアン H0 を生成
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
    vterm = basis.v_array + 0.5
    jterm = basis.j_array * (basis.j_array + 1)
    energy = omega_phz * vterm - delta_omega_phz * vterm**2
    energy += (B_phz - alpha_phz * vterm) * jterm
    H0 = np.diag(energy)
    return H0