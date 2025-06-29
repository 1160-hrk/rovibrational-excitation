"""
Hamiltonian generation functions.

DEPRECATED: Use basis.generate_H0() method instead.
"""

import warnings

from .basis import LinMolBasis


def generate_H0_LinMol(
    basis: LinMolBasis,
    omega_rad_phz=1.0,
    delta_omega_rad_phz=0.0,
    B_rad_phz=1.0,
    alpha_rad_phz=0.0,
):
    """
    分子の自由ハミルトニアン H0 を生成（単位：rad * PHz）
    E(V, J) = ω*(V+1/2) - Δω*(V+1/2)**2 + (B - α*(V+1/2))*J*(J+1)

    DEPRECATED: Use basis.generate_H0() instead.

    Parameters
    ----------
    omega_rad_phz : float
        振動固有周波数（rad/fs）
    delta_omega_rad_phz : float
        振動の非調和性補正項（rad/fs）
    B_rad_phz : float
        回転定数（rad/fs）
    alpha_rad_phz : float
        振動-回転相互作用定数（rad/fs）
    """
    warnings.warn(
        "generate_H0_LinMol is deprecated. Use basis.generate_H0() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    return basis.generate_H0(
        omega_rad_phz=omega_rad_phz,
        delta_omega_rad_phz=delta_omega_rad_phz,
        B_rad_phz=B_rad_phz,
        alpha_rad_phz=alpha_rad_phz,
    )
