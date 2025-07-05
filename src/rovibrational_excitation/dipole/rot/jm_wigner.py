"""
rovibrational_excitation.dipole.rot.jm_wigner
===========================================
Rigid-rotor transition-dipole matrix elements  |J M⟩ ↔ |J' M'⟩
(Linear molecule) evaluated via Wigner-3j symbols.

The spherical tensor expression is
    ⟨J' M'| μ_q |J M⟩ = (-1)^{M'} √[(2J+1)(2J'+1)]
                         × ( J  1  J' ; −M  q  M' )
                         × ( J  1  J' ;  0  0  0 )
where q = -1, 0, +1 corresponds to the spherical components of the
dipole operator.  Cartesian components are obtained through

    μ_x = −(μ_{+1} − μ_{−1}) / √2
    μ_y =  i(μ_{+1} + μ_{−1}) / √2
    μ_z =  μ_0

This module reproduces the analytic implementation in ``jm.py`` but
serves as an independent cross-check using general angular-momentum
algebra.
"""

from __future__ import annotations

import numpy as np
from sympy.physics.wigner import wigner_3j

__all__ = [
    "tdm_jm_x_wigner",
    "tdm_jm_y_wigner",
    "tdm_jm_z_wigner",
    "tdm_jm_dict_wigner",
]

SQRT2 = np.sqrt(2.0)


def _w3j(j1: int, j2: int, j3: int, m1: int, m2: int, m3: int) -> float:
    """Return Wigner-3j value as float (double precision)."""
    return float(wigner_3j(j1, j2, j3, m1, m2, m3))


def _tdm_spherical(J1: int, M1: int, J2: int, M2: int, q: int):
    """Spherical tensor dipole component μ_q (complex128).

    Formula derived from Wigner–Eckart theorem for a *linear* rotor with the
    body-fixed dipole along the molecular axis (K=0):

        ⟨J₂ M₂| μ_q |J₁ M₁⟩ = (−1)^{J₁−M₁}
                              √(2J₁+1)
                              ( J₁  1  J₂ ;  M₁  q  −M₂ )
                              ( J₁  1  J₂ ;  0   0   0 ).

    This expression exactly reproduces the Hönl–London factors implemented
    analytically in ``jm.py`` after the Cartesian transformation below.
    """

    # Overall phase / reduced-element factor
    pref = (-1) ** M2 * np.sqrt((2*J1 + 1)*(2*J2 + 1))

    # First 3j: M-coupling
    three_j_m = _w3j(J1, 1, J2, M1, q, -M2)

    # Second 3j: K-coupling (K=0 for linear molecules)
    three_j_k = _w3j(J1, 1, J2, 0, 0, 0)

    return pref * three_j_m * three_j_k + 0.0j


# ---------------------------------------------------------------------------
# Cartesian components (lab-frame)
# ---------------------------------------------------------------------------

# For a given pair (J₁,M₁) → (J₂,M₂) only one of q=±1 contributes; therefore
# the combinations below reduce correctly to a single term with the factor 1/2.


def tdm_jm_x_wigner(J1: int, M1: int, J2: int, M2: int):
    """μ_x via Wigner-3j (complex128)."""
    t_p1 = _tdm_spherical(J1, M1, J2, M2, q=+1)
    t_m1 = _tdm_spherical(J1, M1, J2, M2, q=-1)
    return -(t_p1 - t_m1) / SQRT2


def tdm_jm_y_wigner(J1: int, M1: int, J2: int, M2: int):
    """μ_y via Wigner-3j (complex128)."""
    t_p1 = _tdm_spherical(J1, M1, J2, M2, q=+1)
    t_m1 = _tdm_spherical(J1, M1, J2, M2, q=-1)
    return 1j * (t_p1 + t_m1) / SQRT2


def tdm_jm_z_wigner(J1: int, M1: int, J2: int, M2: int):
    """μ_z via Wigner-3j (real, returned as complex128)."""
    return _tdm_spherical(J1, M1, J2, M2, q=0)


# Map for convenience ---------------------------------------------------------
tdm_jm_dict_wigner = {
    "x": tdm_jm_x_wigner,
    "y": tdm_jm_y_wigner,
    "z": tdm_jm_z_wigner,
} 

if __name__ == "__main__":
    j1 = 0
    m1 = 0
    j2 = j1 + 1
    m2 = m1 + 1
    print(tdm_jm_x_wigner(j1, m1, j2, m2))
    print(tdm_jm_x_wigner(j2, m2, j1, m1))
    print(tdm_jm_y_wigner(j1, m1, j2, m2))
    print(tdm_jm_y_wigner(j2, m2, j1, m1))
    print(tdm_jm_z_wigner(j1, m1, j2, m1))
    print(tdm_jm_z_wigner(j2, m1, j1, m1))