"""
rovibrational_excitation.dipole.rot.jm
=====================================
Rigid-rotor transition-dipole matrix elements  |J M⟩ ↔ |J′ M′⟩
(Linear molecule, K = 0) evaluated in closed form.

Hönl–London factors (Σ ↔ Σ, perpendicular + parallel) are implemented
exactly, so the result is identical to a Wigner-3j evaluation but many
times faster under Numba JIT.

Selection rules
---------------
    ΔJ = ±1   (J = J′ = 0 is forbidden)
    ΔM = 0, ±1
        μx, μy : ΔM = ±1     (q = ±1 components)
        μz     : ΔM = 0      (q = 0  component)

Returns
-------
complex128  –  μx, μy contain a phase (± or ±i),  μz is real but the
              return type is unified to complex128 for convenience.
"""

from __future__ import annotations

import numpy as np
from numba import complex128, int64, njit


# --------------------------------------------------------------------
# helper (common to x-/y-components)
# --------------------------------------------------------------------
@njit(cache=True)
def _hl_perp(j: int, m: int, dm: int) -> float:
    """
    Hönl–London factor (perpendicular, Σ↔Σ) **without** the 1/2 prefactor.

        |⟨J+1, M+dm | μ_{±1} | J, M⟩|^2 =
            [(J + dm·M + 1)(J + dm·M + 2)]
            ─────────────────────────────────────────────────────────────
                    4 · (2J+1)(2J+3)

    The caller must still multiply by ½ and take the square root.
    """
    num = (j + dm * m + 1) * (j + dm * m + 2)
    den = (2 * j + 1) * (2 * j + 3)
    return num / (4*den)


@njit(cache=True)
def _hl_para(j: int, m: int) -> float:
    """
    Hönl–London factor (parallel, Σ↔Σ) for q = 0 component.

        |⟨J+1, M | μ_0 | J, M⟩|^2 =
            [(J+1)² − M²]
            ────────────────────────────────────────────────────────────
                 (2J+1)(2J+3)
    """
    num = (j + 1 - m) * (j + 1 + m)
    den = (2 * j + 1) * (2 * j + 3)
    return num / den


# --------------------------------------------------------------------
# μx : ΔM = ±1   (real, phase (-)^{J})
# --------------------------------------------------------------------
@njit(complex128(int64, int64, int64, int64), cache=True)
def tdm_jm_x(J1, M1, J2, M2):
    dJ = J2 - J1
    dM = M2 - M1
    if abs(dM) != 1:
        return 0.0 + 0.0j

    # R-branch  (J2 = J1 + 1)
    if dJ == 1:
        amp = np.sqrt(_hl_perp(J1, M1, dM))
        phase = -dM
        return phase * amp + 0.0j

    # P-branch  (J2 = J1 − 1)
    elif dJ == -1:
        amp = np.sqrt(_hl_perp(J2, M2, -dM))
        phase = dM
        return phase * amp + 0.0j

    return 0.0 + 0.0j


# --------------------------------------------------------------------
# μy : ΔM = ±1   (imaginary, phase ±i (-)^{J})
# --------------------------------------------------------------------
@njit(complex128(int64, int64, int64, int64), cache=True)
def tdm_jm_y(J1, M1, J2, M2):
    dJ = J2 - J1
    dM = M2 - M1
    if abs(dM) != 1:
        return 0.0 + 0.0j

    # R-branch
    if dJ == 1:
        amp = np.sqrt(_hl_perp(J1, M1, dM))
        phase = 1
        return 1j * phase * amp

    # P-branch
    elif dJ == -1:
        amp = np.sqrt(_hl_perp(J2, M2, -dM))
        phase = -1
        return 1j * phase * amp

    return 0.0 + 0.0j


# --------------------------------------------------------------------
# μz : ΔM = 0   (real, phase (-)^{J})
# --------------------------------------------------------------------
@njit(complex128(int64, int64, int64, int64), cache=True)
def tdm_jm_z(J1, M1, J2, M2):
    if M1 != M2:                # ΔM ≠ 0 → forbidden
        return 0.0 + 0.0j

    dJ = J2 - J1
    # R-branch
    if dJ == 1:
        amp = np.sqrt(_hl_para(J1, M1))
        phase = 1
        return phase * amp + 0.0j

    # P-branch
    elif dJ == -1:
        amp = np.sqrt(_hl_para(J2, M2))
        phase = 1
        return phase * amp + 0.0j

    return 0.0 + 0.0j


# --------------------------------------------------------------------
# Axis → function mapping
# --------------------------------------------------------------------
tdm_jm_dict = {"x": tdm_jm_x, "y": tdm_jm_y, "z": tdm_jm_z}


if __name__ == "__main__":
    j1 = 0
    m1 = 0
    j2 = j1 + 1
    m2 = m1 + 1
    print(tdm_jm_x(j1, m1, j2, m2))
    print(tdm_jm_x(j2, m2, j1, m1))
    print(tdm_jm_y(j1, m1, j2, m2))
    print(tdm_jm_y(j2, m2, j1, m1))
