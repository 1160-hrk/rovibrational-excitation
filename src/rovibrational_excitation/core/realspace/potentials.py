from __future__ import annotations

import numpy as np


def morse_potential(r: np.ndarray, De: float, a: float, re: float, *, shift: str = "bottom") -> np.ndarray:
    """
    Morse potential V(r) = De * (1 - exp(-a (r - re)))^2 + shift.

    Parameters
    ----------
    r : ndarray
        Radius array (can be broadcast over 1D/2D/3D grids through r = sqrt(x^2+y^2+z^2)).
    De : float
        Dissociation energy.
    a : float
        Range parameter.
    re : float
        Equilibrium bond distance.
    shift : {"bottom", "zero"}
        - "bottom": set V(re) = 0 (default)
        - "zero"  : set V(∞) = 0 (subtract De)
    """
    V = De * (1.0 - np.exp(-a * (r - re))) ** 2
    if shift == "zero":
        V = V - De
    elif shift == "bottom":
        pass
    else:
        raise ValueError("shift must be 'bottom' or 'zero'")
    return V

