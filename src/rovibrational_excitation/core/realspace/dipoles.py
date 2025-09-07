from __future__ import annotations

import numpy as np


def dipole_x_cartesian(x: np.ndarray, y: np.ndarray | None = None, z: np.ndarray | None = None, *, a: float = 1.0) -> np.ndarray:
    """
    Real-space dipole function aligned with x-direction:
        mu_x(r, theta, phi) = r^4 exp(-a r) sin(theta) cos(phi)

    In Cartesian coordinates, sin(theta) cos(phi) = x / r, hence
        mu_x(x,y,z) = r^4 exp(-a r) * x/r = x * r^3 * exp(-a r)

    Parameters
    ----------
    x, y, z : arrays defining the grid (broadcastable). For 1D, pass only x.
    a : float
        Decay parameter in exp(-a r).
    """
    if y is None and z is None:
        r = np.abs(x)
        return x * (r**3) * np.exp(-a * r)
    else:
        if y is None or z is None:
            raise ValueError("For 2D/3D, provide both y and z arrays (z can be 0 for 2D)")
        r = np.sqrt(x**2 + y**2 + z**2)
        # avoid division by zero at r=0; expression already avoids division
        return x * (r**3) * np.exp(-a * r)

