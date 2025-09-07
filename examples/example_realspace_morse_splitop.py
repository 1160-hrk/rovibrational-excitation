"""
Example: Real-space split-operator propagation with Morse potential and
         x-directed dipole mu_x(x,y,z) = x * r^3 * exp(-a r).

This uses a tiny 2D grid to keep runtime low and show how to wire the
components together. Extend to 3D by providing a z-axis as well.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpy as np
import matplotlib.pyplot as plt

from rovibrational_excitation.core.realspace import (
    Grid2D,
    morse_potential,
    dipole_x_cartesian,
    splitop_realspace,
)
from rovibrational_excitation.core.units.constants import CONSTANTS


def main():
    # Grid (2D)
    Nx = Ny = 128
    L = 20.0  # spatial extent
    x = np.linspace(-L / 2, L / 2, Nx)
    y = np.linspace(-L / 2, L / 2, Ny)
    grid = Grid2D(x=x, y=y)
    X, Y = np.meshgrid(x, y, indexing="ij")
    R = np.sqrt(X**2 + Y**2)

    # Morse potential parameters (arbitrary units)
    De = 0.2
    a = 0.5
    re = 2.0
    V = morse_potential(R, De=De, a=a, re=re, shift="bottom")

    # Dipole along x: mu_x = x * r^3 * exp(-a r)
    mu_x = dipole_x_cartesian(X, Y, np.zeros_like(X), a=0.3)

    # Initial wavefunction: Gaussian at the bottom
    sigma = 1.0
    psi0 = np.exp(-((X - re) ** 2 + Y**2) / (2 * sigma**2))
    psi0 = psi0.astype(np.complex128)
    psi0 /= np.sqrt(np.sum(np.abs(psi0) ** 2))

    # Electric field: single-cycle cosine with Gaussian envelope
    tmax = 200.0
    dt = 0.2
    t = np.arange(0.0, tmax, dt)
    omega = 0.08
    envelope = np.exp(-((t - tmax / 2) ** 2) / (2 * (25.0**2)))
    E = envelope * np.cos(omega * t)

    # Use midpoint sampling array of length 2*steps+1
    steps = len(t)
    E_mid_format = np.empty(2 * steps + 1)
    E_mid_format[::2] = 0.0
    E_mid_format[1::2] = E

    # Mass and hbar (dimensionless demo; set both to 1)
    m = 1.0
    hbar = 1.0  # or CONSTANTS.HBAR for SI

    traj = splitop_realspace(
        psi0,
        (x, y),
        m=m,
        dt=dt,
        steps=steps,
        V=V,
        mu_x=mu_x,
        E_t=E_mid_format,
        backend="numpy",
        return_traj=True,
        sample_stride=steps // 50 if steps >= 50 else 1,
        hbar=hbar,
    )

    # Plot final probability density
    psi_final = traj[-1]
    prob = np.abs(psi_final) ** 2
    plt.figure(figsize=(5, 4))
    im = plt.pcolormesh(x, y, prob.T, shading="auto")
    plt.colorbar(im, label="|psi|^2")
    plt.title("Final probability density (2D)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

