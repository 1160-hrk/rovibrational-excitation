import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from rovibrational_excitation.core.dipole_matrix import generate_dipole_matrix_vjm, generate_dipole_matrix_jm, generate_dipole_matrix_vj, generate_dipole_matrix_j, LinMolDipoleMatrix
from rovibrational_excitation.core.basis import LinMolBasis
import numpy as np

def test_func(a, b):
    return a+b

basis = LinMolBasis(V_max=1, J_max=1, use_M=True)
dipole_matrix = LinMolDipoleMatrix(
    basis=basis, mu0_cm=3.33564e-30, omega01=1.0, domega=0.01,
    potential_type='harmonic'
    )

dm_jm_x = generate_dipole_matrix_jm(basis=basis,axis='x')
dm_jm_y = generate_dipole_matrix_jm(basis=basis,axis='y')
dm_jm_z = generate_dipole_matrix_jm(basis=basis,axis='z')

dm_vjm_x = generate_dipole_matrix_vjm(
    basis=basis, axis='x', domega=0.01, potential_type='harmonic',
    mu0_cm=3.33564e-30)

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
map = ax.pcolormesh(np.real(dm_jm_x), cmap="bwr")
ax.set_title(r"$\mu_{jm}^x$")
pp = plt.colorbar(map)

fig, ax = plt.subplots()
map = ax.pcolormesh(np.imag(dm_jm_y), cmap="bwr")
ax.set_title(r"$\mu_{jm}^y$")
pp = plt.colorbar(map)


fig, ax = plt.subplots()
map = ax.pcolormesh(np.real(dm_jm_x + 1j*dm_jm_y), cmap="bwr")
ax.set_title(r"$\mu_{jm}^x + 1j*\mu_{jm}^y$")
pp = plt.colorbar(map)

# fig, ax = plt.subplots()
# map = ax.pcolormesh(np.real(dm_jm_z), cmap="bwr")
# ax.set_title(r"$\mu_{jm}^z$")
# pp = plt.colorbar(map)

# fig, ax = plt.subplots()
# map = ax.pcolormesh(np.real(dipole_matrix.mu_x), cmap="bwr")
# pp = plt.colorbar(map)

# fig, ax = plt.subplots()
# map = ax.pcolormesh(np.real(dm_vjm_x), cmap="bwr")
# pp = plt.colorbar(map)