import sys
sys.path.append('..')
from src.rovibrational_interaction_simulation.core.dipole_matrix import generate_dipole_matrix_vjm, generate_dipole_matrix_jm, generate_dipole_matrix_vj, generate_dipole_matrix_j
from src.rovibrational_interaction_simulation.core.basis import VJMBasis
import numpy as np

basis = VJMBasis(V_max=4, J_max=4)
dm_jm_x = generate_dipole_matrix_jm(basis=basis,axis='x')
dm_jm_y = generate_dipole_matrix_jm(basis=basis,axis='y')
dm_jm_z = generate_dipole_matrix_jm(basis=basis,axis='z')

dm_vjm_x = generate_dipole_matrix_vjm(
    basis=basis, axis='x', domega=0.01, potential_type='harmonic',
    mu0_cm=3.33564e-30)

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
map = ax.pcolormesh(np.real(dm_jm_x), cmap="bwr")
pp = plt.colorbar(map)

fig, ax = plt.subplots()
map = ax.pcolormesh(np.imag(dm_jm_y), cmap="bwr")
pp = plt.colorbar(map)

fig, ax = plt.subplots()
map = ax.pcolormesh(np.real(dm_jm_z), cmap="bwr")
pp = plt.colorbar(map)

fig, ax = plt.subplots()
map = ax.pcolormesh(np.real(dm_vjm_x), cmap="bwr")
pp = plt.colorbar(map)