import sys
import os

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from rovibrational_excitation.core.hamiltonian import generate_H0_LinMol
from rovibrational_excitation.core.basis import LinMolBasis

basis = LinMolBasis(V_max=4, J_max=4)
hamiltonian = generate_H0_LinMol(basis=basis, delta_omega_rad_phz=0.08, B_rad_phz=0.01)
energy = np.diag(hamiltonian)

fig, ax = plt.subplots()
ax.plot(energy, 'd', ms=3)