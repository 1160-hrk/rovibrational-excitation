import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from rovibrational_excitation.core.basis import LinMolBasis
from rovibrational_excitation.core.hamiltonian import generate_H0_LinMol

basis = LinMolBasis(V_max=4, J_max=4)
hamiltonian = generate_H0_LinMol(basis=basis, delta_omega_rad_phz=0.08, B_rad_phz=0.01)
energy = np.diag(hamiltonian)

fig, ax = plt.subplots()
ax.plot(energy, "d", ms=3)
