import sys
sys.path.append('..')
from src.rovibrational_interaction_simulation.core.hamiltonian import generate_free_hamiltonian
from src.rovibrational_interaction_simulation.core.basis import VJMBasis
import numpy as np

basis = VJMBasis(V_max=4, J_max=4)
hamiltonian = generate_free_hamiltonian(basis=basis, delta_omega_phz=0.08, B_phz=0.01)
energy = np.diag(hamiltonian)

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(energy, 'd', ms=3)