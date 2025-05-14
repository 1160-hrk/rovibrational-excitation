import sys
sys.path.append('..')
from src.rovibrational_interaction_simulation.core.basis import LinMolBasis

V_max = 4
J_max = 4
basis = LinMolBasis(V_max, J_max)

basis_jm = basis.basis[:(basis.J_max+1)**2+1, :]
print('size: ', basis.size())
print('basis until first border of v level\n', basis_jm)
print('J border indices', basis.get_border_indices_j())
print('V border indices', basis.get_border_indices_v())

state = (2, 2, -2)
index = state[0] * (J_max+1)**2 + state[1]**2 + state[1] + state[2]
print(basis.get_index(state), index)