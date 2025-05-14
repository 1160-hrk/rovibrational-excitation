import sys
sys.path.append('..')
from src.rovibrational_interaction_simulation.core.states import StateVector, DensityMatrix
from src.rovibrational_interaction_simulation.core.basis import LinMolBasis

basis = LinMolBasis(V_max=4, J_max=4)
state = StateVector(basis=basis)
