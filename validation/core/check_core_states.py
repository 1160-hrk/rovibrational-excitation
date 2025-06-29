import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from rovibrational_excitation.core.basis import LinMolBasis
from rovibrational_excitation.core.states import StateVector

basis = LinMolBasis(V_max=4, J_max=4)
state = StateVector(basis=basis)
