import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from rovibrational_excitation.simulation.runner import load_params, extract_serializable_params
dirpath = os.path.abspath(os.path.join(os.pardir, "examples"))
params_path = os.path.join(dirpath, "params_CO2_AntiSymm.py")
params = extract_serializable_params(load_params(params_path=params_path))
