import sys
import os
sys.path.append('..')
from src.rovibrational_interaction_simulation.simulation.runner import load_params, extract_serializable_params
dirpath = os.path.abspath(os.path.join(os.pardir, "examples"))
params_path = os.path.join(dirpath, "params_CO2_AntiSymm.py")
params = extract_serializable_params(load_params(params_path=params_path))
