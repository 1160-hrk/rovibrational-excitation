# test_runner_call.py
import sys
sys.path.append('..')
from src.rovibrational_interaction_simulation.simulation.runner import run_all

run_all("params_CO2_AntiSymm.py")
