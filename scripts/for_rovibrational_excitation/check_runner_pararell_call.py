# test_runner_call.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from rovibrational_excitation.simulation.runner import run_all, run_all_parallel

run_all_parallel("params_CO2_AntiSymm.py")
