"""Minimal typed two-level propagation.

Tags: beginner, propagation, smoke
"""

import numpy as np

from rovibrational_excitation.simulation import run_simulation_case

PARAMS = {
    "basis_type": "twolevel",
    "energy_gap": 0.2,
    "energy_gap_units": "rad/fs",
    "dipole_scale": 3.0e-30,
    "dipole_scale_units": "C*m",
    "t_start": -10.0,
    "t_end": 10.0,
    "dt": 0.1,
    "duration": 4.0,
    "t_center": 0.0,
    "envelope_kind": "gaussian_fwhm",
    "modulation_kind": "none",
    "carrier_frequency": 0.05,
    "carrier_frequency_units": "PHz",
    "amplitude": 1.0e8,
    "initial_states": [0],
    "backend": "numpy",
    "storage": "dense",
    "algorithm": "rk4",
    "return_traj": True,
    "sample_stride": 1,
    "nondimensional": False,
    "renorm": False,
    "save": False,
}

population = run_simulation_case(PARAMS, field=None)
np.testing.assert_allclose(population.sum(axis=1), 1.0, rtol=1e-9, atol=1e-11)
print("final populations:", population[-1])
