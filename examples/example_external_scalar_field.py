"""Two-level propagation with externally sampled scalar field injection.

Tags: beginner, propagation, external-field, smoke
"""

import numpy as np

from rovibrational_excitation.core.time import TimeGrid
from rovibrational_excitation.fields import ScalarField
from rovibrational_excitation.simulation import run_simulation_case

TIME_GRID = TimeGrid.from_bounds(-10.0, 10.0, 0.1)
times_fs = TIME_GRID.field_times_fs
samples_v_per_m = 1.0e8 * np.exp(-4.0 * np.log(2.0) * (times_fs / 4.0) ** 2)
FIELD = ScalarField(TIME_GRID, samples_v_per_m)

PARAMS = {
    "basis_type": "twolevel",
    "energy_gap": 0.2,
    "energy_gap_units": "rad/fs",
    "dipole_scale": 3.0e-30,
    "dipole_scale_units": "C*m",
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

population = run_simulation_case(PARAMS, field=FIELD)
np.testing.assert_allclose(population.sum(axis=1), 1.0, rtol=1e-9, atol=1e-11)
print("final populations:", population[-1])
