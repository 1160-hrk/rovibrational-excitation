import sys
sys.path.append('..')
from src.rovibrational_interaction_simulation.core.electric_field import ElectricField, gaussian_fwhm
import numpy as np

duration = 20
amplitude = 1
carrier_frequency = 0.2

ti, tf, dt = 0, 1000, 0.1
time = np.arange(ti, tf+dt, dt)
tc = (time[-1]+time[0])/2

param_envelope = [tc, duration]

polarization = np.array(
    [1, 0]
)
Efield = ElectricField(
    tlist=time,
    envelope_func=gaussian_fwhm,
    param_env=param_envelope,
    carrier_freq=carrier_frequency,
    amplitude=amplitude,
    polarization=polarization,
    gdd=2e2
    )

Efield.plot()