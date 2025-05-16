import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
from rovibrational_excitation.core.electric_field import ElectricField, gaussian_fwhm
import numpy as np

duration = 20
amplitude = 1
carrier_frequency = 0.05

ti, tf, dt = 0, 1000, 0.1
time = np.arange(ti, tf+dt, dt)
tc = (time[-1]+time[0])/2

polarization = np.array(
    [1, 0]
)
Efield = ElectricField(tlist=time)

Efield.add_Efield_disp(
    envelope_func=gaussian_fwhm,
    duration=duration,
    t_center=tc,
    carrier_freq=carrier_frequency,
    amplitude=amplitude,
    polarization=polarization,
    gdd=-5e2,
    )

Efield.add_Efield_disp(
    envelope_func=gaussian_fwhm,
    duration=duration,
    t_center=tc+100,
    carrier_freq=carrier_frequency,
    amplitude=amplitude,
    polarization=[1, 0],
    gdd=-5e2,
    )


Efield.plot()