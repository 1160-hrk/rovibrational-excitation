import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from rovibrational_excitation.core.electric_field import ElectricField, gaussian_fwhm
import numpy as np

duration = 100
amplitude = 1
carrier_frequency = 0.03
gdd = 5e3

ti, tf, dt = 0, 3000, 0.01
time = np.arange(ti, tf+dt, dt)
tc = (time[-1]+time[0])/2

polarization = np.array(
    [1, 0]
)
Efield = ElectricField(tlist=time)

Efield.add_dispersed_Efield(
    envelope_func=gaussian_fwhm,
    duration=duration,
    t_center=tc,
    carrier_freq=carrier_frequency,
    amplitude=amplitude,
    polarization=polarization,
    gdd=gdd,
    )

Efield.plot()
width_freq = 0.015
Efield.plot_spectrum(
    freq_range=(carrier_frequency-width_freq, carrier_frequency+width_freq), remove_linear_phase=True,
    t_center=tc,
    center_freq=carrier_frequency, width_fit=0.01
    )

# Efield.init_Efield()
# Efield.add_dispersed_Efield(
#     envelope_func=gaussian_fwhm,
#     duration=duration,
#     t_center=tc,
#     carrier_freq=carrier_frequency,
#     amplitude=amplitude,
#     polarization=polarization
#     )

# Efield.plot()
# Efield.plot_spectrum(freq_range=(0, 0.3), remove_linear_phase=False,
#                     #  center_freq=carrier_frequency, width_fit=0.02
#                      )



binned_spectrum = np.array([1, 0])