"""Complete LinMol runner example with mixed explicit frequency units."""

import numpy as np

description = "mixed_explicit_frequency_units"

basis_type = "linmol"
representation = "m_resolved"
axes = "xy"
V_max = 2
J_max = 2
initial_states = [0]

# Ordinary THz input has no 2*pi. Wavenumber input also has no 2*pi.
vibrational_frequency = 70.424
vibrational_frequency_units = "THz"
anharmonic_shift = 12.3
anharmonic_shift_units = "cm^-1"
rotational_constant = 0.39021
rotational_constant_units = "wavenumber"
vibration_rotation_coupling = 0.0032
vibration_rotation_coupling_units = "cm-1"
potential_type = "harmonic"
mu0_Cm = 0.3 * 3.33564e-30

t_start = -50.0
t_end = 50.0
dt = 0.1
envelope_kind = "gaussian_fwhm"
duration = 30.0
t_center = 0.0
modulation_kind = "none"
carrier_frequency = 70.424
carrier_frequency_units = "THz"
amplitude = 1.0e9
polarization = [1.0 / np.sqrt(2.0), 1.0j / np.sqrt(2.0)]

backend = "numpy"
storage = "dense"
algorithm = "rk4"
nondimensional = False
renorm = False
return_traj = True
sample_stride = 1
save = False
