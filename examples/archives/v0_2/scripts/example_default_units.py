"""Historical filename; current runner has no implicit model-frequency units.

This is a complete small LinMol example using spectroscopic units explicitly.
"""

description = "explicit_spectroscopic_units"

basis_type = "linmol"
representation = "m_resolved"
axes = "xy"
V_max = 2
J_max = 2
initial_states = [0]

vibrational_frequency = 2349.1
vibrational_frequency_units = "cm^-1"
anharmonic_shift = 0.0
anharmonic_shift_units = "cm^-1"
rotational_constant = 0.39021
rotational_constant_units = "cm^-1"
vibration_rotation_coupling = 0.0032
vibration_rotation_coupling_units = "cm^-1"
potential_type = "harmonic"
mu0_Cm = 0.3 * 3.33564e-30

t_start = -50.0
t_end = 50.0
dt = 0.1
envelope_kind = "gaussian_fwhm"
duration = 30.0
t_center = 0.0
modulation_kind = "none"
carrier_frequency = 2349.1
carrier_frequency_units = "cm^-1"
amplitude = 1.0e9
polarization = [1.0, 0.0]

backend = "numpy"
storage = "dense"
algorithm = "rk4"
nondimensional = False
renorm = False
return_traj = True
sample_stride = 1
save = False

# Equivalent ordinary-frequency form; ordinary frequency does not include 2*pi.
# vibrational_frequency = 70.424
# vibrational_frequency_units = "THz"
