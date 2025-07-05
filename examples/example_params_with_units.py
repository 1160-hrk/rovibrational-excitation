"""
Example parameter file demonstrating flexible unit specification.

This example shows how to use various units for different physical quantities.
The runner will automatically convert all values to standard internal units.

Standard Internal Units:
- Frequency: rad/fs
- Dipole moment: C·m
- Electric field: V/m
- Energy: J
- Time: fs

Run with:
python -m rovibrational_excitation.simulation.runner examples/example_params_with_units.py
"""

from rovibrational_excitation.core.electric_field import gaussian_fwhm
import numpy as np

# Simulation description
description = "CO2_with_units_example"

# ================================
# MOLECULAR PARAMETERS WITH UNITS
# ================================

# Vibrational frequency (CO2 antisymmetric stretch)
omega_rad_phz = 100.0              # Value
omega_rad_phz_units = "THz"        # Unit: will be converted to rad/fs

# Anharmonicity (typical for CO2)
delta_omega_rad_phz = 12.3         # Value  
delta_omega_rad_phz_units = "cm^-1" # Unit: wavenumber → rad/fs

# Rotational constant (CO2)
B_rad_phz = 0.39021                # Value
B_rad_phz_units = "cm^-1"          # Unit: wavenumber → rad/fs

# Vibration-rotation coupling (small correction)
alpha_rad_phz = 0.0032             # Value
alpha_rad_phz_units = "cm^-1"      # Unit: wavenumber → rad/fs

# Transition dipole moment (typical for CO2 ν3 mode)
mu0_Cm = 0.3                       # Value
mu0_Cm_units = "D"                 # Unit: Debye → C·m

# ================================
# LASER FIELD PARAMETERS WITH UNITS
# ================================

# Laser intensity (strong field)
amplitude = 1e12                   # Value: 1 TW/cm²
amplitude_units = "W/cm^2"         # Unit: intensity → V/m electric field

# Alternative: direct electric field specification
# amplitude = 100                  # Value
# amplitude_units = "MV/cm"        # Unit: direct field → V/m

# Pulse duration  
duration = 50                      # Value
duration_units = "fs"              # Unit: femtoseconds

# Carrier frequency (resonant with CO2 ν3)
carrier_freq = 100                 # Value
carrier_freq_units = "THz"         # Unit: THz → rad/fs

# Time parameters
t_center = 200                     # Value
t_center_units = "fs"              # Unit: femtoseconds

t_start = 0                        # Value
t_start_units = "fs"               # Unit: femtoseconds  

t_end = 400                        # Value
t_end_units = "fs"                 # Unit: femtoseconds

dt = 0.1                          # Value
dt_units = "fs"                   # Unit: femtoseconds

# ================================
# QUANTUM SYSTEM PARAMETERS
# ================================

# Basis set size
V_max = 2                         # Vibrational levels: v = 0, 1, 2
J_max = 10                        # Rotational levels: J = 0-10
use_M = True                      # Include magnetic quantum numbers

# Initial state (ground vibrational, J=0 rotational)
initial_states = [0]              # Index of (v=0, J=0, M=0) state

# ================================
# SIMULATION PARAMETERS
# ================================

# Polarization (linear x-polarization)
polarization = [1.0, 0.0]         # [x, y] components

# Time evolution
return_traj = True                # Return full trajectory
sample_stride = 1                 # Sampling interval

# Computational backend
backend = "numpy"                 # "numpy" or "cupy"
dense = True                      # Dense matrices (faster for small systems)

# Optional: enable nondimensional calculation
nondimensional = False            # Use dimensionless units internally

# ================================
# ENVELOPE FUNCTION
# ================================

# Gaussian envelope (FWHM-parameterized)
envelope_func = gaussian_fwhm

# ================================
# PARAMETER SWEEPS (if desired)
# ================================

# Example: sweep laser intensity
# amplitude_sweep = [1e11, 5e11, 1e12, 5e12, 1e13]  # W/cm²
# amplitude_units = "W/cm^2"  # All sweep values use same unit

# Example: sweep pulse duration  
# duration_sweep = [10, 30, 50, 100, 200]  # fs
# duration_units = "fs"

# Example: sweep carrier frequency around resonance
# carrier_freq_sweep = np.arange(95, 105, 1)  # THz
# carrier_freq_units = "THz"

# ================================
# ALTERNATIVE UNIT EXAMPLES
# ================================

# Uncomment to try different units:

## European/atomic units
# omega_rad_phz = 0.159              # Frequency
# omega_rad_phz_units = "rad/fs"     # Already in standard unit

# mu0_Cm = 1.0                       # Dipole
# mu0_Cm_units = "ea0"               # Atomic units (electron × bohr)

## High-intensity laser
# amplitude = 5                      # Value  
# amplitude_units = "TW/cm^2"        # Terawatt intensity

## Spectroscopy units
# omega_rad_phz = 2350               # CO2 ν3 wavenumber
# omega_rad_phz_units = "cm^-1"      # Spectroscopic standard

# B_rad_phz = 0.39021                # CO2 rotational constant
# B_rad_phz_units = "wavenumber"     # Alias for cm^-1

## Time units
# duration = 0.05                    # Value
# duration_units = "ps"              # Picoseconds → fs

# t_end = 1                          # Value  
# t_end_units = "ns"                 # Nanoseconds → fs

## Energy units (for two-level systems)
# energy_gap = 0.5                   # Value
# energy_gap_units = "eV"            # Electron volt → J

# energy_gap = 4000                  # Value
# energy_gap_units = "cm^-1"         # Wavenumber → J

print("""
🔬 Parameter file loaded with unit specifications!

The following conversions will be performed automatically:
- Frequencies: THz, cm⁻¹ → rad/fs  
- Dipole moments: Debye → C·m
- Electric fields: W/cm² intensity → V/m field
- Times: various → fs
- Energies: eV, cm⁻¹ → J

Run the simulation to see the conversion messages.
""") 