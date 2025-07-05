"""
Example parameter file using the new default unit system.

Default units are set to commonly used values in molecular physics:
- Frequency: cm⁻¹ (spectroscopic standard)
- Dipole moment: D (Debye)
- Electric field: MV/cm (strong molecular fields)
- Intensity: TW/cm² (femtosecond laser pulses)
- Energy: eV (molecular and electronic energy scales)
- Area: cm² (molecular beam interactions)
- Time: fs (ultrafast dynamics)

Run with:
python -m rovibrational_excitation.simulation.runner examples/example_default_units.py
"""

from rovibrational_excitation.core.electric_field import gaussian_fwhm
import numpy as np
from rovibrational_excitation.core.units import (
    get_default_units,
    set_default_units,
    auto_convert_parameters,
    apply_default_units,
    print_unit_help
)

# Simulation description
description = "CO2_default_units_demo"

# ================================
# MOLECULAR PARAMETERS (default units)
# ================================

# Vibrational frequency (CO2 ν₃ mode)
omega_rad_phz = 2349.1            # cm⁻¹ (default frequency unit)
omega_rad_phz_units = "cm^-1"     # Explicitly specify for clarity

# Anharmonicity correction
delta_omega_rad_phz = 12.3        # cm⁻¹
delta_omega_rad_phz_units = "cm^-1"

# Rotational constant
B_rad_phz = 0.39021              # cm⁻¹
B_rad_phz_units = "cm^-1"

# Vibration-rotation coupling
alpha_rad_phz = 0.0032           # cm⁻¹
alpha_rad_phz_units = "cm^-1"

# Transition dipole moment (typical for CO2 ν3 mode)
mu0_Cm = 0.3                      # Debye (default dipole unit)
mu0_Cm_units = "D"

# ================================
# LASER FIELD PARAMETERS (default units)
# ================================

# Laser intensity using default intensity unit
amplitude = 5.0                   # TW/cm² (default intensity unit)
amplitude_units = "TW/cm^2"

# Alternative: direct electric field specification
# amplitude = 50               # MV/cm (default field unit)
# amplitude_units = "MV/cm"

# Pulse duration
duration = 30                     # fs (default time unit)
duration_units = "fs"

# Carrier frequency (resonant with CO2 ν3)
carrier_freq = 2349.1            # cm⁻¹ (matches molecular transition)
carrier_freq_units = "cm^-1"

# Time parameters
t_center = 100                    # fs
t_center_units = "fs"

t_start = 0                       # fs
t_start_units = "fs"

t_end = 200                       # fs
t_end_units = "fs"

dt = 0.1                         # fs
dt_units = "fs"

# ================================
# QUANTUM SYSTEM PARAMETERS
# ================================

# Basis set size
V_max = 2                        # Vibrational levels: v = 0, 1, 2
J_max = 8                        # Rotational levels: J = 0-8
use_M = True                     # Include magnetic quantum numbers

# Initial state (ground vibrational, J=0 rotational)
initial_states = [0]             # Index of (v=0, J=0, M=0) state

# ================================
# SIMULATION PARAMETERS
# ================================

# Polarization (linear x-polarization)
polarization = [1.0, 0.0]        # [x, y] components

# Time evolution
return_traj = True               # Return full trajectory
sample_stride = 1                # Sampling interval

# Computational backend
backend = "numpy"                # "numpy" or "cupy"
dense = True                     # Dense matrices

# Optional: enable nondimensional calculation
nondimensional = False           # Use dimensionless units internally

# ================================
# ENVELOPE FUNCTION
# ================================

# Gaussian envelope (FWHM-parameterized)
envelope_func = gaussian_fwhm

# ================================
# BEAM AND INTERACTION PARAMETERS (using area defaults)
# ================================

# Beam cross-sectional area (default area unit)
beam_area = 0.1                  # cm² (default area unit)
beam_area_units = "cm^2"

# Interaction cross-section (smaller scale)
cross_section = 1e-4             # cm² (effective interaction area)
cross_section_units = "cm^2"

# Alternative: microscopic areas
# cross_section = 100            # μm²
# cross_section_units = "μm^2"

# ================================
# ENERGY PARAMETERS (using energy defaults)
# ================================

# Electronic transition energy (default energy unit)
energy_gap = 1.5                 # eV (default energy unit)
energy_gap_units = "eV"

# Alternative energy scales
# thermal_energy = 25            # meV (room temperature)
# thermal_energy_units = "meV"

# ================================
# PARAMETER SWEEPS USING DEFAULT UNITS
# ================================

# Intensity sweep (all values in default intensity unit)
# amplitude_sweep = [1, 2, 5, 10, 20]    # TW/cm²
# amplitude_units = "TW/cm^2"            # Default intensity unit

# Frequency sweep around resonance (default frequency unit)
# carrier_freq_sweep = np.arange(2340, 2360, 2)  # cm⁻¹
# carrier_freq_units = "cm^-1"                   # Default frequency unit

# Dipole moment variation (default dipole unit)
# mu0_Cm_sweep = [0.1, 0.2, 0.3, 0.4, 0.5]     # Debye
# mu0_Cm_units = "D"                            # Default dipole unit

# ================================
# COMPARISON WITH OTHER UNIT SYSTEMS
# ================================

# Uncomment to try atomic units:
# mu0_Cm = 0.635                # Equivalent in atomic units
# mu0_Cm_units = "ea0"

# Uncomment to try SI field units:
# amplitude = 5e9               # Equivalent electric field
# amplitude_units = "V/m"

# Uncomment to try THz frequency units:
# omega_rad_phz = 70.4          # Equivalent frequency in THz
# omega_rad_phz_units = "THz"

print(f"""
🎯 Parameter file using default units:
   📊 Frequency: cm⁻¹ (molecular spectroscopy standard)
   🔌 Dipole: D (Debye units)
   ⚡ Field: MV/cm (strong molecular fields)
   💥 Intensity: TW/cm² (femtosecond laser pulses)
   ⚛️ Energy: eV (molecular and electronic energy scales)
   📐 Area: cm² (molecular beam interactions)
   ⏱️ Time: fs (ultrafast dynamics)

These defaults make the parameters more intuitive for molecular physics research!

Physical interpretation:
- CO₂ ν₃ mode at 2349.1 cm⁻¹ (standard IR spectroscopy)
- Dipole moment 0.3 D (typical for molecular transitions)
- Laser intensity 5 TW/cm² (strong field regime)
- 30 fs pulse duration (femtosecond laser)
- Beam area 0.1 cm² (typical laser spot size)
""")

# ================================
# CUSTOMIZING DEFAULT UNITS
# ================================

# To change default units for future parameter files:
# from rovibrational_excitation.core.units import set_default_units
# 
# # For atomic physics:
# set_default_units(frequency="THz", dipole="ea0", field="MV/cm")
#
# # For high-energy physics:
# set_default_units(energy="eV", frequency="THz", intensity="GW/cm^2")
#
# # For low-energy/cold molecule physics:
# set_default_units(frequency="GHz", energy="meV", field="kV/cm")

print("\n💡 Unit system demonstration completed successfully!")
print("   All parameters use their respective default units for intuitive specification.") 