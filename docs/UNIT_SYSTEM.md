# Flexible Unit System

The rovibrational excitation package now supports flexible unit specification for input parameters, allowing researchers to use the units most convenient for their work.

## Overview

### Supported Physical Quantities

| Quantity | Standard Internal Unit | Supported Input Units | Default Unit |
|----------|----------------------|---------------------|--------------|
| **Frequency/Angular Frequency** | rad/fs | THz, GHz, cm⁻¹, PHz, Hz, rad/s | cm⁻¹ |
| **Dipole Moment** | C·m | D (Debye), ea₀ (atomic units) | D |
| **Electric Field** | V/m | V/nm, MV/cm, kV/cm, W/cm², TW/cm², GW/cm² | MV/cm (field), TW/cm² (intensity) |
| **Energy** | J | eV, meV, μJ, mJ, nJ, pJ, cm⁻¹ | eV |
| **Area** | m² | cm², mm², μm², nm² | cm² |
| **Time** | fs | ps, ns, s | fs |

### How It Works

1. **Default Units**: System suggests commonly used units automatically
2. **Parameter Specification**: Add `_units` suffix to specify units
3. **Automatic Conversion**: Runner automatically converts to internal units
4. **Transparent Process**: Conversion messages show what was converted
5. **Backward Compatible**: Existing parameter files work unchanged

### Default Unit Philosophy

The default units are chosen based on common practice in molecular physics:

- **cm⁻¹**: Standard in IR/Raman spectroscopy
- **Debye (D)**: Molecular dipole moment standard
- **MV/cm**: Convenient for strong molecular electric fields
- **TW/cm²**: Typical range for strong laser fields
- **eV**: Standard energy unit for molecular and electronic systems
- **cm²**: Macroscopic beam and interaction areas

## Usage Examples

### Basic Usage

```python
# Parameter file with units
omega_rad_phz = 100.0           # Value
omega_rad_phz_units = "THz"     # Unit specification

B_rad_phz = 0.39               # Value  
B_rad_phz_units = "cm^-1"      # Wavenumber unit

mu0_Cm = 0.3                   # Value
mu0_Cm_units = "D"             # Debye unit

amplitude = 1e12               # Value
amplitude_units = "W/cm^2"     # Intensity unit
```

### Frequency/Angular Frequency

Most commonly used for molecular parameters:

```python
# Vibrational frequency
omega_rad_phz = 2349.1         # CO₂ ν₃ mode
omega_rad_phz_units = "cm^-1"  # Spectroscopic standard

# Alternative: THz units
omega_rad_phz = 70.4           # Equivalent frequency
omega_rad_phz_units = "THz"    # Frequency unit

# Rotational constant
B_rad_phz = 0.39021           # CO₂ rotational constant
B_rad_phz_units = "cm^-1"     # Standard spectroscopic unit
```

**Supported frequency units:**
- `"THz"` - Terahertz
- `"GHz"` - Gigahertz  
- `"cm^-1"`, `"cm-1"`, `"wavenumber"` - Wavenumbers (spectroscopy)
- `"PHz"` - Petahertz
- `"Hz"` - Hertz
- `"rad/s"` - Radians per second
- `"rad/fs"` - Radians per femtosecond (internal standard)

### Dipole Moments

```python
# Debye units (common in molecular physics)
mu0_Cm = 0.3                  # Typical for CO₂ ν₃ mode
mu0_Cm_units = "D"            # Debye

# Atomic units
transition_dipole_moment = 1.0
transition_dipole_moment_units = "ea0"  # electron × bohr radius

# SI units (no conversion needed)
mu0_Cm = 1e-30
mu0_Cm_units = "C*m"         # Coulomb-meter
```

**Supported dipole units:**
- `"D"`, `"Debye"` - Debye units (3.336 × 10⁻³⁰ C·m)
- `"ea0"`, `"e*a0"`, `"atomic"` - Atomic units
- `"C*m"`, `"C·m"`, `"Cm"` - Coulomb-meter (SI standard)

### Electric Field

#### Direct Field Units

```python
# High-field units
amplitude = 100               # Strong field
amplitude_units = "MV/cm"     # Megavolt per centimeter

# Alternative units  
amplitude = 1000              # Field strength
amplitude_units = "kV/cm"     # Kilovolt per centimeter

amplitude = 1e9               # Field strength
amplitude_units = "V/m"       # Volt per meter (SI)
```

#### Intensity Units (Common in Laser Physics)

```python
# Strong laser fields
amplitude = 1e12              # 1 TW/cm²
amplitude_units = "W/cm^2"    # Watt per square centimeter

amplitude = 5                 # 5 TW/cm²  
amplitude_units = "TW/cm^2"   # Terawatt per square centimeter

amplitude = 100               # 100 GW/cm²
amplitude_units = "GW/cm^2"   # Gigawatt per square centimeter
```

**Intensity to Field Conversion:**
E [V/m] = √(2 × I [W/m²] × μ₀ × c)

**Supported field units:**
- Direct field: `"V/m"`, `"V/nm"`, `"MV/cm"`, `"kV/cm"`
- Intensity: `"W/cm^2"`, `"MW/cm^2"`, `"GW/cm^2"`, `"TW/cm^2"`

### Time Parameters

```python
# Pulse duration
duration = 50                 # Pulse length
duration_units = "fs"         # Femtoseconds

duration = 0.1                # Alternative
duration_units = "ps"         # Picoseconds → fs

# Relaxation times
coherence_relaxation_time_ps = 100
coherence_relaxation_time_ps_units = "ns"  # Nanoseconds → ps

# Simulation time
t_end = 1                     # Simulation end
t_end_units = "ns"            # Nanoseconds → fs
```

**Supported time units:**
- `"fs"` - Femtoseconds (internal standard)
- `"ps"` - Picoseconds
- `"ns"` - Nanoseconds
- `"s"` - Seconds

### Energy (for Two-Level Systems)

```python
# Electronic transition energies
energy_gap = 1.5              # Bandgap energy
energy_gap_units = "eV"       # Electron volt

# Thermal energies
energy_gap = 25               # Room temperature
energy_gap_units = "meV"      # Milli-electron volt

# Spectroscopic energies
energy_gap = 10000            # High-energy transition
energy_gap_units = "cm^-1"    # Wavenumber
```

## Parameter Sweeps with Units

Units apply to all values in a sweep:

```python
# Intensity sweep
amplitude_sweep = [1e11, 5e11, 1e12, 5e12, 1e13]  # All in W/cm²
amplitude_units = "W/cm^2"                         # Applied to all

# Frequency sweep  
carrier_freq_sweep = np.arange(95, 105, 1)        # 95-105 THz
carrier_freq_units = "THz"                        # Applied to all

# Duration sweep
duration_sweep = [10, 30, 50, 100, 200]          # All in fs
duration_units = "fs"                             # Applied to all
```

## Advanced Examples

### CO₂ Molecule (Spectroscopic Units)

```python
# Standard spectroscopic parameters
omega_rad_phz = 2349.1        # ν₃ antisymmetric stretch
omega_rad_phz_units = "cm^-1"

delta_omega_rad_phz = 12.3    # Anharmonicity
delta_omega_rad_phz_units = "cm^-1"

B_rad_phz = 0.39021          # Rotational constant
B_rad_phz_units = "cm^-1"

alpha_rad_phz = 0.0032       # Vibration-rotation coupling
alpha_rad_phz_units = "cm^-1"

mu0_Cm = 0.3                 # Transition dipole moment
mu0_Cm_units = "D"           # Debye
```

### High-Intensity Laser Pulse

```python
# Strong laser field
amplitude = 5                # 5 TW/cm²
amplitude_units = "TW/cm^2"

duration = 30                # 30 fs pulse
duration_units = "fs"

carrier_freq = 100           # 100 THz carrier
carrier_freq_units = "THz"
```

### Atomic Units System

```python
# Frequency in atomic units
omega_rad_phz = 0.159        # Atomic frequency unit
omega_rad_phz_units = "rad/fs"  # Already in standard unit

# Dipole in atomic units  
mu0_Cm = 1.0                 # 1 ea₀
mu0_Cm_units = "ea0"

# Time in atomic units
duration = 24.2              # ≈ 1 atomic time unit
duration_units = "fs"
```

## Conversion Messages

When running simulations, the system provides feedback:

```
📊 Loading parameters from examples/params_with_units.py
✓ Converted omega_rad_phz: 100 THz → 628.319 rad/fs
✓ Converted B_rad_phz: 0.39021 cm^-1 → 0.0734712 rad/fs  
✓ Converted mu0_Cm: 0.3 D → 1.00069e-30 C·m
✓ Converted amplitude: 1e+12 W/cm^2 → 2.7435e+09 V/m
📋 Unit conversion completed.
```

## Physical Constants and Relationships

### Conversion Factors

```python
# Frequency conversions (to rad/fs)
THz_to_rad_fs = 2π × 10⁻³
cm_inv_to_rad_fs = 2π × c[cm/s] × 10⁻¹⁵
GHz_to_rad_fs = 2π × 10⁻⁶

# Dipole conversions (to C·m)
Debye_to_Cm = 3.33564 × 10⁻³⁰
ea0_to_Cm = e × a₀ = 8.478 × 10⁻³⁰

# Energy conversions (to J)
eV_to_J = 1.602176634 × 10⁻¹⁹
cm_inv_to_J = h × c × 100

# Field from intensity
E[V/m] = √(2 × I[W/m²] × μ₀ × c)
```

### Typical Values

| Quantity | Typical Range | Common Units |
|----------|--------------|--------------|
| **Vibrational frequency** | 1000-4000 cm⁻¹ | cm⁻¹, THz |
| **Rotational constant** | 0.1-10 cm⁻¹ | cm⁻¹ |
| **Dipole moment** | 0.1-3 D | Debye |
| **Laser intensity** | 10⁹-10¹⁵ W/cm² | W/cm², TW/cm² |
| **Pulse duration** | 10-1000 fs | fs, ps |

## Error Handling

### Invalid Units

```python
omega_rad_phz = 100
omega_rad_phz_units = "invalid_unit"  # Will show warning

# Output:
# ⚠ Warning: Unknown frequency unit: invalid_unit. 
#   Supported: ['rad/fs', 'THz', 'GHz', 'cm^-1', ...]
# Value remains unchanged: 100
```

### Missing Unit Specifications

Parameters without `_units` are used as-is:

```python
omega_rad_phz = 100          # No _units specified
# → Assumed to be in standard unit (rad/fs)
```

## Best Practices

1. **Always specify units** for clarity and reproducibility
2. **Use common units** for your field (cm⁻¹ for spectroscopy, THz for dynamics)
3. **Check conversion messages** to verify expected transformations
4. **Document your choice** of units in parameter files
5. **Be consistent** within parameter sweeps

## Migration from Old System

Existing parameter files work without changes:

```python
# Old style (still works)
omega_rad_phz = 0.628319     # rad/fs

# New style (equivalent)  
omega_rad_phz = 100          # THz
omega_rad_phz_units = "THz"
```

Both approaches produce identical results. 