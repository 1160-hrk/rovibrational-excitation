# Explicit frequency units

Frequency-bearing runner inputs use a neutral quantity name and a required
paired `*_units` field. There is no default frequency unit and no warning-based
fallback. Missing units, unknown units, booleans, arrays, and non-finite values
raise before model or field allocation.

## Supported frequency representations

| Representation | Units | Does the value contain `2*pi`? |
|---|---|---|
| ordinary frequency | `Hz`, `kHz`, `MHz`, `GHz`, `THz`, `PHz` | no |
| wavenumber | `cm^-1`, `cm-1`, `wavenumber` | no |
| angular frequency | `rad/s`, `rad/ps`, `rad/fs` | yes |

All accepted representations are normalized exactly once to internal angular
frequency in `rad/fs` by the immutable `core.units.Frequency` boundary. FFT
consumers request an explicit ordinary-frequency view in cycles/fs. Numerical
model and propagation kernels do not inspect user unit strings.

## Model parameters

LinMol requires:

```python
vibrational_frequency = 2349.1
vibrational_frequency_units = "cm^-1"
anharmonic_shift = 12.3
anharmonic_shift_units = "cm^-1"
rotational_constant = 0.39021
rotational_constant_units = "cm^-1"
vibration_rotation_coupling = 0.0032
vibration_rotation_coupling_units = "cm^-1"
```

VibLadder requires the first two value/unit pairs. TwoLevel instead requires
`energy_gap` and `energy_gap_units`; that field accepts the frequency and energy
units supported by the converter. A Morse potential requires a nonzero
`anharmonic_shift` after conversion.

The old runner names `omega_rad_phz`, `delta_omega_rad_phz`, `B_rad_phz`, and
`alpha_rad_phz` are removed. Supplying an old value or old `_units` key raises a
migration error. Low-level basis constructors may still use angular-frequency
argument names; those are direct numerical APIs, not runner configuration.

## Generated carrier

```python
carrier_frequency = 2349.1
carrier_frequency_units = "cm^-1"
```

The pulse phase receives the canonical angular value. Sinusoidal FFT modulation
receives its explicitly converted cycles/fs center because FFT bins are ordinary
frequency. The separate legacy `carrier_freq_sin_mod` coefficient has not been
renamed: its dimensional meaning is unresolved and must not be inferred.

## Equivalent inputs

These values describe the same ordinary frequency, within printed precision:

```python
vibrational_frequency = 100.0
vibrational_frequency_units = "THz"

# or
vibrational_frequency = 0.1
vibrational_frequency_units = "PHz"

# or the angular value
vibrational_frequency = 2 * np.pi * 0.1
vibrational_frequency_units = "rad/fs"
```

The first two values do not contain `2*pi`; the last one does.

## Other quantities

The legacy `ParameterProcessor` still converts supported time, electric-field,
dipole, dispersion, and area parameters where that API is used. It deliberately
does not pre-convert the new model or carrier frequency fields. This prevents a
canonical number from being paired with a stale input-unit label and converted
twice. Unknown units raise rather than leaving a value unchanged.

For a complete current runner configuration, copy
`examples/params_template.py`. The focused unit and equivalence contracts live
in `tests/unit/test_unit_conversions.py`, `tests/test_simulation_models.py`, and
`tests/physics/test_linear_molecule_reference.py`.
