# Unit-boundary audit

Last verified: 2026-08-31
Current checkpoint: D-045 normal-simulation boundary complete

## Purpose

This document identifies physical values that still cross a public or
configuration boundary without an explicit unit after D-045. It is an audit,
not authorization to alter formulas. A migration may begin only after the
current numerical projection is characterized.

The target pattern is:

~~~text
caller value + required unit
  -> frozen validated boundary
  -> one conversion to a documented canonical unit
  -> unchanged numerical formula
~~~

Caller mappings remain unchanged for provenance. A unit encoded unambiguously
in an argument name, such as `field_dt_fs`, is acceptable while that API is
private and only one unit is supported. Neutral public names should use a
required paired unit.

## Classification

| Class | Meaning | Action |
|---|---|---|
| A | explicit value/unit pair and one canonical conversion | preserve |
| B | the only unit is explicit in the argument name | preserve or migrate mechanically |
| C | unit exists only in documentation, a default, or implementation knowledge | migrate |
| D | dimensions or normalization are not established | ask the user before editing |

## Completed boundary

Normal simulation is Class A under D-045. Model frequencies and dipole scale,
generated-field time, carrier, peak field, modulation delay, GDD, and TOD are
validated as explicit pairs. The input mapping and saved JSON retain the
submitted pair; frozen model and generated-field schemas contain canonical
values.

## Spectroscopy

### Current state

| Input | Current implicit unit | Class |
|---|---:|---|
| `ExperimentalConditions.temperature` | K | C |
| `ExperimentalConditions.pressure` | Pa | C |
| `ExperimentalConditions.optical_length` | m | C |
| `ExperimentalConditions.T2` | ps | C |
| `ExperimentalConditions.molecular_mass` | kg per molecule | C |
| `calculate(..., wavenumber)` and related spectrum methods | cm^-1 | C |
| `device_resolution` / `resolution` | cm^-1 | C |

The formulas already establish these units: number density uses Pa/(k_B K),
coherence decay converts ps to seconds, Beer-Lambert length is meters, Doppler
width uses kilograms per molecule, and the spectral grid is converted from
cm^-1. No alternative unit is currently supported at this boundary.

### Recommended P4.3-f contract

- Make every `ExperimentalConditions` value a required neutral value/unit
  pair: `temperature/temperature_units`, `pressure/pressure_units`,
  `optical_length/optical_length_units`,
  `coherence_time/coherence_time_units`, and
  `molecular_mass/molecular_mass_units`.
- Initially accept only the already implemented units K, Pa, m, ps, and kg.
  Requiring the labels removes ambiguity without introducing new conversion
  formulas.
- Store frozen canonical fields with unit-bearing internal names:
  `temperature_k`, `pressure_pa`, `optical_length_m`,
  `coherence_time_ps`, and `molecular_mass_kg`.
- Require `wavenumber_units="cm^-1"` for spectral arrays and
  `device_resolution_units="cm^-1"` when device broadening is requested.
- Preserve every spectroscopy formula and exact/approximate routing decision.

This is the recommended next implementation unit because the current units are
already explicit in formulas and docstrings and spectroscopy has strong
reference tests.

## Low-level electric-field API

### Current state

- `ElectricField(tlist, time_units="fs", field_units="V/m")` silently supplies
  both units.
- `add_dispersed_Efield` defaults duration, center, carrier, GDD, and TOD
  units. Its `amplitude` value is used as V/m without an amplitude-unit
  argument.
- `add_arbitrary_Efield` treats its array as canonical V/m without stating
  that at the call.
- `ScalarField` and `CartesianField` are already Class B because their
  sample attributes and constructor arguments say `v_per_m`.

### Recommended P4.3-g contract

- Remove unit defaults from direct `ElectricField` construction.
- Keep `from_time_grid` as an explicitly canonical constructor.
- Require an `amplitude_units` argument and convert amplitude exactly once.
- Require GDD/TOD value and unit together when present; omission of both
  retains the accepted exact-zero modifier.
- Require `field_units` on arbitrary field-array injection. Internal
  optimizer calls pass `"V/m"` explicitly.

Characterization must cover every current constructor and waveform reference
before changing signatures.

## Optimization

### Time grids

`total_fs`, `field_dt_fs`, `segment_size_fs`, and returned
`times_fs` are Class B. Their units are explicit in the names. The local
optimizer's odd-length field, endpoint handling, segment slices, midpoint
indices, and factor-of-two RK4 sampling are frozen contracts and must not
change during unit work.

### Krotov initial pulse

The current initial-pulse mapping has Class C inputs:
`duration_initial` and `t_center_initial` are fs,
`amplitude_initial` is V/m, `gdd_initial` is fs^2, and
`tod_initial` is fs^3. Carrier frequency has a value/unit pair but uses
legacy names. Several physical values have numerical defaults.

Recommended P4.3-h:

- add a frozen initial-pulse schema with neutral required pairs;
- require carrier, amplitude, duration, center, and polarization rather than
  selecting hidden physical defaults;
- retain the accepted optional zero GDD/TOD rule;
- preserve the exact sampled initial field before entering any Krotov update.

An externally supplied `efield_initial` should use the sampled-field contract
and must state V/m explicitly. It must never be resampled.

### Local optimizer

`field_max` and `seed_amplitude` are used as V/m but do not state units.
They can eventually become `field_max_v_per_m` and
`seed_amplitude_v_per_m` after exact characterization. This rename must not
change defaults, clipping, segment construction, endpoints, or indices.

The following are Class D and must not be renamed, converted, or assigned a
unit by inference:

- `gain`;
- `c_abs_min`;
- `drive_abs_min`;
- `shape_floor`.

The user must define whether these are dimensionless, field-scaled, coupling
scaled, or expressed in another optimizer-specific normalization.

### GRAPE and Krotov penalties

`learning_rate`, `lambda_a`, and the convergence tolerances may depend on
the exact objective and gradient normalization. They are Class D. Unit work
must wait for an independent objective/gradient reference and a user statement
of their intended dimensions.

### Optimization model construction

`simulation.optimize_runner` still uses legacy unit-encoded model names and
defaults for `input_units` and `output_units`. The long-term solution is to
reuse the frozen model schemas, not add a second converter. Migration requires
exact Hamiltonian, dipole, basis ordering, and optimizer-result parity.
`SymTop` must be handled separately because it is not represented by the
normal-simulation model schemas.

## Recommended implementation order

1. Spectroscopy explicit units (P4.3-f).
2. Low-level electric-field explicit units (P4.3-g).
3. Krotov initial-pulse explicit units (P4.3-h).
4. Local field-limit/seed key rename only after characterization.
5. Optimization model consolidation in the planned model/optimization phases.
6. Class-D optimizer quantities only after user clarification and independent
   references.

Every unit updates this audit, D-045 or a successor decision, the physics
contract, examples, and API inventory in the same commit.
