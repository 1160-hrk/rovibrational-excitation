# Unit-boundary audit

Last verified: 2026-09-01
Current checkpoint: P4.3-i D-051 local field limit and seed units

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

Spectroscopy is Class A under D-046. Experimental conditions retain required
value/unit pairs and expose frozen canonical K, Pa, m, ps, and kg fields.
Every public spectral grid requires cm^-1 explicitly; device resolution is a
conditional value/unit pair. No spectroscopy formula or route was changed.

## Spectroscopy

### Completed P4.3-f state

| Input | Canonical unit | Class |
|---|---:|---|
| `ExperimentalConditions.temperature` | K | A |
| `ExperimentalConditions.pressure` | Pa | A |
| `ExperimentalConditions.optical_length` | m | A |
| `ExperimentalConditions.coherence_time` | ps | A |
| `ExperimentalConditions.molecular_mass` | kg per molecule | A |
| `calculate(..., wavenumber)` and related spectrum methods | cm^-1 | A |
| `device_resolution` / `resolution` | cm^-1 | A |

The formulas already establish these units: number density uses Pa/(k_B K),
coherence decay converts ps to seconds, Beer-Lambert length is meters, Doppler
width uses kilograms per molecule, and the spectral grid is converted from
cm^-1. No alternative unit is currently supported at this boundary.

### Implemented P4.3-f contract

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

All items above are implemented by D-046. Exact-route, Doppler, device,
polarization, phase-matching, radiation, and PFID reference tests remain green.

## Low-level electric-field API

### Current state

- Direct `ElectricField` and `ZeroField` construction is Class A under D-048:
  `time_units` is required, input is converted once to internal fs, and stored
  field samples have the fixed V/m unit. There is no constructor field value
  and therefore no `field_units` argument.
- `add_dispersed_Efield` is Class A under D-049: duration, center, carrier, and
  amplitude units are required and converted once. GDD/TOD are complete
  optional pairs; omission is exact zero. Intensity cannot label amplitude.
- `add_arbitrary_Efield` is now Class A under D-047: every array has a required
  direct amplitude unit and is converted once to V/m. Intensity units raise.
- `ScalarField` and `CartesianField` are already Class B because their
  sample attributes and constructor arguments say `v_per_m`.

### Recommended P4.3-g contract

- Remove unit defaults from direct `ElectricField` construction. Completed by
  P4.3-g unit 2.
- Keep `from_time_grid` as an explicitly canonical constructor. Completed by
  P4.3-g unit 2.
- Require an `amplitude_units` argument and convert amplitude exactly once.
  Completed by P4.3-g unit 3.
- Require GDD/TOD value and unit together when present; omission of both
  retains the accepted exact-zero modifier. Completed by P4.3-g unit 3.
- Require `field_units` on arbitrary field-array injection. Internal
  optimizer calls pass `"V/m"` explicitly. Completed by P4.3-g unit 1.

Constructor characterization covers missing, unsupported, seconds, fs, and
canonical `TimeGrid` inputs. Generated waveform characterization freezes
nonzero dispersion, explicit-zero equivalence, cross-unit equivalence, and
partial-pair rejection. P4.3-g is complete.

## Optimization

### Time grids

`total_fs`, `field_dt_fs`, `segment_size_fs`, and returned
`times_fs` are Class B. Their units are explicit in the names. The local
optimizer's odd-length field, endpoint handling, segment slices, midpoint
indices, and factor-of-two RK4 sampling are frozen contracts and must not
change during unit work.

### Krotov initial pulse

Completed by D-050. `initial_field_kind` explicitly selects `generated` or
`sampled`; no constructed field is silently replaced. Generated seeds are
Class A for duration, center, carrier frequency, amplitude, GDD, and TOD.
Primary pulse values and polarization are required; dispersion is an optional
complete pair or exact zero. Sampled seeds are Class A, use a direct field unit,
and must be real, finite, two-component, and exactly grid-matched. Legacy and
inapplicable keys raise. Frozen samples and the Krotov V=0 to V=3 reference are
unchanged.

### Local optimizer

Completed by D-051. `field_max_v_per_m` and `seed_amplitude_v_per_m` now state
their fixed V/m representation. The unit-ambiguous former names raise with the
replacement. Exact characterization preserves the `1e12` and `1e3` defaults,
seed-before-componentwise-clipping order, stored field, segment input, full RK4
input, odd prefix, tail endpoints, shared boundaries, slices, and indices.

The following are Class D and must not be renamed, converted, or assigned a
unit by inference:

- `gain`;
- `c_abs_min`;
- `drive_abs_min`;
- `shape_floor`.

The user must define whether these are dimensionless, field-scaled, coupling
scaled, or expressed in another optimizer-specific normalization.

D-057 requires only that these values have a finite real representation. It
does not infer their units, permitted sign, physical range, or scaling.

### GRAPE and Krotov penalties

`learning_rate`, `lambda_a`, and the convergence tolerances may depend on
the exact objective and gradient normalization. They are Class D. Unit work
must wait for an independent objective/gradient reference and a user statement
of their intended dimensions. D-057 likewise adds finite-real validation only.

### Optimization model construction

D-054 completes this boundary for LinMol, VibLadder, and TwoLevel. The optimizer
uses neutral physical names with required adjacent units, converts through the
frozen production schemas, and calls the same model-owned operator builders.
The historical rad/fs optimizer Hamiltonian projection is retained. Basis
ordering, H0, SI dipoles, and the stored Krotov result are characterized;
differences are limited to sub-ulp unit-conversion roundoff. Legacy unit-encoded
names and shared input/output unit defaults raise.

SymTop optimization still raises. The model now has a shared production builder,
but no independent SymTop optimization objective/control reference exists; a
structural migration must not invent one. D-055 archives the tracked legacy
optimizer YAML files that lack dipole values rather than assigning an inferred
constant. Three active current-schema configs provide explicit dipole values
and units.

## Recommended implementation order

1. Class-D optimizer quantities only after user clarification and independent
   references.

Every unit updates this audit, D-045 or a successor decision, the physics
contract, examples, and API inventory in the same commit.
