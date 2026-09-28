# Unit-boundary audit

Last verified: 2026-09-28
Current checkpoint: Phase 4 closed; P7.3/D-111 preserves all decided optimizer units

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

D-069 moves the unchanged VibLadder frozen quantity boundary into its model
package. Required caller value/unit pairs, canonical `rad/fs` and C*m values,
and provenance are unchanged; no conversion is added, removed, or repeated.

D-080 moves the concrete dipole mixin unchanged to `models.dipole_base` and
introduces only a type-level access protocol in `core.dipole`. Existing
input-to-internal dipole conversion, SI views, and caller arrays are unchanged;
the protocol performs no conversion or validation.

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

`total_fs`, legacy `field_dt_fs`, standard Krotov `control_dt_fs`,
`segment_size_fs`, and returned `times_fs` are Class B. Their units are explicit in the names. The local
optimizer's odd-length field, endpoint handling, segment slices, midpoint
indices, and factor-of-two RK4 sampling are frozen contracts and must not
change during unit work.

### GRAPE/legacy fields and standard Krotov controls

D-050 governs GRAPE and `legacy_batch_overlap`. `initial_field_kind` explicitly selects `generated` or
`sampled`; no constructed field is silently replaced. Generated seeds are
Class A for duration, center, carrier frequency, amplitude, GDD, and TOD.
Primary pulse values and polarization are required; dispersion is an optional
complete pair or exact zero. Sampled seeds are Class A, use a direct field unit,
and must be real, finite, two-component, and exactly grid-matched. Legacy and
inapplicable keys raise. Frozen samples and the Krotov V=0 to V=3 reference are
unchanged. D-104 reuses that boundary for GRAPE because an exact
terminal-population gradient cannot leave the ordinary zero-field transfer
fixed point; no implicit zero seed is permitted.

D-106 gives standard Krotov a separate Class-A control boundary.
`initial_control_kind` explicitly chooses a generated pulse evaluated at interval
midpoints or a sampled `(N,2)` interval-control array with a required direct
amplitude unit. Exact interval count is required. The standard and legacy key
sets are mutually inapplicable and no resampling or schema conversion occurs.

### Local optimizer

Completed by D-051, D-058, and D-059. `field_max_v_per_m` remains a
private Class-B component limit with its fixed V/m representation and
historical `1e12` default. Local initialization is now an explicit required sum
type. `seed_field` is Class A: its required `amplitude/amplitude_units` pair
accepts only direct electric-field units and converts once to V/m, while its
positive `max_segments` count is dimensionless. The active `1000 V/m`, five-
segment values are unchanged. `none` owns no field quantity and fails before
propagation when the existing initial trigger is active. The former top-level
seed names raise with migration guidance.

Exact characterization preserves seed-before-componentwise-clipping order,
stored field, segment input, full RK4 input, odd prefix, tail endpoints, shared
boundaries, slices, and indices.

Local `gain` is no longer Class D. The user defined it as a strictly positive
field-squared-time quantity. It requires an adjacent `gain_units` label,
accepts only the four explicit V/m, MV/m, GV/m, and TV/m squared forms with
femtoseconds, and converts to canonical `(V/m)^2 fs` before the existing
update expression. The active `1000 (GV/m)^2 fs` value is exactly the former
`1e21` canonical value.

The following remain Class D and must not be renamed, converted, or assigned a
unit by inference:

- `c_abs_min`;
- `drive_abs_min`;
- `shape_floor`.

The user must define whether these are dimensionless, field-scaled, coupling
scaled, or expressed in another optimizer-specific normalization.

D-057 requires only that the remaining values have a finite real
representation. It
does not infer their units, permitted sign, physical range, or scaling.

### GRAPE, standard Krotov, and legacy penalties

GRAPE `lambda_a` and `learning_rate`, GRAPE/legacy convergence tolerances,
and the `legacy_batch_overlap` `lambda_a` remain Class D because their discrete
normalizations are not physical fluence definitions. D-104 fixes the GRAPE
objective and derivative but assigns no physical units to those values. D-057
therefore continues finite-real validation only for those routes.

D-106 resolves standard Krotov separately. Its update divides
`<chi|dH/dE|psi>`, with dipole coupling in `rad/fs/(V/m)`, by `lambda_a` to
produce V/m. With radians dimensionless, standard Krotov `lambda_a` is Class A:
the required `lambda_a/lambda_a_units` pair converts once to canonical
`1 / ((V/m)^2 fs)`. Supported labels use V/m, MV/m, GV/m, or TV/m inside the
squared field factor. This unit and its numerical value are not applied to the
legacy batch formula.

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

## Phase 4 disposition

All identified Class-A and Class-B public/configuration boundaries are migrated
and the Phase 4 acceptance checks pass. The remaining optimizer quantities are
Class D because their dimensions depend on unresolved objective or response
normalization. They are explicitly deferred and do not authorize guessed units,
renaming, conversion, or default changes. P5.1-c reuses an already converted
field-grid endpoint inside the numerical kernel and introduces no unit boundary
or conversion. P5.4-a audits solver capabilities and backend transfers without
changing any physical quantity or unit conversion.

P6.1-a found a separate internal conversion inconsistency rather than a
missing public unit. P6.1-b resolves it under the approved D-064:
`CONSTANTS.HBAR` is derived from exact Planck's constant, runtime aliases share
that value, and Hamiltonian conversion methods use the central converter. The
production TwoLevel builder now preserves its configured `rad/fs` gap across J
storage; no public unit field or conversion boundary changed.

P6.2-a adds an ownership-migration guard without changing the VibLadder unit
boundary. `VibLadderParameters` retains each caller frequency value and unit,
exposes canonical `angular_rad_per_fs`, and converts the explicitly unit-labeled
dipole once to C*m. The production builder still consumes only those canonical
values. Exact construction parity is now tested before any file move.

P6.2-b changes only module ownership and imports. The same basis constructor
still converts its explicit low-level input unit to rad/fs, the frozen schema
still supplies canonical rad/fs and C*m values, and the same Hamiltonian and
dipole objects cross the unchanged workflow boundaries. No value is converted,
renamed, defaulted, or reinterpreted by the move.

P6.1-c moves the characterized TwoLevel implementation into
`models/two_level/` without changing value/unit pairs, canonical units, or any
conversion function. `TwoLevelParameters` remains in `models/parameters.py`
until its own bounded ownership migration.

P6.1-d completes that migration to `models/two_level/parameters.py`. The class
continues to call the same finite-scalar, energy-or-frequency-unit, and dipole
conversion helpers, so no validation threshold, accepted unit, or canonical
value changes. Those helper bodies now live in private
`models/_parameter_validation.py` rather than the shared schema monolith.

## Recommended implementation order

1. Continue Class-D GRAPE, legacy-batch, Local-threshold, and convergence
   quantities only after user clarification and independent references.
2. Preserve the resolved Class-A standard Krotov penalty boundary under D-106.


P7.3 acceptance under D-111 changes no value or conversion. All optimizer
modules and the high-level runner are now strict-mypy targets, but the Class-D
quantities above remain deliberately undefined rather than receiving inferred
units. Their resolution is not required to claim that the already decided
Class-A/Class-B boundaries are preserved.

Every unit updates this audit, D-045 or a successor decision, the physics
contract, examples, and API inventory in the same commit.
