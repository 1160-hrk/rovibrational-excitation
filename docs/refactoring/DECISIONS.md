# Refactoring decision log

Last updated: 2026-08-27

## How to use this log

- `Accepted` decisions are binding for refactoring.
- `Proposed` decisions describe the roadmap but may be revised before the
  affected phase begins.
- `Open` items require explicit user input before behavior changes.
- When a decision is superseded, keep the old entry, mark it `Superseded`, and
  link the replacement ID.
- Every physics-affecting commit must reference or add a decision ID in its
  description or accompanying documentation.

## Accepted decisions

### D-001: Backward compatibility is not required

Status: Accepted
Scope: Public Python API and repository structure

The repository currently has one user. Old import paths, constructor
signatures, parameter aliases, and serialized layouts may be broken when the
new design is clearer.

Consequences:

- Do not preserve wrappers solely for hypothetical external users.
- Do preserve scientific behavior unless a separate physics decision approves
  a change.
- Configuration and result schema changes must still be versioned so old
  scientific results remain interpretable.
- Remove compatibility adapters by the end of the phase that introduces them.

### D-002: Numerical logic preservation takes priority during restructuring

Status: Accepted
Scope: All phases

Structural work must keep the original calculation logic as far as possible.
When physical meaning or intended behavior is uncertain, Codex must ask the
user rather than decide.

Consequences:

- Characterization tests precede implementation replacement.
- Formatting, moves, API changes, and formula changes use separate commits.
- “Cleaner” is not sufficient justification for changing signs, normalization,
  thresholds, axes, sampling, or derived parameters.

### D-003: Morse with zero anharmonicity is invalid

Status: Accepted
Scope: LinMol, VibLadder, SymTop Morse construction

`potential_type="morse"` with `delta_omega == 0` must raise a clear error.
Falling back to harmonic behavior is forbidden.

Implementation anchor: `dipole/vib/morse.py` and simulation model validation.

### D-004: Morse level parameter is derived locally

Status: Accepted
Scope: Morse transition dipoles and basis validation

The former conceptual `N=200` is not a universal constant. The level parameter
is derived from each model's frequency and anharmonic shift:

~~~text
N = (omega01 + delta_omega) / delta_omega - 1/2
~~~

It must be stored on or passed through the relevant instance/call. Global
mutable Morse state is forbidden.

### D-005: Backend selection must be consistent and honest

Status: Accepted
Scope: Dipole construction and propagation

The same simulation backend selection applies to dipole construction and time
propagation unless a future explicit transfer boundary is introduced.

Consequences:

- Reject unsupported combinations before array conversion.
- Do not claim CuPy support for Liouville while the implementation converts to
  NumPy.
- Do not silently fall back from CuPy to NumPy.
- Separate backend implementations are acceptable when their input/output
  contract and parity tests are unified.

Implementation status (2026-07-31):

- RK4 pure-state propagation now keeps the same low-level final-only shape,
  `(1, dimension)`, for NumPy and CuPy paths. A CPU-runnable mocked-CuPy
  contract test protects the dispatch boundary.
- A real NumPy/CuPy numerical parity test exists under the `gpu` marker, but
  remains unverified locally until it runs on a CUDA-capable environment.

### D-006: Physically defining parameters should be required

Status: Accepted
Scope: Configuration and typed problem construction

Parameters whose omission can produce extreme or meaningless calculations
should be required. Safe representation defaults remain allowed.

The exact required field set is model- and field-construction-specific and is
defined in `PHYSICS_CONTRACTS.md` and the future typed configuration schema.

### D-007: Runner list input is coherent

Status: Accepted
Scope: `initial_states` in ordinary simulation configuration

Multiple basis indices form an equal-amplitude, equal-phase normalized coherent
superposition. They are not an incoherent population sum.

### D-008: Incoherent mixtures use a dedicated propagation path

Status: Accepted
Scope: `MixedStatePropagator`

An `IncoherentEnsemble` is propagated component by component. Vector norm
squared provides each raw statistical weight. Weights are normalized to sum to
one before density operators are summed. Raw list input is rejected at the
propagator boundary.

No coherent cross terms are introduced.

### D-009: Electric-field grid uses half propagation steps

Status: Accepted
Scope: RK4, split operator, returned time arrays

The configured field spacing is the half-step required for left/mid/right
sampling. One state update advances twice that interval:

~~~text
propagation_dt = 2 * field_dt
~~~

Time arrays must report the actual state-update time, not the field half-step.

### D-010: TwoLevel and VibLadder use scalar coupling

Status: Accepted
Scope: Model construction and polarization

These models have no physical polarization degree of freedom in the current
library. Their excitation result must not depend on which polarization vector
the configuration contains.

Current `x` and `z` axes are storage conventions. The target API will represent
the coupling as scalar.

### D-011: Interaction sign is minus

Status: Accepted
Scope: All propagation algorithms

All solvers use:

~~~text
H(t) = H0 - mu E(t)
~~~

Liouville and Schrödinger must match for a pure-state density operator.

Implemented in commit `613ce93`.

### D-012: Density matrices receive scale-aware physical validation

Status: Accepted
Scope: Mixed-state and Liouville input

Density matrices must be finite, square, Hermitian, positive semidefinite, and
have positive real trace. The numerical threshold is:

~~~text
tol = 100 * max(1, dimension) * machine_epsilon * spectral_norm
~~~

A negative eigenvalue within this bound is treated as roundoff.

Implemented in commit `613ce93`.

### D-013: Density validation does not silently repair input

Status: Accepted
Scope: Density matrix validation

Validation may accept roundoff-scale deviations but does not clip
eigenvalues, symmetrize, project, or normalize a matrix. `DensityState` requires
trace one, and mixed-state propagation passes the validated matrix unchanged.

### D-014: Refactoring is staged, not a big-bang rewrite

Status: Accepted for planning
Scope: Repository-wide execution

The sequence is:

1. physics characterization;
2. repository and CI normalization;
3. typed contracts;
4. package migration;
5. units/nondimensionalization;
6. solver and model consolidation;
7. workflow decomposition;
8. public API and release.

Each phase has independent acceptance criteria in `EXECUTION_PLAN.md`.

### D-015: AGENTS.md routes Codex to authoritative documents

Status: Accepted
Scope: Agent workflow

Codex reads root `AGENTS.md` first. Detailed physical, architecture, and phase
information lives under `docs/refactoring/`. Whenever implementation changes a
documented contract, the corresponding document changes in the same commit.

### D-016: Vibrational omega is the fundamental transition frequency

Status: Accepted
Scope: VibLadder Hamiltonian construction

`omega`/`omega01` is the angular frequency of the `v=0 -> 1` transition.
`delta_omega` is the per-level decrease in adjacent transition frequency.

The vibrational energies are:

~~~text
E_v = (omega01 + delta_omega) (v + 1/2)
      - (delta_omega / 2) (v + 1/2)^2
E_(v+1) - E_v = omega01 - v delta_omega
~~~

Stored-parameter and temporary-override Hamiltonian generation must call the
same implementation. The earlier override path used a different formula and
was incorrect.

### D-017: Reduced LinMol uses fixed-linear M-block averaging

Status: Accepted
Scope: LinMol magnetic-degeneracy representation, polarization, initial states,
propagation, and results

`use_M=True` is the explicit `|v,J,M>` Cartesian model. `use_M=False`
means a qualitative, lower-cost calculation that averages unresolved magnetic
degeneracy. It is not an `M=0` pure-state approximation.

For `use_M=False`, any fixed linear laboratory polarization is accepted. Its
Jones vector is normalized, a common complex phase is removed, and the
quantization axis is aligned with that direction. Propagation then uses only
the internal z component, so `Delta M=0`.

For an initial rotational quantum number `J0`, each M component has
statistical weight `1/(2 J0 + 1)`. Fixed-M blocks evolve separately and
populations are summed incoherently. Because the z-coupling matrix is identical
for `+M` and `-M`, only non-negative `|M|` representatives are propagated:
`M=0` has multiplicity one and `M>0` has multiplicity two.

Consequences:

- circular, elliptical, and time-dependent polarization are rejected;
- `axes` is not applicable and is rejected instead of ignored;
- equal-amplitude coherent initial states are supported only when every
  selected reduced state has the same J; coherence across v is retained;
- a coherent selection spanning different J is rejected because an isotropic
  M average does not define unique cross-J coherences;
- the returned population index is the reduced `(v,J)` ordering;
- serialized results identify `m_incoherent_average`, store representative
  block wavefunctions and weights, and do not store a fictitious aggregate
  `psi`;
- constructing a Cartesian `LinMolDipoleMatrix` from a basis without explicit
  M quantum numbers is an error;
- backend selection remains common to block dipole construction and block time
  propagation under D-005.

The normal-simulation configuration names above were replaced by D-041:
`m_resolved` maps exactly to the former `use_M=True` branch and
`m_incoherent_average` maps exactly to the former `use_M=False` branch.
Direct `LinMolBasis` construction and optimization retain their separately scoped
internal `use_M` flag during migration.

The fixed-linear test tolerance is
`128 * machine_epsilon` after Jones-vector normalization. It distinguishes
roundoff from a physical relative phase without introducing a field-scale
threshold.

Implementation anchors:
`simulation/m_average.py`,
`simulation/runner.py`, and
`tests/physics/test_linear_molecule_reference.py`.

### D-018: RK4 matrix storage is explicit and CSR propagation is JIT-compiled

Status: Accepted
Scope: NumPy RK4 dense/sparse dispatch, CSR preparation, and numerical policy

The previous dispatch sent `sparse=True` and SciPy CSR inputs through a
Python/SciPy RK4 loop, while dense inputs were scanned and converted to CSR
inside a Numba function. The public storage choice therefore did not describe
the executed kernel and repeated dense-to-CSR scans obscured performance.

`sparse=True` now means that each operator is copied into canonical SciPy CSR
once before propagation and its contiguous `data`, `indices`, and `indptr`
arrays are passed to a fused Numba RK4 kernel. Sparse operator input without
`sparse=True` is rejected rather than inferred. `sparse=False` uses the
allocation-stable dense Numba kernel.

CSR preparation sums duplicate entries, removes stored exact zeros, and sorts
indices without mutating caller-owned matrices. It performs no tolerance-based
truncation. Approximate sparsification would change the operator and therefore
requires a separate explicit policy and user-approved threshold.

The sparse Hamiltonian application preserves `H0 - mu_x*Ex - mu_y*Ey` and
does not use `fastmath`. Dense `fastmath` is retained only after invariant and
dense/sparse parity tests found final differences below the documented
tolerances. Opt-in renormalization raises for a zero or non-finite norm instead
of skipping trajectory storage.

Consequences:

- CSR matrices remain sparse from model construction through the hot loop;
- final-only propagation allocates only one returned state;
- field stages are indexed directly as left, midpoint, and right samples;
- the current stride endpoint behavior remains governed by O-001;
- dense and sparse implementations stay separate inside the hot loop;
- `tests/physics/test_sparse_rk4_reference.py` is the numerical anchor.

Implementation commit: `6e154ec`

### D-019: Split-operator polarization models are explicit

Status: Accepted
Scope: Schrödinger split propagation for fixed and complex polarization

The old CPU path kept the upper triangle of a complex Cartesian dipole
combination and added its adjoint. Later changes cast the Jones vector to
float and replaced that construction with an average of the full matrix.
That discarded helicity and made the CPU and GPU physics inconsistent.

The default `split_interaction="cartesian"` uses the same Hamiltonian as RK4:

~~~text
H(t) = H0 - mu_x Ex(t) - mu_y Ey(t).
~~~

For an M-resolved LinMol xy operator,
`D(phi) mu_x D(phi)^dagger = cos(phi) mu_x + sin(phi) mu_y`, with
`D_nn(phi) = exp(i M_n phi)`. The split kernel diagonalizes `mu_x` once and
applies `D`, the spectral interaction exponential, and `D^dagger` at each
midpoint. This keeps two dense matrix-vector products per propagation step.
A fixed real field direction uses one static Hermitian interaction and needs
no M labels.

The explicit `split_interaction="helicity_projected"` approximation builds
`T = triu(-p_x mu_x - p_y mu_y, k=1)` and uses `T + T^dagger`, without a
factor of one half. Under the current carrier and tensor convention,
`p=(1,+i)/sqrt(2)` selects resonant Delta M=+1 absorption and the opposite
sign selects Delta M=-1. This construction is a defined one-way transition
model, not permission to repair arbitrary non-Hermitian input.

Consequences:

- Cartesian is the default and must converge to RK4 with second-order Strang error;
- helicity-projected must be requested explicitly and may differ for strong or ultrashort fields;
- complex Jones vectors remain complex and normalized at the field boundary;
- component dipoles are validated as Hermitian with a scale-aware roundoff tolerance;
- changing Cartesian direction requires M labels and verified xy rotation covariance;
- spectral eigenvectors are dense even when the input operators are sparse;
- CPU and GPU paths implement the same interaction construction and final-state shape;
- `tests/physics/test_split_operator_polarization.py` is the physics anchor.

Implementation commit: `93ee9eb`

### D-020: Nondimensionalization never invents missing scales or time grids

Status: Accepted
Scope: dynamics/scaling, propagation preparation, returned wavefunction phase

A zero Hamiltonian, zero transition dipole, or zero electric field previously
triggered arbitrary replacements corresponding to 1 fs, 1 Debye, or
1e8 V/m. The energy helper could also cap the derived time scale at 1000 fs,
and auto_timestep could resample the caller's field grid using empirical
coupling thresholds. These choices changed the normalized generator without a
scientific error bound and were not visible in the result.

For a finite Hermitian free Hamiltonian and the coupling components active in
the selected propagation mode, define

~~~text
epsilon_min = min eig(H0)
H0_centered = H0 - epsilon_min I
Delta_H = max eig(H0) - min eig(H0)
mu_ref = max_a ||mu_a||_2
E_ref_field = max_t ||E(t)||_2
V_ref = mu_ref E_ref_field
E_ref = max(Delta_H, V_ref)
t_ref = hbar / E_ref
~~~

A caller may replace only E_ref with an explicit positive energy_scale_J; its
provenance is recorded as explicit. The numerical interaction coefficient is
V_ref / E_ref. The physical coupling ratio is separate: V_ref / Delta_H when
Delta_H > 0, undefined for a driven gapless system, and zero for an explicit
field-free system.

An identically zero ordinary ElectricField is ambiguous and raises. ZeroField
is the explicit field-free type. Its field scale is inactive (None) and its
normalized samples and interaction coefficient are zero. A driven system with
a zero coupling operator raises. A zero coupling operator is allowed with
ZeroField and has an inactive dipole scale. A completely zero generator has no
low-level characteristic scale and raises; a future high-level
trivial-evolution shortcut must be explicit.

The free Hamiltonian is normalized after energy-origin centering. Schrodinger
propagation restores the exact global factor

~~~text
exp(-i epsilon_min (t - t_start) / hbar)
~~~

on both trajectories and final states. Density propagation needs no correction
because the global phase cancels. Thus absolute dimensional and
nondimensional wavefunctions, not only populations, remain comparable.

Consequences:

- full eigenspectra and operator 2-norms are used; diagonal-only and
  off-diagonal-only shortcuts are forbidden;
- component Hamiltonians and dipoles are validated as finite and Hermitian;
- no 1000 fs cap, 1 fs, 1 Debye, or 1e8 V/m fallback remains;
- heuristic weak/intermediate/strong labels are not emitted without a
  model-specific accepted threshold;
- scale values carry derived, explicit, or inactive provenance;
- auto_timestep raises at propagation boundaries; obsolete recommendation
  helpers are absent from the scaling API;
- explicit time-array construction requires a positive step that divides the
  requested duration and never extends the endpoint;
- tests/contracts/test_strict_nondimensional_contracts.py anchors scale,
  zero-state, gapless, and absolute-phase behavior.

Implementation commit: `4c33359`

### D-021: Requested capabilities and options never silently fall back

Status: Accepted
Scope: propagation options, backend selection, scaling, configuration boundaries

A requested physical model, numerical algorithm, backend, scale, or time grid
must either execute as requested or raise before propagation. A removed option
must raise migration guidance even when its supplied value equals the old
default; accepting it would hide stale configuration. Unknown propagation
kwargs must raise rather than disappear into a variadic signature.

An optional acceleration dependency may use a slower implementation only when
the selected numerical backend, array type, equations, and result contract are
unchanged. Such a performance fallback must be observable through capability
reporting. It must never turn an explicit CuPy request into NumPy.

Batch runners may isolate a failed case only when they retain a structured
failure record and traceback. Validation errors, unit errors, and optimization
rule changes must not be converted into successful results or warning-only
execution.

Consequences:

- removed auto_timestep and target_accuracy options raise at every public route;
- Schrodinger, Liouville, and MixedState reject unknown propagation kwargs;
- dipole backend selection raises when CuPy is requested but unavailable;
- strict validation may not fall back to raw unit-ambiguous attributes;
- remaining findings are tracked in FALLBACK_AUDIT.md;
- physics-bearing configuration defaults require a separate user decision.

Implementation commit: `4c33359`

### D-022: Physical inputs and scaling semantics are explicit

Status: Accepted
Scope: basis and dipole construction, simulation configuration, nondimensionalization API

Omitting a physical constant or pulse width must not be indistinguishable from
intentionally choosing zero. Zero remains a valid physical value where the
model permits it, but it must be written explicitly.

The simulation contract requires:

- every model: `mu0_Cm`;
- LinMol: `V_max`, `J_max`, `omega_rad_phz`,
  `delta_omega_rad_phz`, `B_rad_phz`, `alpha_rad_phz`, and
  `potential_type`;
- VibLadder: `V_max`, `omega_rad_phz`, `delta_omega_rad_phz`, and
  `potential_type`;
- TwoLevel: `energy_gap` and `energy_gap_units`;
- every pulse-driven simulation: `duration`.

The basis constructors enforce the corresponding constants when called
directly, including `omega`, `B`, `C`, `alpha`, and `delta_omega` for
the experimental SymTop basis. Direct dipole construction requires `mu0`;
vibrational dipoles additionally require `potential_type`, while TwoLevel
rejects that inapplicable option. `pulse_duration` is removed rather than
aliased or converted. Krotov requires a positive finite `duration_initial`.

Array-based nondimensionalization requires explicit Hamiltonian and time units.
Object-based nondimensionalization requires the active coupling axes and
scalar-versus-Cartesian coupling mode. There is one scaling representation:
strict generator scaling from D-020. Competing lambda-absorption strategies,
automatic timestep wrappers, heuristic verification, demo parameter factories,
and compatibility re-export modules are deleted. `analyze_regime` remains a
neutral report and does not assign universal strength thresholds.

Consequences:

- omitted values raise before model construction or propagation;
- explicit `0.0` is preserved without replacement;
- no half-window duration fallback or implicit harmonic-potential selection;
- the public nondimensional namespace contains only strict transformation,
  scale metadata, exact conversion helpers, and neutral reporting;
- tests cover direct basis and dipole calls, simulation validation, and
  dimensional equivalence.

Implementation commit: `7d14fda`

### D-023: Spectroscopy numerical policies are explicit

Status: Accepted on 2026-08-10.

Scope: spectroscopy response evaluation, broadening, and experimental inputs.

Decision:

- `ExperimentalConditions` requires positive finite temperature, pressure,
  optical length, dephasing time, and molecular mass. None is invented.
- Callers explicitly select `matrix`, `loop`, `2d`, `chunked`, `auto`, or
  `approximate_sparse`. The former four are exact evaluation routes and do not
  discard response-relevant nonzero matrix elements.
- Sparse approximation is available only through `approximate_sparse`. Its
  relative threshold is mandatory, scale-relative, and reported together with
  the discarded commutator L2 fraction.
- `auto` is opt-in, requires an explicit memory budget and chunk size, and
  reports the method actually executed. `auto` with Doppler broadening is
  rejected until the exact routes share one characterized broadening kernel;
  memory pressure must not silently select different physics.
- Doppler broadening is decided from the actual uniform frequency-grid spacing.
  Fixed absolute skip thresholds are removed.
- A requested device function must be recognized, fully parameterized, and
  applied. Unsupported or inapplicable controls raise instead of being ignored.
- Spectroscopy uses the authoritative constants layer, including exact SI
  Boltzmann and Avogadro constants.

Consequences:

- the old `optimized` route and ignored `sparse_threshold` option are removed;
- exact chunked response evaluation uses the same transition-frequency
  orientation as the loop reference;
- method-specific controls cannot leak into unrelated routes;
- each calculation exposes a `SpectroscopyCalculationReport` describing its
  requested and executed numerical policy;
- focused tests compare realistic-dipole exact routes, validate every explicit
  mode contract, and exercise grid-derived Doppler and device broadening.

Implementation commit: `e4102d0`


### D-024: Spectroscopy polarization is a Jones bra-ket contraction

Status: Accepted on 2026-08-11.

Scope: absorption polarization, Cartesian component selection, susceptibility
conversion, and Doppler capability.

Decision:

- `axes` is a nonempty unique ordered subset of `xyz`; `pol_int` and `pol_det`
  are finite nonzero Jones kets with exactly `len(axes)` components in that
  order.
- Interaction uses `mu_int = sum_a e_int[a] mu_a`. Detection is the analyzer
  bra and uses `mu_det = sum_a conj(e_det[a]) mu_a`. `pol_det=None` means the
  same physical polarization ket, not the same un-conjugated coefficients.
- Every selected Cartesian component contributes. A third component may not be
  accepted and then ignored.
- Detection support removes only scale-relative machine-roundoff noise at
  `eps * max(abs(mu_det))`; physical SI dipoles are never compared with an
  absolute cutoff.
- The projected molecular polarizability is converted with
  `chi = number_density * response / epsilon_0`. There is no unconditional
  extra factor of `1/3`; rotational/orientational averaging belongs in the
  state and dipole model that define `response`.
- Doppler broadening is public only for `matrix` and `loop`, where every
  transition uses its own center-frequency width. `2d`, `chunked`,
  `approximate_sparse`, and `auto` raise instead of using a mean width or
  convolving absorbance after susceptibility conversion.

Consequences:

- absorption is invariant under a global Jones-vector phase;
- opposite helicities select opposite delta-M channels, an M-symmetric state
  has equal helicity spectra, and reversing M orientation reverses circular
  dichroism;
- real linear-polarization results retain their previous contraction;
- exact three-axis input is supported and malformed vectors fail before matrix
  construction;
- the obsolete aggregate-Doppler implementation is deleted.

Physics anchor: `tests/physics/test_spectroscopy_reference.py`.

Implementation commit: `834f8ef`.

### D-025: Pump-probe pathway selection uses equal-vibrational-order blocks

Status: Accepted on 2026-08-11.

Scope: pre-probe absorption density, phase-matching proxy, and PFID/radiation.

Decision:

- Every `AbsorbanceCalculator` construction requires an explicit
  `phase_matching` mode. The accepted modes are `pump_probe` and `unfiltered`;
  there is no default or automatic fallback.
- For the current pump-probe workflow, vibrational quantum number is the
  pathway proxy because it records the net optical absorption/emission order.
  `pump_probe` applies `V_i == V_j` to the density matrix immediately before
  the probe commutator.
- Equal-V blocks are retained in full. Populations and coherences among
  different rotational or M states with the same V remain; only cross-V
  density elements are discarded.
- `pump_probe` requires `basis.V_array` with one entry per Hamiltonian level.
  Missing or malformed labels raise instead of silently selecting all entries.
- `unfiltered` passes the complete pre-probe density matrix to the response.
- The calculation report records the selected mode and the discarded density
  Frobenius-norm fraction. This selection is a physical pathway choice, not a
  sparse or performance approximation.
- Radiation and PFID consume the already post-probe density matrix directly.
  They do not apply the pre-probe equal-V selection, because the radiating
  optical coherence is generally cross-V.

Consequences:

- the ambiguous `use_v_mask` flag and its `abs(delta_v) < 2` rule are removed;
- every exact numerical route receives the same already-selected density;
- pump-probe and unfiltered spectra are observably distinct when cross-V
  coherence is present;
- current pump-probe selection is documented as a V-label proxy, not as a
  universal wave-vector phase-matching engine.

Physics anchor: `tests/physics/test_spectroscopy_reference.py`.

Implementation commit: `874b1c4`.

### D-026: Phase 2 propagation contracts are explicit and endpoint-complete

Status: Accepted on 2026-08-11.

Scope: typed propagation inputs, execution policy, trajectory results, density
trace, incoherent split propagation, renormalization, and backend ownership.

Decision:

- Typed propagation always requires an explicit initial state. Configuration
  may not silently choose the first basis state.
- Algorithm, array backend, and dense/CSR storage are explicit typed choices.
  No factory infers an algorithm from polarization or matrix sparsity.
- A typed trajectory always includes the exact final state. If
  `sample_stride` does not divide the propagation-step count, the endpoint is
  appended and the final output interval is shorter; integration steps and
  field sampling are unchanged.
- Direct typed `DensityState` input requires trace one within the existing
  scale-aware tolerance. It is never automatically normalized or repaired.
- `IncoherentEnsemble` supports split-operator propagation by independently
  propagating its pure components and summing density operators with the
  already accepted normalized weights. Split propagation of a density matrix
  remains unsupported.
- Renormalization remains an explicit production policy. The caller must
  choose no renormalization or per-step renormalization; it is never silently
  enabled, and the chosen policy is recorded in the result.
- Result state arrays remain native to the selected backend. Host conversion
  occurs only through an explicit `PropagationResult.to_numpy()` boundary;
  persistence performs that conversion at the I/O boundary.

Transition rule:

Legacy low-level kernels retain their characterized array shapes and regular
stride storage until all high-level callers use the typed facade. Endpoint
completion, trace-one enforcement, and unified result construction are applied
at the new typed boundary so the hot numerical loops do not change during
Phase 2.

Consequences:

- canonical simulation uses `TimeGrid`, which never rounds or extends a span;
- optimization-specific layouts remain explicit when their characterized semantics differ;
- typed solver construction rejects unsupported combinations before matrix
  allocation;
- configuration and result metadata are reproducible without relying on
  undocumented defaults;
- existing numerical kernels remain reference implementations during the
  contract migration.

Implementation checkpoints:

- P2.1-a `TimeGrid` core and normal simulation migration: `53bfb2c`.
- P2.1-b-local endpoint migration: `965dcda` under D-027.
- Explicit backward RK4 direction: `b211610` under D-028.
- GRAPE and Krotov canonical time migration: D-029 and its contract tests.

### D-027: Local optimizer preserves its versioned legacy time layout

Status: Accepted on 2026-08-11.

Scope: `optimization/local.py`, local-optimizer time arrays, segment indices,
and the final RK4 field view.

Observed behavior:

- `_build_segments_and_tlist` gives positive `segment_size_steps` precedence,
  floors the selected segment length to an even number, evaluates
  `ceil(time_total / dt / steps)` in the existing order, and builds `tlist`
  with the existing `np.arange` expression. None of these expressions may be
  algebraically rewritten.
- A segment writes its new field to `[start + 1:end + 1]` but propagates over
  `[start:end + 1]`. Therefore the shared `start` sample belongs to the
  preceding segment. The first sample remains the pre-existing zero value.
- The midpoint is `(start + end) // 2`; lookahead ends at `end - 1`; the shape
  horizon and running-cost grid use the unsliced `tlist`, including its tail.
- Floating-point `np.arange` behavior is part of the contract. With the
  repository settings `dt=0.1` fs and `segment_size_fs=0.5` fs, `total=6000`
  fs produces 60003 samples ending at 6000.200000000001 fs and two samples
  after the last segment. `total=200000` fs produces 2000002 samples ending at
  200000.1 fs and one sample after the last segment.
- The legacy dense RK4 kernel used `(field_length - 1) // 2` steps. It consumed
  all samples for odd lengths and ignored only the final sample for even
  lengths.

Decision:

- Local optimization uses `LocalOptimizerLegacyGridV1`, not canonical
  `TimeGrid`. The versioned type wraps the arrays returned by the unchanged
  builder and exposes the existing write, propagation, and midpoint indices.
- The returned `tlist`, full electric field, shape horizon, and running cost
  retain the complete legacy storage arrays. Only the final RK4 call receives
  the odd prefix `0:2 * ((len(tlist) - 1) // 2) + 1`. This exactly reproduces
  the samples consumed by the legacy floor-based RK4 loop while satisfying the
  current explicit odd-length solver contract.
- Normal simulation and local optimization may share typed propagation,
  execution-policy, result, unit, and persistence boundaries. They must not
  share a constructor that rebuilds or repairs the local optimizer time array.
- Endpoint completion, `linspace` replacement, rounding cleanup, slice
  normalization, and automatic fallback are forbidden for this legacy policy
  without a new user-approved decision and new reference data.

Verification:

- exact-array tests cover both one- and two-tail-sample `np.arange` cases;
- spy tests capture segment and full-propagation arrays at the solver boundary;
- a dense numerical test proves bitwise equality between an even-length legacy
  kernel call and the validated odd-prefix call, including an extreme ignored
  final sample.

Implementation: `965dcda`.

### D-028: Backward RK4 direction is explicit

Status: Accepted on 2026-08-12.

Scope: Schrödinger RK4 and Krotov costate propagation.

Krotov originally constructed a decreasing `ElectricField`, reversed the
control samples, and obtained a negative propagation interval from that
container. This worked before commit `7ce9419`, when `ElectricField` accepted
decreasing arrays. Strict field validation correctly made public field grids
increasing, but left the Krotov reverse call without a migration path.

Decision:

- `ElectricField` remains strictly increasing; decreasing public time arrays
  stay invalid.
- `PropagationDirection.BACKWARD` is the only accepted backward request. A
  string or sign is rejected rather than interpreted.
- Dimensional RK4 backward propagation reverses both Cartesian field sample
  arrays and multiplies the positive propagation interval by minus one before
  entering the unchanged kernel. Its returned physical times run from the
  configured final endpoint toward the initial endpoint.
- Backward CuPy, split-operator, and nondimensional propagation raise until
  they have independent numerical references. No fallback to forward
  propagation is permitted.
- Krotov must use this direction instead of constructing a decreasing
  `ElectricField`.

The reference test compares the complete trajectory with the legacy reversed
field and negative-dt RK4 call using exact array equality.

Implementation: `b211610`; the NumPy-only capability guard is included in
the P2.1-b time-contract change.

### D-029: Optimization time spacing and output sampling are explicit

Status: Accepted on 2026-08-12.

Scope: GRAPE and Krotov time configuration, optimizer-internal trajectories,
and output-only trajectory thinning. Local optimization remains governed by
D-027.

Observed behavior:

- GRAPE and Krotov historically interpreted configured `dt_fs` as one RK4
  propagation interval, then generated `2 * int(total_fs / dt_fs) + 1` field
  samples with `np.linspace`. For the repository configurations, `dt_fs = 0.1`
  fs therefore meant a 0.05 fs field interval and a 0.1 fs propagation step.
- Krotov backward propagation worked while decreasing `ElectricField` grids
  were accepted. The later strict-increasing field validation exposed an
  incomplete migration, not evidence that the original Krotov update had never
  executed. D-028 restores that route explicitly without changing the RK4
  kernel.
- Passing `sample_stride` into optimizer propagation can remove states needed
  by objective and update indexing. It is therefore not merely an output
  control.

Decision:

- GRAPE and Krotov require `field_dt_fs`; `dt_fs` is rejected. Repository
  Krotov values migrate from 0.1 to 0.05 fs so the generated arrays and
  effective 0.1 fs propagation step remain exactly unchanged.
- `total_fs` must be an integer multiple of `2 * field_dt_fs`. No rounding,
  endpoint extension, or implicit resampling is permitted.
- Optimizer forward and costate calculations always use the complete internal
  trajectory with propagation `sample_stride = 1`.
- `output_stride` thins only the final returned trajectory. It never changes
  integration, fidelity, gradient, update, or field samples, and the exact
  endpoint is always retained. The explicit returned trajectory times are
  used for plotting so endpoint retention cannot create a time-state length
  mismatch.
- The removed `sample_stride` optimization option is rejected for GRAPE and
  Krotov rather than reinterpreted. Local optimization retains `sample_stride`
  and its original `field_dt_fs = 0.1` values because its indices and endpoint
  handling are frozen by D-027. Only the local configuration key is renamed
  from `dt_fs` to `field_dt_fs`; its numerical value and use are unchanged.

Verification:

- exact-array tests compare the migrated 200, 500, and 1000 fs field grids with
  the historical construction;
- solver spies require full internal trajectories and explicit forward and
  backward directions;
- a real TwoLevel Krotov case completes one forward-backward update iteration;
- local exact-array, segment-index, and bitwise RK4 reference tests remain
  unchanged apart from the configuration-key rename.

Implementation anchors: `tests/contracts/test_optimization_time_grid_contracts.py`,
`tests/contracts/test_optimization_solver_time_contracts.py`, and
`tests/physics/test_optimization_time_reference.py`.

### D-030: Typed state kinds validate without inference or repair

Status: Accepted on 2026-08-12 as the P2.2-a implementation of D-026.

Scope: immutable typed initial-state values before solver dispatch.

Decision:

- `PureState` owns a defensive read-only complex128 vector. It requires a
  finite, nonempty, one-dimensional input whose norm squared differs from one
  by no more than `100 * n * eps`. It never silently normalizes input.
- `IncoherentEnsemble` takes the already accepted norm-encoded raw vectors. It
  skips exact zero-norm vectors, normalizes each retained component, and
  normalizes their norm-squared statistical weights. It rejects empty,
  dimensionally inconsistent, non-finite, and overflowing inputs. Its density
  operator contains no coherent cross terms.
- `DensityState` owns a defensive read-only complex128 matrix. It applies the
  existing physical density validation and additionally requires trace one
  within the same scale-aware tolerance. It never normalizes, clips,
  symmetrizes, or projects input.
- Typed constructors never distinguish state kind from list membership or array
  shape. As of P2.2-b, `MixedStatePropagator` also rejects raw list and array
  input; direct Schrodinger and Liouville arrays remain single-kind legacy
  boundaries until their own migration.
- P2.2-a types are canonical NumPy host values. They do not advertise native
  CuPy ownership; backend transfer and capability validation belong to P2.3.

The shared factor 100 is not a new fitting parameter. It is the named existing
`NUMERICAL_VALIDATION_EPSILON_FACTOR` used for LAPACK-scale density validation,
and is now reused for unit-norm roundoff.

Implementation anchors: `core/states.py` and
`tests/contracts/test_state_kind_contracts.py`.

### D-031: Mixed-state propagation dispatches only by explicit state kind

Status: Accepted on 2026-08-12 as the P2.2-b implementation of D-026.

Scope: `MixedStatePropagator.propagate` initial-state boundary.

Decision:

- accepted inputs are exactly `IncoherentEnsemble | DensityState`;
- raw iterables and raw square arrays raise `TypeError`;
- ensemble components and normalized weights are unwrapped immediately before
  the unchanged Schrodinger propagation path;
- the validated trace-one density matrix is passed unchanged to the existing
  Liouville propagation path;
- RK4/split and dense/sparse capability rules are unchanged.

This removes semantic inference and the legacy density trace normalization. It
does not change either numerical time-development kernel.

Implementation anchors: `dynamics/mixed_state.py` and
`tests/contracts/test_mixed_state_kind_dispatch.py`.

### D-032: Single-state propagators expose typed facades over unchanged arrays

Status: Accepted on 2026-08-12 as the P2.2-c implementation of D-026.

Scope: `SchrodingerPropagator`, `LiouvillePropagator`, and optimization callers.

Decision:

- public Schrodinger propagation accepts exactly `PureState`;
- public Liouville propagation accepts exactly `DensityState`;
- both facades reject raw arrays before unit or solver validation, then unwrap
  the stored read-only array without normalization or repair;
- the former `propagate` calculation bodies are retained as private
  `_propagate_array` methods without numerical edits;
- normal simulation and fixed-M averaging construct typed initial states;
- GRAPE, Krotov, mixed-state component propagation, numerical reference tests,
  and the local optimizer use the private array bridge during migration;
- local optimization must pass the identical intermediate ndarray and retain
  its versioned odd-prefix, shared-endpoint, and index behavior.

The normal runner historically stored a column vector while RK4 immediately
applied `np.asarray(...).ravel()`. It now performs that same flattening before
`PureState` construction, making the existing shape conversion explicit without
changing component order or values. `PropagatorBase` is generic in the accepted
initial-state type so subclasses do not violate the base signature.

Implementation anchors: `dynamics/{base,schrodinger,liouville}.py`,
`tests/contracts/test_single_state_kind_dispatch.py`, and the unchanged local
optimizer reference contracts.

### D-033: Execution policy is explicit and capability-checked before allocation

Status: Accepted on 2026-08-12 as the P2.3-a implementation of D-026.

Scope: backend/storage values and propagation capability preflight.

Decision:

- `ExecutionPolicy` has two required enum fields: `ArrayBackend` (`numpy` or
  `cupy`) and `MatrixStorage` (`dense` or `csr`);
- it has no defaults and accepts no raw strings at its typed constructor;
- configuration strings cross one explicit `from_strings` parser, where unknown
  values raise instead of falling back or being inferred;
- one capability registry encodes the accepted pure, incoherent-ensemble, and
  density state combinations for RK4 and split operator;
- structural incompatibilities such as CuPy CSR and non-RK4 density propagation
  raise before optional-backend availability checks, conversion, or allocation;
- requested but unavailable CuPy raises and never substitutes NumPy;
- `dense` and `sparse` properties exist only as legacy adapter projections from
  the single storage choice.

P2.3-b removed normal-runner dual-boolean inference and passes one policy to
dipole construction and propagation. D-035 replaces the old factory automatic
algorithm selection with required typed choices and capability preflight.

Implementation anchors: `core/execution.py`,
`dynamics/capabilities.py`, and
`tests/contracts/test_execution_policy_contracts.py`.

### D-034: One execution policy controls normal simulation construction and propagation

Status: Accepted on 2026-08-12 as the P2.3-b implementation of D-026 and D-033.

Scope: normal simulation validation, model/dipole construction, fixed-M
averaging, and solver dispatch.

Decision:

- every normal simulation case requires explicit `backend`, `storage`, and
  `algorithm` fields;
- the removed `dense` and `sparse` configuration booleans raise, rather
  than being reconciled or used as fallbacks;
- validation parses one `ExecutionPolicy` and one `PropagationAlgorithm`,
  performs capability preflight, and returns those typed values to the runner;
- the identical policy instance controls dipole backend/storage and propagator
  backend/storage for both ordinary pure-state simulations and the fixed-M
  incoherent average;
- TwoLevel and VibLadder implement real SciPy CSR dipoles for NumPy, with
  exact dense element parity; CuPy CSR remains unsupported and raises before
  matrix allocation;
- low-level propagator constructor booleans remain temporary adapters derived
  only from `ExecutionPolicy`; they are not separate configuration sources.

Consequences:

- missing execution choices and unknown or structurally unsupported
  combinations fail during simulation preflight;
- model builders require an `ExecutionPolicy`, so direct callers cannot
  accidentally construct matrices with defaults that differ from propagation;
- no Hamiltonian, dipole-element, field-sampling, RK4, split-operator,
  population-summing, or optimization formula changes in this step;
- dense/CSR parity is tested for scalar-model dipoles and for the fixed-M
  population trajectory.

Implementation anchors: `simulation/validation.py`, `simulation/runner.py`,
`simulation/models/`, `dipole/twolevel/cache.py`,
`dipole/viblad/cache.py`,
`tests/contracts/test_simulation_execution_wiring.py`, and
`tests/physics/test_linear_molecule_reference.py`.

Implementation commit: pending.

### D-035: Propagator factory dispatch is explicit and typed

Status: Accepted on 2026-08-12 as the P2.3-c completion of D-026.

Scope: `dynamics.PropagatorFactory` dispatch only.

Decision:

- factory construction requires keyword-only `StatePath`,
  `PropagationAlgorithm`, `ExecutionPolicy`, and `renorm`;
- raw strings, omitted choices, and the removed `const_polarization` and
  `dipole_matrix` heuristic inputs raise;
- the shared capability registry runs before any propagator constructor;
- pure state dispatch returns `SchrodingerPropagator`, incoherent ensemble
  dispatch returns `MixedStatePropagator`, and density dispatch returns
  `LiouvillePropagator`;
- incoherent ensemble plus split operator is accepted as required by D-026;
- density propagation rejects `renorm=True` because typed density input is never
  repaired or normalized.

Consequences:

- polarization, sparsity inspection, array shape, and optional dependency
  availability never choose an algorithm;
- backend and sparse constructor flags are temporary projections of the one
  execution policy;
- numerical kernels and all optimizer propagation calls are unchanged;
- the factory remains a temporary public migration facade until P2.4 replaces
  it with typed `PropagationOptions`.

Implementation anchors: `dynamics/factory.py`,
`tests/contracts/test_solver_contracts.py`, and
`tests/physics/test_solver_invariants.py`.

Implementation commit: pending.

### D-036: Propagation options are immutable and required

Status: Accepted on 2026-08-14 as the P2.4-a implementation of D-026.

Scope: typed propagation configuration, factory construction, and normal
simulation adaptation.

Decision:

- `PropagationOptions` is an immutable slots dataclass with no defaults;
- it requires `PropagationAlgorithm`, `ExecutionPolicy`, trajectory selection,
  a positive integer sample stride, `ScalingMode`, and
  `RenormalizationPolicy`;
- scaling is exactly `dimensional` or `nondimensional`; renormalization is
  exactly `disabled` or `per_step`;
- normal simulation configuration explicitly requires the existing adapter
  keys `return_traj`, `sample_stride`, `nondimensional`, and `renorm` in
  addition to algorithm, backend, and storage;
- simulation validation constructs one options object and passes it unchanged
  to the ordinary or fixed-M path;
- `PropagatorFactory` accepts one options object instead of separate algorithm,
  execution-policy, and renormalization arguments;
- boolean and string projections exist only at calls into unchanged legacy
  solver bodies.

Consequences:

- omitted trajectory, scaling, stride, or renormalization choices fail before
  field or matrix construction;
- no option is inferred from state shape, polarization, dependency
  availability, or another option;
- numerical kernels, field sampling, endpoint handling, and optimization are
  unchanged;
- P2.4-b must replace public propagator `**kwargs` with the typed options
  boundary; P2.5 will replace conditional array/tuple results;
- `dynamics` uses lazy public exports so importing `core.states` first
  cannot trigger a solver/state circular import.

Implementation anchors: `dynamics/options.py`,
`dynamics/factory.py`, `simulation/validation.py`,
`simulation/runner.py`, `simulation/m_average.py`, and
`tests/contracts/test_propagation_options_contracts.py`.

Implementation commit: `f2cd328`.

### D-037: Public propagation calls use explicit typed fields

Status: Accepted on 2026-08-14 as the P2.4-b implementation of D-026.

Scope: public Schrödinger, Liouville, and mixed-state propagation methods and
the normal simulation adapters.

Decision:

- every public `propagate()` requires one `PropagationOptions`; no public
  propagator accepts unrestricted `**kwargs`;
- options algorithm, backend, storage, and renormalization must agree with the
  constructed solver, otherwise calculation stops before units, allocation, or
  numerical work;
- coupling is explicit: Cartesian mode requires `axes` and rejects
  `coupling_axis`; scalar mode requires `coupling_axis` and rejects `axes`;
- split-operator propagation requires an explicit `split_interaction` at the
  public call, and it must equal the constructor mode; RK4 rejects that field;
- therefore `cartesian`, the physical RK4-equivalent reference, can never be
  silently exchanged with the `helicity_projected` approximation;
- `return_times` remains a temporary explicit adapter until P2.5 introduces a
  non-conditional `PropagationResult`;
- private `_propagate_array()` retains its characterized keyword adapter while
  optimization and low-level numerical callers are migrated separately. It is
  not a public API.

Consequences:

- misspelled and removed public options are rejected by the Python signature;
- normal and fixed-M runners pass the same options object used during
  validation, and pass exactly the split interaction used at construction;
- numerical kernels, field samples, return arrays, ensemble weighting, and
  dimensionalization formulas are unchanged;
- `optimization/local.py` is untouched, including its legacy time arrays, odd
  RK4 prefix, segment indices, midpoint, write slices, and endpoint handling.

Verification:

- signature contracts cover all three public propagators;
- projection tests capture every legacy value passed to the unchanged private
  body;
- coupling ambiguity, options conflicts, and split-mode omission or mismatch
  fail before numerical work;
- the full CPU suite and all optimization time/reference contracts pass.

Implementation anchors: `dynamics/{base,schrodinger,liouville,mixed_state}.py`,
`simulation/runner.py`, `simulation/m_average.py`, and
`tests/contracts/test_public_*`.

Implementation commit: `1f4169d`.

### D-038: Models own coupling and public propagation accepts one problem

Status: Accepted on 2026-08-14 as the P2.4-c implementation of D-026.

Scope: public Schrödinger, Liouville, and mixed-state propagation boundaries;
normal simulation and fixed-M averaging adapters.

Decision:

- `CouplingSpec` is an immutable exclusive sum type: scalar coupling owns one
  typed axis; Cartesian coupling owns one ordered pair of typed axes; neither
  representation has a default;
- `SystemModel` owns name, basis, Hamiltonian, dipole, coupling, and read-only
  metadata, and validates basis/Hamiltonian/dipole dimensions without rebuilding
  any operator;
- `PropagationProblem` owns one `SystemModel`, the exact `ElectricField` and
  `TimeGrid`, and one typed initial state; field samples must exactly equal the
  canonical grid and the state dimension must equal the model dimension;
- all public `propagate()` methods accept the problem as their only physical
  input. Loose Hamiltonian, field, dipole, state, mode, axes, and coupling-axis
  arguments are removed;
- LinMol normal propagation retains its configured ordered Cartesian axes;
  TwoLevel retains scalar x coupling; VibLadder and fixed-M averaging retain
  scalar z coupling;
- fixed-M averaging builds one complete problem per block and still performs
  separate propagation followed by the same incoherent weighted population sum;
- `split_interaction`, `return_times`, and `verbose` remain temporary explicit
  call controls until later typed result and workflow contracts own them. D-039
  subsequently removes `return_times`; the other two controls remain explicit.

Consequences:

- a model and its coupling can no longer disagree at the solver call site;
- an `ElectricField` from another grid or a state from another model fails before
  unit conversion, allocation, or numerical work;
- normal and fixed-M paths pass the identical Hamiltonian, field, dipole, state
  arrays, coupling projection, options, and split mode into the frozen private
  adapters;
- private `_propagate_array()` remains the intentionally isolated compatibility
  seam for optimization and low-level migration;
- `optimization/local.py` remains byte-for-byte untouched, including the legacy
  odd prefix, boundary ownership, midpoint, indices, slices, and endpoints.

Verification:

- focused contracts cover exclusivity, dimension checks, metadata immutability,
  exact field-grid equality, state dimension, object identity, state path, and
  all three public signatures;
- projection tests capture all values passed to frozen private adapters;
- the complete suite passes 677 tests with 10 optional-GPU skips, including all
  physics, sparse RK4, split-operator, integration, and optimizer reference tests;
- Ruff, formatting, and strict mypy for the 14 named typed modules pass.

Implementation anchors: `dynamics/problem.py`,
`dynamics/{base,schrodinger,liouville,mixed_state}.py`,
`models/factory.py`, `simulation/runner.py`,
`simulation/m_average.py`, and
`tests/contracts/test_propagation_problem_contracts.py`.

Implementation commit: `873ad6e`.

### D-039: Public propagation has one backend-explicit reproducible result

Status: Accepted on 2026-08-14 as P2.5 and the final implementation of D-026.

Scope: public Schrödinger, Liouville, and mixed-state results; normal simulation, fixed-M averaging, and propagation benchmarking boundaries.

Decision:

- every public `propagate()` returns one frozen `PropagationResult`; boolean-dependent array/tuple returns and the public `return_times` control are removed;
- `times_fs`, `state`, `state_kind`, `trajectory`, `backend`, and `metadata` are unconditional fields; a final-only result contains exactly one endpoint time;
- trajectory output contains the exact configured start and endpoint. Output stride is applied only after integration, and an already computed endpoint is appended when the stride does not divide the propagation-step count;
- private array adapters and numerical kernels keep their characterized regular-stride behavior. During this transition the public boundary requests a full private trajectory with stride one, then thins output. Stride one reuses the complete state trajectory without copying it; stride greater than one temporarily requires the full internal trajectory allocation;
- state arrays remain on the requested backend. Host transfer is allowed only through explicit `PropagationResult.to_numpy()`; normal simulation and fixed-M averaging call it at their storage/population-analysis boundary;
- metadata is recursively copied into immutable JSON-compatible values. Unsupported objects and non-finite numbers raise instead of being stringified or omitted;
- metadata records independent result-schema and package versions, model declaration, coupling, every propagation/execution choice, time grid, and the nondimensional scales actually returned by the same propagation call;
- the deterministic SHA-256 configuration hash has the explicit scope `declared_model_metadata_and_propagation_contract`. It deliberately does not hash numerical Hamiltonian, dipole, or field arrays, because doing so would cause an implicit device-to-host transfer and would overstate reproducibility;
- typed `DensityState` remains immutable. The Liouville adapter makes one writable C-contiguous `complex128` working copy with identical values because the existing Numba LVNE kernel writes into its initial work array. It does not normalize, symmetrize, clip, or otherwise repair the density matrix.

Numerical and performance evidence:

- the 4001-field-point baseline was rerun for all seven NumPy dense/CSR Schrödinger and dense Liouville workloads; every final state is exactly equal to the committed Numba-CSR baseline (L2 difference `0.0`);
- after removing an accidental stride-one trajectory copy and caching package-version lookup, 16/18-dimensional and Liouville public timings are between 6% faster and 8% slower than that baseline, while two-level CSR is 6% slower;
- the two-level dense case changes from 0.177 ms to 0.276 ms (+56%, about 0.10 ms absolute). Profiling attributes the fixed cost to typed validation and immutable reproducibility metadata/result validation, not to the RK4 kernel. This deliberately tiny kernel is the documented performance exception accepted for the correctness and provenance contract;
- GPU state retention is contract-tested with a device-like object, but CUDA execution remains unverified because CuPy/CUDA is unavailable in this environment.

Consequences:

- callers never infer return shape from flags and cannot silently receive a host array from a device calculation;
- the configuration hash proves equality only for its named declared scope. Full input-data provenance requires future explicit content digests at the persistence boundary;
- Phase 5 must add an endpoint-complete low-level storage policy so large `sample_stride > 1` calculations do not allocate a full internal trajectory; that optimization must preserve the characterized integration and field-index sequence;
- optimization continues to use private migration adapters. In particular, `optimization/local.py` and its legacy odd-length grid, boundary indices, slices, and endpoint behavior are unchanged.

Verification:

- result contracts cover shape/kind/backend agreement, immutable times and nested metadata, explicit host conversion, deterministic hash scope, actual nondimensional scales, forward/backward endpoints, no-copy stride one, endpoint append, and rejection of ambiguous metadata;
- workflow, public-signature, mixed-state, density, integration, physics-reference, and all seven benchmark-path tests pass;
- the full suite passes 698 tests with 10 optional-GPU skips; branch coverage remains 67%; Ruff, formatting, strict mypy for 15 typed modules, and diff checks pass.

Implementation anchors: `dynamics/result.py`, `dynamics/{base,schrodinger,liouville,mixed_state}.py`, `simulation/runner.py`, `simulation/m_average.py`, `benchmarks/run_baseline.py`, and `tests/{contracts,physics,performance}`.

Implementation commit: this P2.5 milestone commit.

### D-040: Mechanical package migration uses target ownership without shims

Status: Accepted on 2026-08-15 for the mechanical Phase 3 migration checkpoints.

Scope: target-package file moves, direct imports, package boundaries, and
distribution contents; no physical or numerical implementation behavior.

Decision:

- tracked modules move with `git mv` to the owner already selected in
  `TARGET_ARCHITECTURE.md`;
- internal imports move directly to the target path, and superseded module
  paths are removed rather than retained through compatibility shims;
- each movement unit is limited to file ownership and import repair. Formula,
  array, unit, backend, time-grid, and kernel changes require a later dedicated
  change;
- AST dependency tests reject unrecorded imports from `core` into higher
  application layers or through the root convenience namespace. During staged
  movement, every unavoidable dependency is an exact allowlisted debt that must
  shrink when its target owner moves;
- wheel smoke tests must prove that the new module is packaged and the removed
  module is absent.

The first unit moves the unchanged generic `Hamiltonian` implementation from
`core/basis/hamiltonian.py` to `core/operators.py`. The temporary package-root
`Hamiltonian` re-export remains until O-008 resolves the final root namespace,
but `core.basis.Hamiltonian` and `core.basis.hamiltonian` are removed.

Consequences:

- this is an intentional Python import break permitted by D-001;
- the move does not authorize combining legacy basis/state classes or moving
  model formulas, because those changes require Phase 6 ownership work;
- a structural milestone must pass focused tests, the full physics/contract
  suite, Ruff, formatting, mypy, build metadata checks, and clean-wheel import.

Verification: 90 focused tests and the full 700-test CPU suite pass with 10
optional-GPU skips; branch coverage remains 67%. Ruff, formatting, strict mypy,
sdist/wheel build, Twine checks, and isolated wheel import all pass.

Implementation anchors: `core/operators.py`, `core/__init__.py`, and
`tests/contracts/test_package_architecture.py`.

Implementation commit for P3.1-a: `cf9e7a2`.

The second unit moves `core/electric_field/{core,envelopes,modulation}.py` to
`fields/{field,envelopes,modulation}.py`, preserving all class and function
bodies. Root `ElectricField` remains the identical class object, while the old
subpackage path is removed. Import and wheel tests fix the new module ownership.

Five `core` to `fields` references remained explicitly allowlisted after
P3.1-b. The allowlist rejects additions and also fails if a resolved dependency
is not removed. It is not a permitted target dependency direction.

P3.1-b verification: 154 focused tests and the full 702-test CPU suite pass
with 10 optional-GPU skips; branch coverage remains 67%. Ruff, formatting,
strict mypy, sdist/wheel build, Twine checks, wheel contents, and isolated
new-path/root-identity/old-path-absence imports all pass.

Implementation anchors: `fields/`,
`tests/contracts/test_fields_package_architecture.py`, and the exact transition
debt in `tests/contracts/test_package_architecture.py`.

Implementation commit for P3.1-b: `7b68046`.

The third unit moves `core/propagation/` mechanically to `dynamics/`, preserving
the lazy public facade, adapters, options, result contracts, and every numerical
kernel. The old package is removed without a compatibility shim. All eight Python files under `dynamics/algorithms/` are exact renames; facade
differences are import path repairs only.

This move reduces exact transitional `core` to `fields` debt from five entries
to three: two in the future `dynamics/scaling` converter and one in the unused
field-construction helper in `core/units/parameter_processor.py`. It also makes
two existing ownership debts explicit: `core/states.py` imports generic state
validation from `dynamics/algorithms/validation.py`, and `dynamics/utils.py`
imports the legacy dipole base class. Both are exact, test-enforced exceptions,
not permitted target dependency directions.

P3.1-c verification: 184 focused tests pass with 6 optional-GPU skips, and the
full 704-test CPU suite passes with 10 optional-GPU skips; branch coverage
remains 67%. Ruff, formatting, strict mypy, and distribution checks pass. The
wheel contains `dynamics/`, excludes `core/propagation/`, and passes isolated
new-path/old-path-absence imports.

Implementation anchors: `dynamics/`,
`tests/contracts/test_dynamics_package_architecture.py`, and the exact
transition debt in `tests/contracts/test_package_architecture.py`.

Implementation commit for P3.1-c: `7acab34`.

The fourth unit moves `core/nondimensional/` mechanically to
`dynamics/scaling/`. The package facade, converter, reporting helper, and
conversion utilities are exact renames. `scales.py` changes only the relative
import needed to keep using the identical `core.units.constants` object. No
scale formula, validation threshold, fallback policy, conversion order, dtype,
or result metadata behavior changes.

The move reduces exact transitional `core` to `fields` debt from three entries
to one, in the legacy `core/units/parameter_processor.py`. The converter type
annotation dependency on `dipole.base` becomes visible under the stricter
`dynamics` boundary and is exact-allowlisted alongside the existing
`dynamics/utils.py` dependency. These are migration debts, not target
dependency directions.

P3.1-d verification: 122 focused tests pass with 1 optional-GPU skip, and the
full 705-test CPU suite passes with 10 optional-GPU skips; branch coverage
remains 67%. Ruff, formatting, strict mypy, sdist/wheel build, Twine checks,
wheel contents, and isolated new-path/old-path-absence imports all pass.

Implementation anchors: `dynamics/scaling/`,
`tests/contracts/test_scaling_package_architecture.py`, and the exact
transition debt tests for `core` and `dynamics`.

Implementation commit for P3.1-d: `7618364`.

The fifth unit moves generic numerical and state validation from
`dynamics/algorithms/validation.py` to `core/validation.py`. The file is a 100%
exact rename: `NUMERICAL_VALIDATION_EPSILON_FACTOR`, all scale-aware tolerance
calculations, finite/Hermitian/positive-semidefinite checks, shape rules, odd
field-length handling, and backend checks are byte-for-byte unchanged. Only
caller imports move.

This eliminates the reverse `core.states -> dynamics` dependency. The exact
`core` transition-debt allowlist now contains only the legacy
`core.units.parameter_processor -> fields` dependency. The two exact
`dynamics -> dipole.base` debts remain unchanged.

P3.1-e verification: 128 focused tests pass with 6 optional-GPU skips, and the
full 706-test CPU suite passes with 10 optional-GPU skips; branch coverage
remains 67%. Ruff, formatting, strict mypy, sdist/wheel build, Twine checks,
wheel contents, and isolated new-path/old-path-absence imports all pass.

Implementation anchors: `core/validation.py`,
`tests/contracts/test_core_validation_architecture.py`, and the reduced exact
transition debt in `tests/contracts/test_package_architecture.py`.

Implementation commit for P3.1-e: `3bb04c9`.

The sixth unit moves model-construction modules from `simulation/models/` to a
flat top-level `models/` owner. `__init__.py`, `common.py`, and the three model
builders are exact renames. `factory.py` changes only its relative import back
to the unchanged simulation validator. Basis indices, Hamiltonian and dipole
construction, initial-state mapping, coupling selection, and execution-policy
forwarding do not change.

`linmol_m_average.py` is not a model constructor: it executes propagation and
reduces block populations. It therefore moves as a 100% exact rename to
`simulation/m_average.py`, preserving the odd/even M multiplicities, normalized
weights, reduced indices, fixed-linear-polarization tolerance, block order,
field/time handling, and incoherent population sum. The obsolete
`simulation.models` path is removed without a shim.

Three pre-existing higher-layer dependencies are exact-allowlisted at the new
`models` boundary: its facade and factory import temporary contracts from
`dynamics.problem`, and the factory imports `simulation.validation`. They must
be resolved during typed model consolidation and are not permitted target
directions.

P3.1-f verification: 183 focused tests pass with 1 optional-GPU skip, and the
full 708-test CPU suite passes with 10 optional-GPU skips; branch coverage
remains 67%. Ruff, formatting, strict mypy, sdist/wheel build, Twine checks,
wheel contents, and isolated ownership/old-path-absence imports all pass.

Implementation anchors: `models/`, `simulation/m_average.py`, and
`tests/contracts/test_models_package_architecture.py`.

Implementation commit for P3.1-f: `718a53c`.

The seventh unit moves `simulation/{checkpoint,serialization,storage}.py` to
top-level `io/`. All three implementation files are 100% exact renames. Only
the runner, validation, and test imports change, and a narrow `io/__init__.py`
facade makes the new owner explicit. The old module paths are removed without
compatibility shims.

This checkpoint is ownership-only. It preserves `checkpoint.json`,
`failed_cases.json`, `result.npz`, `summary.csv`, and `summary_success.csv`;
the checkpoint key set, timestamp and hash representation, runtime-key
exclusions, completed/failed deduplication, recursive JSON conversion,
polarization deserialization, summary status rules, and write/overwrite
behavior do not change. Adding a schema version or separating persistence from
`CheckpointManager` is a redesign and is intentionally deferred until after
the mechanical move.

`io` has no dependency on simulation, models, dynamics, fields, optimization,
spectroscopy, visualization, or CLI layers. An AST ownership test enforces that
boundary without adding transition debt. The repository therefore retains the
same six exact pre-existing transition-debt entries recorded after P3.1-f.

P3.1-g verification: 73 focused tests pass, and the full 714-test CPU suite
passes with 10 optional-GPU skips; branch coverage remains 67%. Ruff,
formatting, strict mypy, sdist/wheel build, Twine checks, wheel contents, and an
isolated install with new-owner imports and old-path absence all pass.

Implementation anchors: `io/`,
`tests/contracts/test_io_package_architecture.py`, and
`tests/contracts/test_persistence_contracts.py`.

Implementation commit for P3.1-g: `980c317`.

The eighth unit moves all five files under `plots/` to top-level
`visualization/`. Every implementation file is a 100% exact rename. Only the
root package, lazy optimization runner, and test imports change; the old `plots`
namespace is removed without a compatibility shim. The explicit
`visualization/__init__.py` exports no functions because Python assigns
same-named submodules such as `plot_all` onto the package, which would make a
function facade import-order dependent. Callers use explicit owning modules.
The root still exposes the `visualization` package but does not load Matplotlib.

This checkpoint preserves spectrogram window indexing, FFT frequencies and
magnitudes, explicit trajectory-time selection, plotted series, axis labels,
limits, filename patterns, DPI, bounding boxes, and save/show order. It also
preserves four characterized debts rather than silently changing behavior:
`plot_population.state_index` is unused and all populations are plotted; the
three standalone result-directory plotters call `show()` before `savefig()`;
`plot_electric_field` requests a legend without labeled artists; and optional
spectrum/spectrogram exceptions in `plot_all` remain print-only. These are
cleanup candidates for a separate behavior commit, not part of the move.

`visualization` imports no simulation, optimization, models, io, spectroscopy,
or CLI modules. Its only package-internal dependency is the existing core unit
converter in `plot_all`. The six exact transition debts elsewhere in the
repository are unchanged.

P3.1-h verification: 18 focused visualization/time tests and 39 broader
optimization, CLI, reference, and architecture tests pass. The full suite
passes 723 tests with 10 optional-GPU skips; branch coverage rises to 69%.
Ruff, formatting, strict mypy, sdist/wheel build, Twine checks, wheel contents,
and an isolated install with root-import laziness, new-owner imports, and old
namespace absence all pass.

Implementation anchors: `visualization/`,
`tests/contracts/test_visualization_package_architecture.py`, and
`tests/contracts/test_visualization_contracts.py`.

Implementation commit for P3.1-h: `6ebbcdb`.

The ninth unit, P3.2-a, begins the Phase 3 acceptance cleanup. A repository-wide
reference, import-graph, module-import, and distribution audit establishes that
`simulation/manager.py` contains only a comment, `simulation/timegrid.py` has no
production callers and only returns a mutable copy from the canonical
`TimeGrid.from_bounds`, and the two construction helpers on
`ParameterProcessor` have no callers. All four obsolete surfaces are removed;
time-grid tests and reference tests call the canonical owner directly.

This cleanup preserves every field-grid value, propagation interval, midpoint,
endpoint, and validation failure. It does not touch numerical kernels, physical
formulas, thresholds, backends, or model construction. Immutability is now
consistent at the typed boundary instead of being discarded by an unused
adapter. Removing `create_efield_from_params` also removes the last exact
`core -> fields` reverse dependency, reducing exact transition debts from six
to five.

The audit classifies the residual structure rather than hiding it. The only
top-level mutual dependency is `models <-> simulation`: `models.factory`
imports simulation-owned input validation and `simulation.runner` imports
model construction. Moving those unchanged model predicates to
`models.validation` is the next mechanical cleanup. The model factory, generic
dipole factory, and propagator factory are not duplicates because they select
different objects at different layers. The two `models -> dynamics.problem`
and two `dynamics -> dipole.base` debts remain deferred to the separately
characterized Phase 6 model consolidation.

P3.2-a verification: 97 focused tests pass, and the full suite passes 726 tests
with 10 optional-GPU skips; branch coverage remains 69%. Ruff, formatting,
strict mypy for 14 modules, sdist/wheel build, Twine checks, all 96 discovered
module imports, dependency audit, wheel contents, and isolated old-path/API
absence checks pass.

Implementation anchors: `core/time.py`,
`core/units/parameter_processor.py`,
`tests/contracts/test_phase3_acceptance_architecture.py`, and
`tests/contracts/test_time_grid_contracts.py`.

Implementation commit for P3.2-a: `3f05abd`.

The tenth unit, P3.2-b, extracts the model-owned subset from
`simulation/validation.py` to `models/validation.py`: the model key map, model
selection normalization, required-key check, and potential-name check. Their
predicate order, default, accepted values, sorted missing-key message, and
construction order remain unchanged. Direct model construction receives the new
`ModelConfigurationError`; `validate_simulation_case` translates that error to
the existing `SimulationConfigurationError` with the identical message.

Time-grid, pulse, polarization, execution, capability, split-interaction, and
M-average validation remain in `simulation.validation`. Moving the entire file
would incorrectly make workflow policy model-owned. No Hamiltonian, dipole,
field, state, propagation, optimizer, index, threshold, or unit-conversion logic
changes.

This removes `models.factory -> simulation.validation` and therefore the final
top-level mutual dependency. The four remaining exact migration debts are two
`models -> dynamics.problem` and two `dynamics -> dipole.base` imports. They
require Phase 6 model/operator consolidation and are not hidden by a structural
allowance. The Phase 3 acceptance audit now finds no package-level cycle, no
internal root convenience imports, no unclassified duplicate factory, and all
target packages in the wheel.

P3.2-b verification: 74 focused tests and the full 732-test CPU suite pass with
10 optional-GPU skips; branch coverage remains 69%. Ruff, formatting, strict
mypy for 15 modules, sdist/wheel build, Twine checks, all 97 discovered module
imports, dependency audit, wheel contents, and isolated model/simulation error
boundary checks pass.

Implementation anchors: `models/validation.py`,
`simulation/validation.py`, `models/factory.py`, and
`tests/contracts/test_model_validation_ownership.py`.

Implementation commit: this P3.2-b milestone commit.

### D-041: Simulation inputs, field kinds, and model parameters are explicit

Status: Accepted on 2026-08-24.

Scope: normal simulation configuration, model parameter schemas, field
construction/injection, LinMol representation, and split-interaction options.

The user accepted all recommendations in O-010, with the additional
requirement that Python callers can inject an externally constructed electric
field. Structural time-grid validation is mandatory for injected fields; the
library must not silently reinterpret or repair their sampling.

Decision:

- `basis_type` and `initial_states` are required. The runner never infers
  LinMol or the ground state. Multiple indices retain the coherent semantics of
  D-007.
- LinMol requires an explicit representation, named `m_resolved` or
  `m_incoherent_average` in the target schema. The old `use_M` boolean is not a
  target public option. Cartesian axes apply only to `m_resolved` and are
  rejected for the M-averaged and scalar-model routes.
- TwoLevel, VibLadder, and LinMol M averaging consume an explicit scalar field.
  M-resolved LinMol consumes an explicit Cartesian field. Scalar physics does
  not require or retain a dummy Jones polarization.
- `split_interaction` is required only where split-operator propagation has a
  genuine Cartesian/helicity choice: M-resolved LinMol. RK4 rejects it.
  Scalar models and fixed-linear M averaging use their existing scalar Strang
  interaction internally and do not expose an inapplicable mode selector.
- Generated pulses require an explicit envelope kind and all parameters that
  define that envelope, including its center when applicable. Additive absence
  values such as zero phase, zero GDD, and zero TOD remain explicit safe
  defaults. Modulation kind is explicit; selecting sinusoidal modulation
  requires all of its defining parameters.
- Unknown, removed, and model/field/algorithm-inapplicable keys raise with the
  offending key and reason. The mixed-case `Sinusoidal_modulation` spelling is
  not part of the versioned target schema.
- YAML remains a declarative CLI input, but it is not the only construction
  route. The Python API accepts an already sampled external field together with
  the canonical `TimeGrid`. Generated and injected fields converge to the same
  typed simulation-case boundary.
- An injected field must match its `TimeGrid` exactly. The time samples are
  finite, strictly increasing, uniform, odd in count, contain both endpoints,
  and have length `2 * propagation_steps + 1`. Scalar samples are one
  dimensional; Cartesian component arrays have identical one-dimensional
  shape; all field samples are finite. The propagation interval remains
  exactly twice the field-sampling interval.
- Injection never trims, pads, rounds, interpolates, resamples, normalizes, or
  otherwise repairs samples. Any future resampling utility is a separate,
  explicitly invoked preprocessing operation and never part of simulation
  validation.
- Structural validity of a time step is distinct from numerical adequacy. The
  former is validated automatically. Accuracy depends on the complete
  generator and observable, so it is assessed only by an explicit convergence
  operation with a caller-selected tolerance (for example comparing `dt` and
  `dt/2`). Such an assessment reports results and never changes the requested
  calculation.
- Phase 6 introduces one immutable, frozen parameter schema per model. It owns
  required fields, numeric types, finiteness, ranges, units, Morse constraints,
  and derived instance-local values before basis or matrix allocation. Valid
  parameter sets must produce the same basis ordering, Hamiltonian, dipole,
  initial state, and propagation inputs as the characterized implementation.

Consequences:

- This intentionally breaks configurations that depended on defaults or
  supplied ignored/inapplicable keys; backward compatibility is not preserved
  under D-001.
- The migration is split into test-protected commits: required selection,
  representation, field types/injection, generated-field schema, model
  schemas, strict unknown-key rejection, and explicit convergence reporting.
- No numerical kernel, local-optimizer time array, segment index, endpoint,
  Hamiltonian sign, physical threshold, or model formula changes as part of
  these input-boundary migrations.
- `LocalOptimizerLegacyGridV1` remains governed exclusively by D-027 and is not
  reconstructed through the new normal-simulation field boundary.

Implementation: all seven bounded units are complete in this checkpoint.
Normal simulation requires `basis_type`, `initial_states`, and LinMol
`representation`; `m_resolved` additionally requires `axes`. The removed `use_M`
key raises migration guidance. `ScalarField` and `CartesianField` own defensive,
read-only, real V/m samples and the exact canonical `TimeGrid`.
`simulation.runner.run_simulation_case` accepts one externally sampled field and
rejects every generated-field parameter in the same call. Generated pulses still use
the characterized `ElectricField` construction, then freeze its exact sampled values
without numerical transformation. TwoLevel, VibLadder, and M averaging receive
`ScalarField`; M-resolved LinMol receives `CartesianField`. Optional explicit
scalar/Jones decomposition metadata preserves the existing generated
`helicity_projected` path; a general Cartesian field does not invent that
decomposition. Generated fields now also require `carrier_frequency` and
`carrier_frequency_units` under D-042. The immutable `Frequency` boundary
normalizes once to `rad/fs`; explicit ordinary-frequency and wavenumber inputs
omit `2π`, and the FFT modulation center is explicitly cycles/fs. The removed
unit-ambiguous `carrier_freq` key raises.

Generated fields now require `envelope_kind`, `t_center`, and
`modulation_kind`; the four existing three-argument envelope functions preserve
their exact sampled arrays. Sinusoidal selection requires amplitude, frequency,
and `phase`/`amplitude` type, while additive phases and dispersion retain zero
defaults. Custom and Voigt waveforms use explicit sampled-field injection.
The generated and externally injected routes now converge to the immutable
`simulation.case.SimulationCase` after field sampling and before matrix
allocation. It owns the frozen model schema, immutable initial-state indices,
sampled field and `TimeGrid`, representation, axes, and propagation controls.
Model and fixed-M builders consume the frozen schema without re-reading raw input.

The final mapping rejects unknown names and parameters that belong to another
model, field route, or algorithm. TwoLevel and VibLadder reject a dummy
`polarization`; only generated M-resolved LinMol requires it. Public
`split_interaction` is required only for M-resolved LinMol split propagation and
is rejected by RK4, scalar models, and M averaging. Optional runner controls
`save`, `validate_units`, and `verbose` require actual booleans. Python parameter
loading excludes imported module objects but deliberately retains scalar helper
values so a misspelled or undeclared input cannot disappear before validation.

`simulation.convergence.assess_simulation_convergence` is the separate,
opt-in accuracy service. It requires caller-supplied coarse and fine cases, a
nonempty observable name, a callable observable, and a finite nonnegative
tolerance. Generated cases may differ only in `dt`; injected cases may differ
only in their same-kind `ScalarField` or `CartesianField` and its `TimeGrid`.
Endpoints must match exactly and the fine field-grid step must be strictly
smaller. The report stores defensive read-only observable values and evaluates
`max(abs(coarse - fine)) <= tolerance`. It forces no output writes, never
resamples or changes a field, never chooses a new step, and never reruns a case.

Verification: the full CPU suite passes 872 tests with 10 optional-GPU skips;
branch coverage is 70%. Ruff, formatting, strict mypy for 21 named modules,
sdist/wheel build, and Twine checks pass. The local optimizer source, time grid,
endpoint ownership, and indices are unchanged.

### D-042: Frequency inputs use neutral names and explicit units

Status: Accepted on 2026-08-27.

Scope: public frequency-bearing configuration and typed unit boundaries.

User-facing frequency values use a quantity name that does not encode a unit and
a required paired `*_units` field. Ordinary-frequency and wavenumber values do
not contain `2π`; angular-frequency values do. Supported ordinary-frequency
units are Hz through PHz, supported wavenumber spellings are `cm^-1`, `cm-1`,
and `wavenumber`, and supported angular-frequency units are `rad/s`, `rad/ps`,
and `rad/fs`.

`core.units.Frequency` is an immutable finite-scalar boundary. It converts once
to canonical angular frequency in `rad/fs`; numerical consumers do not inspect
unit strings. An explicit `cycles_per_fs` view exists only for boundaries whose
mathematics is ordinary frequency, including FFT bins. No parameter processor
may pre-convert the new neutral value and then leave a stale input-unit label.

The generated-field schema therefore requires `carrier_frequency` and
`carrier_frequency_units`. The removed ambiguous `carrier_freq` configuration
key raises before allocation. Pulse phase construction receives canonical
`rad/fs` with that unit stated explicitly. Sinusoidal spectral modulation
receives the equivalent center in cycles/fs because `rfftfreq` uses ordinary
frequency. Field samples and propagation formulas are otherwise unchanged.

The first bounded implementation unit covers generated-field carrier frequency.
The second unit adds frozen `LinMolParameters`, `VibLadderParameters`, and
`TwoLevelParameters`, validated before matrix allocation. LinMol and VibLadder
now require neutral `vibrational_frequency` and `anharmonic_shift` value/unit
pairs; LinMol also requires `rotational_constant` and
`vibration_rotation_coupling` pairs. Old unit-encoded runner keys and their
`_units` variants raise migration errors. Builders and fixed-M averaging receive
the same canonical `rad/fs` numbers as before, while low-level basis formulas
and propagation kernels are unchanged. The general `ParameterProcessor` does
not pre-convert these typed fields or `energy_gap`, preventing double
conversion against a stale unit label. This corrects the old TwoLevel runner
path for noncanonical `energy_gap_units`; a processor-to-Hamiltonian regression
test fixes the single-conversion behavior.

`carrier_freq_sin_mod` is deliberately excluded: its current formula multiplies
an FFT-frequency difference, so its dimensional meaning must be confirmed
before renaming or conversion.

Verification anchors: `tests/unit/test_unit_conversions.py`,
`tests/contracts/test_model_validation_ownership.py`,
`tests/test_simulation_models.py`, and
`tests/physics/test_linear_molecule_reference.py`. The complete CPU suite passes
834 tests with 10 optional-GPU skips; branch coverage remains 70%. Ruff,
formatting, strict mypy for 17 configured modules, all 100 discovered-module
imports, sdist/wheel build, and Twine checks pass. The local optimizer source,
time grid, endpoint ownership, and indices are unchanged.

### D-043: Spectral modifiers and unit validation use physical explicit semantics

Status: Accepted on 2026-08-29.

Scope: spectral modulation, dispersion, intensity conversion, and strict unit
validation.

Sinusoidal spectral modulation uses ordinary FFT frequency `f` and center
`f0` in cycles/fs. Its time-like input is a physical delay `tau`, supplied
as a finite scalar with an explicit time unit and converted once to fs. The
argument is

~~~text
theta(f) = 2*pi*tau*(f - f0) + phi0
~~~

Phase modulation multiplies the spectrum by
`exp(-i*A*sin(theta))`. Amplitude modulation multiplies it by
`1 + m*sin(theta)` and requires `0 <= m <= 1`. Zero depth is an exact
identity and bypasses the FFT. Clipping, absolute-value repair, and the legacy
offset `A*sin(theta) + A` are forbidden. The old
`amplitude_sin_mod`, `carrier_freq_sin_mod`, `phase_rad_sin_mod`, and
`type_mod_sin_mod` configuration keys raise a migration error.

GDD and TOD are physical spectral-phase derivatives. With
`delta_omega = 2*pi*(f - f0)`, the applied phase is

~~~text
Phi(delta_omega) =
    (GDD/2) * delta_omega**2 + (TOD/6) * delta_omega**3
~~~

and the existing Fourier sign convention applies `exp(-i*Phi)`.

Intensity inputs are cycle-averaged intensities. Conversion returns peak
electric-field amplitude through
`E_peak = sqrt(2*I*mu_0*c)`; aliases share this same convention.

Propagation-unit validation is structural and strict. It requires formal
canonical accessors for Hamiltonian J, dipole C*m, time fs, and field V/m and
checks compatible shapes and a finite positive field-grid step. It contains no
typical-value ranges, invented 1000 fs scale, one-fifth time-step recommendation,
interaction threshold, raw-attribute fallback, warning-only exception
downgrade, clipping, or repair. General parameter conversion has no
`strict=False` or heuristic `validate=True` path; a failed known conversion
raises.

Verification anchors:
`tests/test_electric_field.py`,
`tests/unit/test_unit_conversions.py`,
`tests/contracts/test_simulation_contracts.py`, and
`tests/contracts/test_unit_boundary_characterization.py`.
The complete CPU suite passes 924 tests with 10 optional-GPU skips and measured
branch coverage is 72%.

Implementation commit: `bae1e69`.

### D-044: Supported examples are executable; historical examples are archival

Status: Accepted on 2026-08-29.

Scope: `examples/`, `benchmarks/`, `scripts/`, launcher behavior, and CI.

The supported example set is the three top-level typed examples listed in
`examples/README.md`. They use the current public simulation boundary, finish
in seconds, write no result files, and run in CI through
`scripts/smoke_examples.py`. The launcher scans only top-level
`example_*.py` files and never descends into helpers or archives.

Former v0.2 examples, dedicated optimization helpers, parameter modules, and
notebooks are historical migration material under
`examples/archives/v0_2_scripts/`. Legacy local-optimizer scripts with
undefined experiment-specific constants and the external C++ RK4 example were
archived without guessing values or changing their calculation logic.
`examples/archives/` is excluded from Ruff and smoke execution.

Active examples, benchmarks, and scripts must pass Ruff lint/format and Python
compilation. CI enforces those gates and executes all supported examples.
Moving an archived example back into the supported set requires migration to
the current public API, a bounded quick execution, and inclusion in the smoke
runner.

Implementation commits: `e58a009` (active utility formatting), `22c0313`
(content-preserving archive moves), and `ea07387` (facade, examples, and CI).

### D-045: Public scalar quantities require units; internal values are canonical

Status: Accepted on 2026-08-30; implemented on 2026-08-31.

Scope: normal-simulation configuration, model dipole scale, generated-field
time/amplitude/dispersion inputs, parameter loading, and saved parameters.

Every user-supplied scalar physical quantity at the normal-simulation boundary
has an explicit unit. Each time value has its own required pair:
`t_start/t_start_units`, `t_end/t_end_units`, `dt/dt_units`,
`duration/duration_units`, and `t_center/t_center_units`. Generated amplitude
requires `amplitude/amplitude_units`. The public model dipole input is the
neutral `dipole_scale/dipole_scale_units`; the unit-encoded `mu0_Cm` key is
removed from the normal-simulation schema. Existing frequency pairs and
`modulation_delay/modulation_delay_units` remain required.

If GDD or TOD is supplied, its matching `gdd_units` or `tod_units` is
required; a unit without its value is also an error. Omission of both members
retains the accepted exact zero modifier. Radian-only phase inputs remain
explicitly unit-bearing in their names (`phase_rad` and
`modulation_phase_rad`).

Caller-owned mappings are never rewritten. Python files, direct mappings,
CLI, batch, and checkpoint routes preserve the submitted values and unit
labels for saving and provenance. Frozen typed quantities convert each value
exactly once to the internal canonical system: fs, V/m peak field, C*m,
fs^2, fs^3, and rad/fs. Numerical consumers receive only canonical values and
never inspect source-unit strings.

The generic `ParameterProcessor`, its singleton, print-based conversion, and
route-dependent preprocessing are removed now that all normal-simulation
consumers use the typed boundary. Direct and batch execution of the same
value/unit mapping must produce identical sampled fields, Hamiltonians, and
populations. Numerical kernels, optimizer grids and indices, and external
sampled-field injection are unchanged.

The implementation is divided into four bounded units:

1. frozen dipole, peak-field, GDD, and TOD values wrap unchanged converter
   formulas;
2. frozen model schemas require and canonicalize
   `dipole_scale/dipole_scale_units`;
3. `GeneratedFieldParameters` requires a unit for each public time and
   amplitude value, conditionally requires GDD/TOD pairs, and sends canonical
   values to the unchanged `ElectricField` waveform operations;
4. parameter files and mappings retain submitted values and labels, saved
   `parameters.json` retains those pairs, and the obsolete
   `ParameterProcessor` is deleted.

Regression tests cover missing pairs, cross-unit sampled-field and population
equivalence, non-mutation, saved provenance, and batch-route preservation.
Implementation commits: `b56eea2`, `3590667`, and `60629f1`.

### D-046: Spectroscopy inputs require exact unit labels

Status: Accepted and implemented on 2026-08-31.

Scope: `spectroscopy.ExperimentalConditions`, absorbance/radiation/PFID
spectral grids, device resolution, the spectroscopy factory, and public
spectroscopy examples.

The existing spectroscopy formulas already assume K, Pa, m, ps, kg per
molecule, and cm^-1. P4.3-f makes those assumptions executable public
contracts without introducing alternative conversions. Each experimental
condition is a required value/unit pair. `T2` is renamed to the neutral
`coherence_time` input, and the frozen boundary exposes canonical
`temperature_k`, `pressure_pa`, `optical_length_m`, `coherence_time_ps`, and
`molecular_mass_kg` fields to numerical consumers.

All public wavenumber arrays require `wavenumber_units="cm^-1"`. A requested
device function requires both `device_resolution` and
`device_resolution_units="cm^-1"`; either member outside that mode is an
error. Direct 2D preparation, radiation, PFID, and device-function entry points
have the same strict boundary. Missing or unsupported units raise and never
fall back to formula knowledge or a default.

The number-density, coherence-decay, Beer-Lambert, Doppler, response,
polarization, phase-matching, broadening, exact/approximate, and auto-routing
formulas are unchanged. Internal 2D preparation receives an already canonical
cm^-1 array and does not inspect unit strings. Regression tests retain all
exact-route comparisons and add missing/wrong-unit, device-pair, factory,
radiation, and PFID coverage.

Implementation commit: `cf4e70a`.

### D-047: Arbitrary low-level field arrays require amplitude units

Status: Accepted and implemented on 2026-08-31 as P4.3-g unit 1.

Scope: `ElectricField.add_arbitrary_Efield`, `ZeroField`, direct field-unit
classification, and GRAPE/Krotov/local-optimizer field-array construction.

Every arbitrary field array now requires `field_units` and is converted once
to V/m before the existing shape check and addition. The accepted labels are
the converter's direct electric-field amplitude units. Intensity labels are
not accepted: intensity has no sign or carrier phase and therefore cannot
define an arbitrary field waveform without adding an unapproved reconstruction
rule. Missing, unsupported, or intensity units raise explicitly.

Existing optimizer-produced arrays are already V/m and now state that label at
the call. Their values and shapes are unchanged. In particular, the local
optimizer's odd legacy grid, shared endpoints, segment/full slices, indices,
and RK4 consumption are byte-for-byte untouched. The Krotov V=0 to V=3
integration reference and all local optimizer time/index characterizations
remain green.

P4.3-g remains incomplete: constructor units and generated low-level pulse
amplitude/GDD/TOD pairs are separate units. Implementation commit: `dbeadae`.

### D-048: Low-level field construction has one canonical storage unit system

Status: Accepted and implemented on 2026-08-31 as P4.3-g unit 2.

Scope: `ElectricField` and `ZeroField` direct construction,
`ElectricField.from_time_grid`, field-scale reporting, and the two uncalled
construction helpers.

Direct construction requires the caller to label `tlist` with `time_units`.
The array is converted once to fs and validated there. Construction accepts no
`field_units`: there is no field-valued constructor input to label, and the
zero-initialized field plus every subsequently stored sample use V/m.
`from_time_grid` is the canonical path and therefore consumes its fs values
without exposing a redundant unit selector.

Field-scale reporting requires the desired direct amplitude unit and reports
both the fixed V/m scale and that requested representation. Intensity labels
raise. The uncalled `create_from_SI` and `create_with_units` alternatives are
removed so there is one constructor contract. The legacy `get_time_SI` name is
retained for now but is documented accurately as returning fs; explicit output
conversion uses `get_time_in_units`.

All active source, tests, and benchmark callers now state their time unit.
Cross-unit tests prove that equivalent seconds and fs inputs produce the same
canonical time and field arrays. Optimization-created grids are explicitly fs;
the local optimizer's odd length, endpoints, slices, indices, values, and RK4
calls are unchanged. The raw-array nondimensionalization boundary continues to
require its own time label and receives the field's canonical fs values.

P4.3-g remains incomplete only for generated low-level pulse amplitude and
optional dispersion unit pairing. Implementation commit: `8d3e1e6`.

### D-049: Low-level generated pulses require explicit physical units

Status: Accepted and implemented on 2026-08-31 as P4.3-g unit 3.

Scope: `ElectricField.add_dispersed_Efield`, its active source, test, and
benchmark callers, and the existing Krotov initial-pulse adapter.

Every low-level generated pulse now requires `duration_units`,
`t_center_units`, `carrier_freq_units`, and `amplitude_units`; `amplitude`
itself is also required. Each quantity is converted once to the existing
canonical fs, fs, cycles/fs, and V/m representation before waveform
evaluation. Only direct electric-field amplitude labels are accepted.
Intensity labels raise because they do not preserve a caller-defined signed
amplitude without an additional physical reconstruction rule.

GDD and TOD remain independent optional effects. Each accepts either a complete
value/unit pair or no pair. Complete pairs convert once to fs^2 or fs^3;
omission gives the same exact zero used previously. Partial pairs raise before
polarization state or field samples are mutated.

The envelope functions, carrier phase, `2*pi` handling, dispersion Taylor
coefficients, FFT implementation, polarization inference, scalar split field,
and addition order are unchanged. A frozen nonzero-GDD/TOD sample reference is
unchanged at double precision. Equivalent ps/THz/MV/cm/ps^2/ps^3 and canonical
inputs agree within conversion roundoff. The Krotov adapter merely labels its
existing amplitude V/m; its defaults, field values, update indices, and V=0 to
V=3 integration reference are unchanged.

This completes P4.3-g. Optimization-specific initial-pulse defaults and unit
encoded parameter names remain the separately characterized P4.3-h work.
Implementation commit: this checkpoint.

### D-050: Krotov initial fields require an explicit source and physical units

Status: Accepted and implemented on 2026-09-01 as P4.3-h.

Scope: Krotov initial-field parameters, generated Gaussian seed construction,
sampled seed injection, and the active V=0 to V=3 reference configuration.

`initial_field_kind` is required and is either `generated` or `sampled`.
There is no generated-field construction followed by an implicit
`efield_initial` override. Generated seeds require neutral value/unit pairs for
duration, center, carrier frequency, and direct electric-field amplitude, plus
a finite nonzero two-component polarization. GDD and TOD retain the D-049
complete-pair-or-exact-zero contract. The generated kind means the existing
Gaussian-FWHM Krotov seed with zero carrier phase; the existing waveform
formula and polarization normalization are unchanged.

Sampled seeds require `initial_field_samples` and `initial_field_units`. Only
real finite `(n_field_points, 2)` arrays and direct electric-field amplitude
units are accepted. Values are copied and converted once to V/m. Their length
must exactly match the canonical odd `TimeGrid`; no interpolation, trimming,
padding, phase reconstruction, intensity conversion, or normalization occurs.
Generated-only and sampled-only keys are mutually exclusive. Unknown
`initial_*` names and every legacy Krotov initial-pulse name raise with an
explicit migration error.

The Krotov update equation, objective, backward propagation, half-step field
grid, `i * 2` update index, endpoint handling, and numerical optimizer defaults
are unchanged. The frozen generated samples and V=0 to V=3 initial/optimized
fidelities remain unchanged. Class-D penalty and tolerance quantities remain
unresolved and are not renamed or converted by this decision.

Implementation commit: this checkpoint.


### D-051: Local field limit and seed names state V/m without numerical change

Status: Accepted and implemented on 2026-09-01 as P4.3-i.

Scope: the local optimizer's component field limit, zero-drive seed amplitude,
active local YAML configurations, and exact legacy local propagation contracts.

`field_max` is replaced by `field_max_v_per_m`, and `seed_amplitude` is replaced
by `seed_amplitude_v_per_m`. These are unit-encoded private optimization inputs
whose only supported representation is direct electric-field amplitude in V/m.
The former ambiguous keys raise with the replacement name instead of being
ignored or interpreted. The existing defaults remain exactly `1e12` and `1e3`.

The rename is a projection-only change. Seed construction still precedes the
same independent `np.clip` operation on each of the two control components.
Characterization fixes an explicit `40 V/m` seed and `25 V/m` component limit,
including both stored field components, segment propagation input, full RK4
input, and untouched tail endpoints. The odd `np.arange` layout, shared
boundary ownership, segment construction and midpoint, slices, lookahead,
field values, and RK4-consumed prefix remain unchanged.

No meaning or unit is inferred here for `gain`, `c_abs_min`,
`drive_abs_min`, or `shape_floor`; those quantities remain blocked at this
checkpoint. D-058 later resolves `gain` only.

Implementation commit: this checkpoint.


### D-052: SymTop and molecular symmetry start from a narrow explicit contract

Status: Accepted; symmetry foundation implemented on 2026-09-01 as P4.3-j.

Scope: future SymTop production model, molecular-symmetry descriptors,
nuclear-spin state selection, and name-based symmetry presets.

The first supported SymTop physics is a rigid symmetric top in a
nondegenerate, totally symmetric parallel vibrational band. Its basis is
`|v,J,K,M>` with signed K and M. The body-fixed transition dipole is along the
symmetry axis, so the initial selection rules are Delta K=0 and lab-frame
Delta M=0,+/-1. Hamiltonian frequencies use the existing `omega01` plus
positive anharmonic-shift convention. Neutral perpendicular/parallel
rotational constants and separate perpendicular/parallel vibration-rotation
couplings will be required. NumPy dense and CSR RK4 are the first production
execution routes; unsupported routes must raise. Split-operator support is a
later reference-tested unit. No hyperfine or nuclear-spin conversion dynamics
is implied.

`models.symmetry` is a model-layer foundation, not a character-table engine.
It separates geometric point group, an optional permutation-inversion group,
rotational state classification, and nuclear-spin policy. Point-group names
never infer nuclear-spin weights. Every preset carries a canonical preset ID,
explicit aliases, rule source, and rule version. Resolving a
molecule name supplies symmetry rules only: it never supplies a rotational
constant, vibrational frequency, dipole, temperature, field, or unit.

Initial presets are `H2`, `D2`, `T2`, `HD`, and `CH3F`. In the supported
totally symmetric vibronic manifold, H2/T2 use even-J/odd-J weights 1/3,
D2 uses 6/3, and HD uses a state-independent weight 6. CH3F classifies
K=3n as ortho and other K as para, including signed K through `abs(K)`.
CH3F statistical weights are deliberately unavailable until the signed-K
basis is symmetry adapted; requesting them raises. Unknown names, unsupported
vibronic symmetry, invalid quantum numbers, and unsupported isomer names also
raise without fallback.

Statistical weights belong to thermal or otherwise explicit incoherent
population construction. They are not coherent state-vector amplitudes.
Filtering by a selected spin isomer is separate from obtaining a statistical
weight and never rewrites quantum numbers.

This checkpoint does not connect presets to the experimental legacy SymTop
basis or change any Hamiltonian, dipole, propagation, or optimization result.
The next unit must add independent primitive references before replacing the
legacy SymTop implementation.

Implementation commit: this checkpoint.


### D-053: Production SymTop is an independent rigid parallel-band model

Status: Accepted and implemented on 2026-09-02 as P4.3-k.

Scope: normal-simulation SymTop Hamiltonian, dipole, symmetry filtering, and
explicit unsupported-route boundaries.

D-053 implements D-052 without reusing the known-broken legacy formulas. The
signed `|v,J,K,M>` basis is ordered by `v,J,K,M`. Its energy uses the accepted
`omega01` convention, separate perpendicular/parallel rotational constants,
and separate vibration-rotation couplings. The parallel dipole uses independent
rank-one Wigner-3j factors with Delta K=0 and Cartesian Delta M=0,+/-1. The
Morse factor and per-instance bound-level derivation retain the accepted model.

The constructor requires explicit quantities with units, a named symmetric-top
preset, Cartesian axes, and exactly one pure `ortho` or `para` sector. The
CH3F preset filters signed K sectors but supplies neither constants nor a
statistical weight. `all` raises rather than coherently combining spin isomers.

Supported execution is NumPy dense or CSR RK4, including the strict
nondimensional path. CuPy and split operator raise before propagation. The
optimization runner also rejects SymTop before constructing the legacy basis;
it will become supported only after it consumes the shared frozen model and its
result parity is characterized without changing optimizer grids or indices.

Independent SymPy references cover every low-J Wigner element through J=3;
dense and CSR dipoles are exactly equal; Hamiltonian, unit-conversion,
Morse-overtone, dense/CSR population, and dimensional/scaled parity tests pass.
No optimizer numerical kernel or legacy direct SymTop class is changed.

Implementation commit: this checkpoint.


### D-054: Optimization reuses frozen production model construction

Status: Accepted and implemented on 2026-09-02 as P4.3-l.

Scope: optimization model input, basis/Hamiltonian/dipole construction, and
optimization quantum-state selection. Optimizer algorithms and time grids are
excluded.

The optimization runner now consumes the same frozen LinMol, VibLadder, and
TwoLevel physical parameter schemas and the same model-owned operator builders
as normal simulation. Optimizer state semantics remain separate: `initial` and
`target` are exact quantum-number tuples, not normal-runner basis indices, and
the boundary never adds or removes an M quantum number. LinMol optimization
therefore requires `representation=m_resolved`; the separately propagated
M-incoherent-average workflow is not silently approximated by a no-M basis.

The optimization schema uses neutral physical names with a required unit next
to every scalar quantity. Legacy `*_cm`, shared `input_units`/`output_units`,
`mu0`/`unit_dipole`, `use_M`, unknown keys, and model-inapplicable keys raise
with a precise migration error. `system.type=vibladder` replaces `viblad`.
Construction remains NumPy CSR, and the returned optimization Hamiltonian
retains the historical rad/fs representation. Three-model characterization
preserves basis ordering and Hamiltonian values; SI dipoles agree within
unit-conversion roundoff (maximum observed relative difference about
`2.1e-16`). The stored four-level Krotov initial and ten-iteration fidelities
are unchanged.

No local-optimizer grid, segment, endpoint, midpoint, field sample, RK4 slice,
GRAPE/Krotov trajectory, update index, objective, gradient, penalty, or
tolerance changes in this decision. SymTop remains an explicit optimization
error because sharing construction does not establish a validated SymTop
objective or control contract.

Most tracked `config_temp_*` and historical LinMol optimizer YAML files already
lacked required dipole values and were not executable before this decision.
They are not assigned invented physical constants; only the complete stored
Krotov reference config is migrated. Their removal or completion requires a
separate explicit decision.

Implementation commit: this checkpoint.


### D-055: Incomplete v0.2 optimization documents are historical archives

Status: Accepted and implemented on 2026-09-03 as P4.3-m.

Scope: historical optimization YAML disposition and missing dipole values.

Thirteen incomplete v0.2 optimizer YAML files are moved, with history, to
`examples/archives/v0_2_optimization_configs/`. They remain historical records
and receive no invented dipole value. They are excluded from the supported
configuration set and must be explicitly migrated and tested before reuse.

Implementation commit: this checkpoint.


### D-056: Optimization documents are closed and explicit

Status: Accepted and implemented on 2026-09-03 as P4.3-n.

Scope: current optimization YAML ownership, algorithm-option names, control
axes, output and plotting policy, and supported configuration examples.

The configured optimization boundary requires exactly `system`, `states`,
`time`, `algorithm`, `algorithms`, `plot`, and `output`. Each section has a
closed key set. The selected algorithm must have a supplied parameter mapping;
unknown algorithm names, unknown options, removed names, and malformed nested
spectral-constraint keys raise before model construction. Krotov also validates
its generated/sampled initial-field discriminator, value/unit pairs, and exact
sampled-grid length at this boundary.

`control_axes` is required for local, Krotov, and GRAPE. It is an ordered
two-character lowercase selection from `x`, `y`, and `z`; no case conversion or
`xy` fallback is permitted. GRAPE accepts only its actually implemented `xy`
path. Krotov and local retain their existing ordered two-column projection.
For scalar VibLadder and TwoLevel models this remains the historical optimizer
adapter, not a newly introduced physical polarization dependence.

`run_from_config` accepts no unrestricted keyword arguments. Required
`output.dir` is used unless the explicit Python/CLI output argument overrides
it. Required `plot.enabled` is used unless an explicit API value or CLI
`--no-plot` overrides it. Missing plotting result data and exceptions from the
top-level plot call now raise instead of being printed as successful runs. The
separately characterized print-only handling inside optional spectrum and
spectrogram helpers remains deferred to its dedicated visualization change.

The active `configs/` set is exactly three new or current-schema documents: a
Krotov V=0 to V=3 reference, a spectral Krotov example, and a local-optimizer
example. Every active document explicitly supplies the model dipole value/unit
and `control_axes`.

No objective, gradient, penalty, tolerance, field value, time point, local
segment/index/endpoint rule, Krotov update index, or propagation formula changes.
Class-D quantities remain unresolved under O-006.

Implementation commit: this checkpoint.


### D-057: Optimization option values never coerce, fall back, or self-repair

Status: Accepted and implemented on 2026-09-03 as P4.3-o.

Scope: optimizer option values, ordered control axes, Local evaluation modes
and failure handling, and Krotov spectral-constraint input.

Optimization option validation now rejects bool-as-integer, numeric strings,
fractional iteration counts, and nonfinite numeric values rather than applying
Python `bool`, `int`, `float`, or `str.lower` coercions. `target_fidelity` is a
finite value in `[0,1]`. `control_axes` contains two distinct lowercase axes;
duplicate controls such as `xx` are errors. Class-D quantities are checked only
for a finite real representation: this decision assigns no unit, sign, range,
or normalization meaning to them. D-058 later strengthens `gain` without
changing the remaining quantities.

Local `eval_mode` is exactly `target` or `weights`. `weight_mode` is exactly
`by_v`, `by_v_power`, or `custom`; the implicit `*_reverse` suffix is removed
in favor of the existing explicit `weight_reverse` boolean. All Local booleans
must be actual booleans and integer controls must be exact integers. Custom
weights have one explicit array-or-mapping source and are never accepted by a
non-custom mode.

Local no longer requests Hamiltonian eigenvalues when lookahead is disabled.
When lookahead is requested, missing, failing, complex, nonfinite, or
dimension-mismatched eigenvalues raise. Target-weight adjustment and the
display-only running-cost calculation no longer suppress exceptions. Their
formulas and successful-input values are unchanged. The frozen legacy grid,
segment construction, midpoint, field-write slice, shared endpoint ownership,
final odd RK4 prefix, and every field/update expression remain untouched.

Monotonic spectral constraints require nonempty finite `[center,width]` bands,
a supported frequency unit, exact pass/stop and max/sum modes, an actual FWHM
boolean, a finite nonnegative alpha scale, and applicable finite nonnegative
sum weights. The update accepts only a one-dimensional finite nonnegative
`alpha_mask` of the exact rFFT length and evaluates
`U_hat = S_hat / (1 + alpha)` directly. The old `max(1+alpha, 1e-16)` was an
unreachable repair for valid `alpha >= 0` and is removed. Valid outputs remain
bitwise equal to direct division.

Implementation commit: this checkpoint.


### D-058: Local control gain is a positive field-squared-time quantity

Status: Accepted and implemented on 2026-09-07 as P4.3-p.

Scope: Local optimizer gain input, canonical conversion, active configuration,
update-equation characterization, and Local result diagnostics.

The user defines the Local input as gain rather than its reciprocal penalty.
Both `gain` and `gain_units` are required. Exact supported labels are
`(V/m)^2 fs`, `(MV/m)^2 fs`, `(GV/m)^2 fs`, and `(TV/m)^2 fs`;
the canonical internal unit is `(V/m)^2 fs`. Values must be finite, numeric,
and strictly positive. Numeric strings, booleans, missing units, unknown unit
spellings, zero, negative values, and conversion overflow raise before the
optimization loop.

The definition follows directly from the existing expression
`E_a = gain * S * response_a`: the projected dipole response has units
`rad/fs/(V/m)`, so gain has units `(V/m)^2 fs`. The active example changes
only its representation from `1e21` canonical units to
`1000 (GV/m)^2 fs`. The canonical value entering both target and weights
updates is therefore unchanged. Segment construction, midpoint, response,
seed, clipping order, field samples, shared endpoints, slices, indices, and
RK4 prefix are not changed.

The unreachable reciprocal repair `1 / max(gain, 1e-30)` is replaced by
`1 / gain` after strict positive validation. The misleading
`running_cost` result key is removed. `field_fluence_proxy` reports the same
valid-input expression and is explicitly not the authoritative objective.
Additional read-only diagnostics report canonical gain, stored-field vector
maximum and RMS, the fraction of segments altered by componentwise clipping,
and `gain * max(abs(mu'_a))` separately for each ordered control axis.

`c_abs_min`, `drive_abs_min`, and `shape_floor` remain unresolved Class-D
quantities. This decision assigns them no dimension, range, or normalization.

Implementation commit: this checkpoint.


## Open decisions

### O-001: Trajectory endpoint when stride does not divide steps

Resolved by D-026 on 2026-08-11. Typed trajectories always append the endpoint;
legacy low-level kernels remain unchanged during migration.

### O-002: Trace policy for direct Liouville input

Resolved by D-026 on 2026-08-11. Typed `DensityState` requires trace one and
does not normalize or repair its input.

### O-003: Split operator for incoherent ensembles

Resolved by D-026 on 2026-08-11. Typed incoherent pure-state ensembles expose
split propagation; density-matrix split propagation remains unsupported.

### O-004: Renormalization role

Resolved by D-026 on 2026-08-11. Renormalization remains an explicit production
policy and is never silently enabled.

### O-005: SymTop production scope

Resolved in specification by D-052 and implemented for normal simulation by
D-053. Independent Hamiltonian/dipole references, basis ordering, unit
conversion, Morse, dense/CSR, and scaled propagation contracts pass. Legacy
direct classes remain experimental. Split operator, CuPy, and optimization are
explicitly unsupported rather than open physics choices.

### O-006: Optimization reference behavior

D-027 resolves the local optimizer time-array, segment-index, shared-boundary,
and legacy RK4-consumption contracts with exact and bitwise-equivalence tests.
D-028 and D-029 resolve GRAPE and Krotov direction, grid-spacing, and
trajectory-sampling semantics. They do not establish an independent scientific
objective reference. A deterministic four-level V=0 to V=3 Krotov workload is
stored in `benchmarks/krotov-v0-v3-v0.3.{json,npz}` with an independent final
forward propagation and a short integration guard. This protects the current
end-to-end behavior but is not independent evidence for the update equation.
Before algorithmic refactoring, the user must still identify one trusted
reference problem per supported optimizer and acceptable objective and gradient
tolerances, including the spectral-constraint update.

### O-007: Spectroscopy reference behavior

`spectroscopy/absorbance_calculator.py` has 11% measured coverage and several
APIs. Before decomposition, define trusted spectra or sum rules for absorption,
PFID, emission, thermal state handling, broadening, and FFT conventions.

D-023 resolves the numerical-policy ambiguities found by the P1.5 audit:
ignored options, fixed response cutoffs, automatic memory heuristics, fixed
Doppler cutoffs, and duplicated constants are no longer accepted behavior.
O-007 remains open only for independent scientific references and acceptable
tolerances beyond the exact-route equivalence tests.

### O-008: Public v0.3 namespace

The target packages are proposed, but the exact root re-exports remain open.
Decide which small set should be available as
`import rovibrational_excitation as rve` and which names require subpackage
imports.

P0.1 working proposal (not yet accepted):

~~~python
__all__ = [
    "__version__",
    "ElectricField",
    "gaussian",
    "gaussian_fwhm",
    "TwoLevelModel",
    "VibLadderModel",
    "LinearMoleculeModel",
    "TwoLevelParameters",
    "VibLadderParameters",
    "LinearMoleculeParameters",
    "PropagationProblem",
    "PropagationOptions",
    "PropagationResult",
    "propagate",
]
~~~

Specialized capabilities remain public through explicit subpackages:

- `rovibrational_excitation.core`: states, operators, time, units;
- `rovibrational_excitation.fields`: additional envelopes and modulation;
- `rovibrational_excitation.models`: advanced model-owned basis/dipole types;
- `rovibrational_excitation.optimization`: optimization entry functions;
- `rovibrational_excitation.spectroscopy`: spectroscopy facade;
- `rovibrational_excitation.simulation`: configured workflows;
- `rovibrational_excitation.visualization`: plotting helpers.

The proposal intentionally removes generic state/operator classes,
model-specific dipole caches, spectroscopy names, factories, low-level kernels,
and runner helpers from the root. Exact model class names should be accepted
only after the Phase 2 typed contracts show whether a separate `*Parameters`
object is useful or redundant. See `API_INVENTORY.md` for every current name's
disposition.


### O-009: Vibrational-coherence mask in spectroscopy

Resolved by D-025 on 2026-08-11. The old implicit `abs(delta_v) < 2` mask was
replaced by required `pump_probe` (`V_i == V_j`) and `unfiltered` modes. The
selected mode and discarded density norm are observable, and post-probe
radiation/PFID is not filtered.

### O-010: Normal simulation configuration must stop inferring physics

Status: Resolved by D-041 on 2026-08-24.

The Phase 3 validation-ownership audit found several defaults or accepted
inapplicable keys that can change the physical problem without appearing in the
configuration. The recommended Phase 7 typed-schema decisions are:

1. Require `basis_type`; do not infer `linmol`.
2. Require `initial_states`; do not infer the ground state `[0]`. Keep the
   accepted coherent-list semantics from D-007.
3. Replace LinMol `use_M` with a required explicit representation choice such
   as `explicit_m` or `m_incoherent_average`. Require Cartesian `axes` only for
   `explicit_m`; reject it for M averaging and scalar models.
4. Do not require a dummy Jones polarization for TwoLevel or VibLadder. Their
   typed field is scalar under D-010. LinMol Cartesian input continues to
   require a finite nonzero Jones vector.
5. Require `split_interaction` exactly when `algorithm=split_operator` and
   reject it for RK4. For fixed-linear M averaging, accept only the existing
   Cartesian/scalar-z interaction.
6. Make pulse shape explicit: require an envelope kind and the parameters that
   define that envelope, including its center when applicable. Retain explicit
   zero defaults for additive modifiers (`phase`, GDD, TOD) and an explicit
   no-modulation choice because those values mean absence rather than an
   invented physical scale.
7. Reject unknown and model-inapplicable keys. If sinusoidal modulation is
   selected, require and validate all modulation parameters before field
   construction. Replace the mixed-case `Sinusoidal_modulation` key in the
   versioned schema.

Separately, direct `build_model` currently validates required-key presence and
potential names but relies on deeper constructors for numeric type, finiteness,
ranges, Morse bounds, and unit errors. Phase 6 frozen parameter schemas should
own those checks before matrix allocation.

The user accepted items 1-7, explicit external-field injection with strict
grid validation, and frozen model parameter schemas. D-041 is authoritative
for the staged implementation.

### O-011: Remaining field-modulation and legacy unit semantics

Status: Resolved by D-043 on 2026-08-29.

The spectral slope is a physical delay with an explicit time unit; sinusoidal
phase/amplitude multipliers, GDD/TOD Taylor factors, cycle-averaged intensity,
peak field amplitude, and strict structural unit validation are fixed by
D-043. Legacy range heuristics and warning/raw-attribute fallbacks are deleted.

### O-012: Active and archival example quality scope

Status: Resolved by D-044 on 2026-08-29.

Active examples, benchmarks, and scripts are linted and formatted; all three
supported examples execute in CI. Historical files live under
`examples/archives/` and are explicitly excluded until independently migrated
to the current public API.

## Decision template

Copy this template for a new entry:

~~~markdown
### D-NNN: Short title

Status: Proposed | Accepted | Superseded
Scope: Affected modules and behavior

Context and observed current behavior.

Decision.

Consequences:

- required implementation;
- required tests;
- forbidden alternatives.

Implementation commit: hash or pending
~~~
