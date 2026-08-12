# Refactoring decision log

Last updated: 2026-08-12

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
Scope: LinMol `use_M`, polarization, initial states, propagation, and results

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

The fixed-linear test tolerance is
`128 * machine_epsilon` after Jones-vector normalization. It distinguishes
roundoff from a physical relative phase without introducing a field-scale
threshold.

Implementation anchors:
`simulation/models/linmol_m_average.py`,
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
Scope: core/nondimensional, propagation preparation, returned wavefunction phase

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

Implementation anchors: `core/propagation/mixed_state.py` and
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

Implementation anchors: `core/propagation/{base,schrodinger,liouville}.py`,
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

P2.3-a does not yet change production constructors. P2.3-b must remove the
runner dual-boolean inference and pass the same policy to dipole construction
and propagation. The old factory automatic algorithm selection remains a known
transitional violation and must not be reused by the typed facade.

Implementation anchors: `core/execution.py`,
`core/propagation/capabilities.py`, and
`tests/contracts/test_execution_policy_contracts.py`.

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

SymTop basis and dipole code exist, but the main simulation model factory does
not expose SymTop.

User input and reference data are needed to define:

- supported quantum numbers;
- Hamiltonian model;
- coupling/polarization semantics;
- validated use cases;
- whether it belongs in v0.3 stable scope.

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
