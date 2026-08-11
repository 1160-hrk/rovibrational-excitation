# Refactoring decision log

Last updated: 2026-08-10

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

An ensemble of state vectors is propagated independently. Vector norm squared
provides each raw statistical weight. Weights are normalized to sum to one
before density operators are summed.

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
eigenvalues, symmetrize, or project a matrix. Mixed-state explicit density input
continues to perform its documented positive trace normalization after
validation.

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

Implementation commit: pending.

## Open decisions

### O-001: Trajectory endpoint when stride does not divide steps

Current behavior records the initial state and every divisible stride. The
final state is absent when `n_steps % sample_stride != 0`.

User decision needed:

- preserve this exact regular-grid behavior; or
- always append the endpoint, making the final interval shorter.

Do not change until decided.

### O-002: Trace policy for direct Liouville input

Current behavior validates positive real trace but does not require trace one
and does not normalize direct Liouville input. Explicit density input through
`MixedStatePropagator` is normalized.

User decision needed:

- require trace one;
- normalize automatically;
- permit positive scaled density operators.

### O-003: Split operator for incoherent ensembles

Direct `MixedStatePropagator(algorithm="split_operator")` can propagate an
ensemble of pure states, but `PropagatorFactory` rejects split operator for
`state_type="mixed"`.

User decision needed: expose this capability or intentionally restrict the
factory to RK4 mixtures.

### O-004: Renormalization role

Current wavefunction solver default is `renorm=False`. Renormalization can hide
integration error but may be useful for long calculations.

User decision needed before API stabilization:

- retain as an explicit production option;
- restrict it to diagnostics;
- remove it and require timestep correction instead.

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

GRAPE, Krotov, local optimization, and spectral constraints currently have 0%
measured automated coverage. Before refactoring, the user must identify one
trusted reference problem per supported algorithm and the acceptable objective
and gradient tolerance.

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

`use_v_mask=True` currently zeros density-matrix elements with
`abs(delta_v) >= 2` before the probe commutator. This is a physical
approximation, not an implementation optimization, and it is presently the
default.

Recommended resolution: make the unmasked response the exact default and move
the mask to an explicitly named approximate mode that reports the discarded
density-matrix norm. Retaining the current implicit default would make an
otherwise exact response route silently depend on a reduced-coherence model.

Do not change this physics-facing default until the user confirms whether
`abs(delta_v) >= 2` coherences are intentionally excluded in production
spectra.

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
