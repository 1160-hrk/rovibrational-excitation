# Refactoring decision log

Last updated: 2026-07-31

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
