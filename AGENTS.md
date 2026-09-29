# Codex repository instructions

Last verified: 2026-09-29
Active refactor branch: `refactor/v0.3`
Verified checkpoints: P8.2-d/D-135 sweep guide; P8.2-c/D-134 parameter guide
Latest infrastructure checkpoint: `7d4368b`
Documentation/workflow audit: `docs/refactoring/DOCUMENTATION_WORKFLOW_AUDIT.md`

## Purpose

This repository implements rovibrational excitation simulations. Refactoring is
allowed to break the old Python API because the repository currently has a
single user. Numerical and physical behavior must nevertheless be preserved
unless the user explicitly approves a physics change.

This file is the entry point for Codex and other coding agents. Detailed
refactoring documents live under `docs/refactoring/`.

## Required reading order

Before changing source code, read these files in order:

1. `docs/refactoring/PHYSICS_CONTRACTS.md`
2. `docs/refactoring/DECISIONS.md`
3. `docs/refactoring/TARGET_ARCHITECTURE.md`
4. `docs/refactoring/EXECUTION_PLAN.md`
5. `docs/refactoring/PHASE5_ACCEPTANCE_AUDIT.md`
6. `docs/refactoring/FALLBACK_AUDIT.md`
7. `docs/refactoring/UNIT_BOUNDARY_AUDIT.md`
8. The source files and tests directly involved in the requested phase

`docs/refactoring/README.md` records the current baseline and document status.

If a legacy README conflicts with the documents above, do not silently choose
one. Check the implementation and tests, then update the decision log or ask
the user when the answer affects physics.

## Non-negotiable rules

- Preserve physical formulas and numerical results during structural changes.
- Do not infer the meaning of a physical constant, threshold, sign, axis,
  normalization, or time step. Ask the user when it is not already decided.
- Add or strengthen characterization tests before replacing numerical logic.
- Never advertise a backend, sparse mode, algorithm, or model capability that
  does not actually execute through that path.
- Do not silently ignore unsupported options. Validate and raise a precise
  error.
- Do not silently clip, renormalize, symmetrize, or otherwise repair user data
  unless that behavior is an explicit documented contract.
- Keep formatting-only changes, file moves, API changes, and physics changes in
  separate commits.
- Use `git mv` for tracked file moves so history remains reviewable.
- Preserve unrelated user changes in a dirty worktree.
- Update the relevant refactoring document in the same commit whenever a
  decision, phase status, public contract, or target path changes.

## Current physical invariants

The authoritative details and formulas are in
`docs/refactoring/PHYSICS_CONTRACTS.md`. The short version is:

- Interaction Hamiltonian: `H(t) = H0 - mu * E(t)`.
- The same sign convention applies to Schrödinger and Liouville evolution.
- The electric-field sampling interval is half of one propagation step:
  `propagation_dt = 2 * field_dt`.
- An RK4 field grid contains `2 * n_steps + 1` points.
- Low-level field construction requires `time_units`, converts time to fs, and
  stores field samples only in V/m; it has no constructor field-unit selector.
- Low-level generated pulses require explicit duration, center, carrier, and
  direct-amplitude units; GDD/TOD are complete optional pairs or exact zero.
- GRAPE and `legacy_batch_overlap` fields explicitly select generated or sampled
  input on the canonical odd field grid. Standard Krotov separately selects
  generated or sampled piecewise-constant interval controls, requires
  `control_dt_fs` and a `lambda_a` unit, and never converts the old field schema.
  None of these routes resamples or silently overrides caller data. GRAPE minimizes
  `1-fidelity+(lambda_a/2)*sum(E**2)` with the exact discrete adjoint of the
  normalized dense NumPy RK4 graph and rejects custom propagators without a
  matching derivative.
- Local optimization requires explicit `initialization`: `seed_field` requires
  a positive direct-amplitude value/unit pair and positive segment count;
  `none` injects no field and raises before propagation when the existing
  initial trigger detects a zero-control fixed point. Seed trigger/signs,
  componentwise clipping order, legacy odd grid, shared endpoints, slices,
  indices, and RK4 prefix remain fixed.
- Local control `gain` requires an explicit unit, converts to positive
  canonical `(V/m)^2 fs`, and enters the unchanged update expression.
- Typed trajectories always include the exact endpoint; if stride does not
  divide the step count, only the final output interval is shorter.
- Typed propagation requires an explicit initial state, algorithm, backend,
  dense/CSR storage, trajectory, stride, scaling, and renormalization policy.
- Every normal-simulation scalar physical input requires its own explicit unit.
  Caller value/unit pairs are saved unchanged; frozen boundaries convert once
  to the documented internal canonical units.
- Spectroscopy conditions require explicit K, Pa, m, ps, and kg-per-molecule
  labels. Public spectral grids and device resolution require explicit cm^-1;
  frozen canonical fields feed the unchanged spectroscopy formulas.
- Low-level arbitrary field arrays require a direct electric-field amplitude
  unit and convert once to V/m. Intensity labels are invalid for signed field
  samples; optimizer grids, endpoints, indices, and values remain unchanged.
- Public propagation requires explicit Cartesian axes or one scalar coupling
  axis and accepts no unrestricted keyword arguments.
- Split-operator calls require the constructor interaction mode again; omission
  or a Cartesian/helicity-projected mismatch is an error.
- Typed density input requires trace one and is never repaired automatically.
- Result arrays remain backend-native until explicit `to_numpy()` conversion.
- Multiple `initial_states` in the normal simulation runner form an
  equal-amplitude, equal-phase coherent superposition and are normalized.
- Incoherent mixtures use `MixedStatePropagator`. Ensemble vector norms encode
  raw statistical weights, which are normalized before propagation.
- A Morse potential with zero anharmonicity is invalid.
- The Morse level parameter is derived per model instance; it must not be
  global state or a fixed `N=200`.
- TwoLevel and VibLadder use scalar coupling and reject the inapplicable
  `polarization` input. LinMol M-resolved coupling is Cartesian.
- Production SymTop is a rigid parallel band in signed `|v,J,K,M>` order with
  CH3F ortho/para filtering, Delta K=0, and Cartesian Delta M=0,+/-1. It
  requires all constants and units, explicit axes, and exactly one spin-isomer
  sector. NumPy dense/CSR RK4 is supported; CuPy, split operator, all-isomer
  pure states, and optimization raise explicitly.
- Density matrices must be finite, square, Hermitian, positive semidefinite,
  and have positive real trace within the documented scale-aware tolerance.
- Liouville propagation currently supports NumPy dense RK4 only.
- Dense Liouville RK4 reuses only the bitwise-identical right/next-left
  endpoint Hamiltonian; its field samples, commutator, stages, and outputs are
  unchanged.
- Validated Liouville wrappers own problem checks and numeric array
  preparation; `liouville_numpy.py` owns only the unchanged prevalidated
  NumPy/Numba loop.
- Spectroscopy exact routes never prune response-relevant nonzero elements.
  Approximation and automatic routing are separate explicit modes with required
  controls and an observable calculation report.
- Spectroscopy experimental conditions are required, and Doppler broadening is
  derived from the actual uniform frequency-grid spacing.
- Simulation convergence is an opt-in comparison of caller-selected coarse and
  fine grids. It uses a caller-named observable, a caller-selected tolerance,
  and maximum absolute difference; it never changes or resamples either grid.
- Checkpoint payloads require schema v1 and an exact failure sidecar. Resume
  validates the complete ordered expanded-run SHA-256 and stored case
  membership before filtering or execution. Invalid, unknown, unversioned, or
  different-run checkpoints raise; no implicit fallback, repair, or upgrade is
  allowed.
- Standalone result-directory plotters consume only validated disk-schema-v1
  results through `io.result_schema`. They map `t_E/E` to field plots and
  `t_p/pop` to population plots; unversioned legacy NPY collections never
  trigger a fallback.

Changing any item above requires explicit user approval and a regression test.

## Dependency direction

The target dependency direction is:

~~~text
cli
 └── simulation / optimization
      ├── models
      ├── dynamics
      └── io
           ↓
         core
~~~

Lower layers must never import runners, CLI modules, plotting, or storage.
Numerical kernels must accept arrays and scalar parameters; they must not load
configuration, perform file I/O, or inspect model-specific classes.

## Required implementation workflow

For each bounded refactoring unit:

1. Identify the current behavior and affected public/physics contracts.
2. Add a characterization or contract test that fails if behavior drifts.
3. Make one kind of change: move, interface replacement, implementation
   replacement, or cleanup.
4. Run focused tests.
5. Run the complete test suite.
6. Run lint and diff checks on touched files.
7. Update documentation and the decision log.
8. Commit with a narrow message.

Do not delete old code until the replacement has tests and all imports have
moved. Since backward compatibility is not required, adapters should be
temporary and removed within the same phase where practical.

## Validation commands

Current local CPU baseline after P8.1-a:

~~~bash
pytest -q
~~~

~~~text
1455 passed, 10 GPU tests skipped (1465 collected)
~~~

The pre-change Phase 0 artifact is `benchmarks/baseline-v0.2.10.json`; the
Numba CSR comparison is `benchmarks/numba-csr-v0.2.10.json`; exact Liouville
endpoint reuse is `benchmarks/liouville-endpoint-reuse-v0.3.json`. CUDA
remains unverified.

Use Ruff without broad automatic fixes while a worktree contains unrelated
changes:

~~~bash
ruff check --no-fix <touched files>
ruff format --check <touched files>
git diff --check
~~~

A repository-wide formatting cleanup is a dedicated Phase 1 commit. Do not run
`ruff check --fix .` as part of a behavioral change.

Coverage must use a temporary data file so repository-local coverage databases
are not created:

~~~bash
coverage run --data-file=/tmp/rve-coverage \
  --source=src/rovibrational_excitation --branch -m pytest -q
coverage report --data-file=/tmp/rve-coverage --show-missing --fail-under=47
~~~

For release-facing phases also run:

~~~bash
python -m build
python -m twine check dist/*
~~~

GPU tests may be skipped when CuPy/CUDA is unavailable. A skipped GPU test is
not evidence that the GPU path works; CI must eventually provide a real GPU
validation job or the capability must remain explicitly unverified.

## Quality baseline and targets

Measured at `613ce93`:

- Full tests: 360 passed, 9 skipped.
- Statement/branch coverage report: 47% total.
- Ruff: 1,143 findings, of which 925 are automatically fixable.
- Ruff formatter baseline: 63 files would be reformatted.
- Current active source, tests, examples, benchmarks, and scripts: 0 format
  failures and 0 Ruff findings; historical `examples/archives/` is excluded by
  D-044.
- Current branch coverage: 80%; the initial mandatory CI floor is 47%.
- Optimization modules: 72-100% measured coverage.
- Spectroscopy monolith: 94% measured coverage.
- RK4 Schrödinger implementation: 20% measured line/branch coverage.
- README claims 63% coverage and contains removed APIs; it is not authoritative.

Targets are defined phase-by-phase in
`docs/refactoring/EXECUTION_PLAN.md`. Coverage must never decrease from the
recorded baseline for a phase.

## Current next work

Phase 0, Phase 1, and Phase 2 are complete. D-039 remains the verified typed
propagation boundary. Phase 3 is complete under D-040. Target package owners exist, superseded paths
are removed, all 129 discovered modules import, internal modules avoid root
convenience imports, and the top-level import graph has no mutual dependency.
P3.2-b moved model selection and required-input validation to
`models/validation.py`; simulation retains time, field, execution, and M-average
workflow validation and translates model errors at its boundary.
Phase 4 is complete for all decided unit/scaling contracts; unresolved Class-D
optimizer quantities and adaptive integration are explicitly deferred. P5.1-b
separates validated Liouville preparation from the unchanged dense NumPy/Numba
kernel. P5.1-c reuses only its exactly shared right/next-left endpoint
Hamiltonian; all recorded outputs are bitwise equal to the retained old loop.
P5.4-a verifies every CPU Phase 5 acceptance row. Phase 5 remains open because
the actual CuPy low-level paths round-trip through host memory and no real CUDA
job is available.
P6.1-a characterizes the complete TwoLevel projection before its ownership
move. P6.1-b resolves O-013 under D-064: `CONSTANTS.HBAR` is the sole derived
authority and all conversion paths share it. P6.1-c implements D-065: basis,
dipole, stateless dipole construction, and model builders now share
`models/two_level/`; the three superseded owners are removed.
P6.1-d implements D-066: `TwoLevelParameters` joins that owner, the unused
mapping/stateless builders are deleted, and the legacy generic dipole factory
no longer imports or constructs TwoLevel. P6.1 is complete.
P6.2-a implements D-067 without changing source behavior. Nine new contracts
plus the existing independent physics references freeze VibLadder units,
basis/state order, Hamiltonian and dipole values, Morse instance-local N and
bounds, scalar-z coupling, dense/CSR storage, cache identity, propagation, and
all current construction-path parity.
P6.2-b implements D-068: the basis, dipole class, stateless builder, and model
builders moved to `models/vib_ladder/`. P6.2-c implements D-069: the frozen
schema joins that owner and the unused mapping and stateless wrappers are
deleted. The generic dipole factory no longer handles VibLadder. D-067 values
remain unchanged and P6.2 is complete.
P6.3-a implements D-070 without changing source behavior. Nine new contracts
plus 29 independent physics cases freeze LinMol parameter conversion, signed
basis order, Hamiltonian, Cartesian dipoles and coupling, coherent basis-index
states, dense/CSR parity, M averaging, Morse behavior, propagation, and every
current construction path before ownership moves.
P6.3-b implements D-074: basis, stateful/stateless dipole implementation, and
all model builders now share `models/linear_molecule/`; the former owners are
removed without shims. The generic dipole factory drops LinMol to avoid a
reverse dependency. D-070 values and the simulation-owned M-average workflow
remain unchanged.
P6.3-c implements D-075: `LinMolParameters` joins that owner unchanged, the
unused mapping wrapper is removed, and the required dipole function becomes a
private model kernel. The package exports only its schema, basis, stateful
dipole, and typed builders. Strict mypy covers 41 modules and P6.3 is complete.
P6.4-a/D-076 finds no production caller of the experimental legacy SymTop
core/dipole paths. They differ from production ordering/filtering, anharmonic
semantics, and x/y phase, and both legacy dipole storage paths fail direct
execution. Delete them without migrating formulae; D-053 production remains
authoritative. The complete suite passes 1220 tests with 10 optional-GPU skips.
P6.4-b/D-077 removes those legacy paths and their factory/`jmk` helper; direct
tests now point to production `models.symmetric_top`. No production formula,
phase, array, or RK4 path changes. The full suite has 1217 passes and 10
optional-GPU skips; measured branch coverage is 77%.
P6.4-c/D-078 moves the unchanged linear-rotor analytic kernel to LinMol and
the independent Wigner reference to tests. The full suite has 1218 passes and
10 optional-GPU skips; shared dipole base/vibration ownership remains.
P6.4-d/D-079 moves the unchanged harmonic/Morse files to `models.vibration`
for LinMol and VibLadder. The full suite has 1219 passes and 10 optional-GPU
skips; real CUDA execution remains unverified.
P6.4-e/D-080 moves the unchanged concrete dipole mixin to `models.dipole_base`
and adds the type-only `core.dipole.DipoleOperator`. The two
`dynamics -> dipole.base` transitions are gone. The full suite has 1223 passes
and 10 optional-GPU skips, 77% branch coverage, and strict mypy for 42 modules.
P6.4-f/D-081 removes the empty `dipole` shell and its root import. The full
suite has 1224 passes and 10 optional-GPU skips; branch coverage remains 77%,
and the wheel contains no obsolete `dipole` files. The two exact
`models -> dynamics.problem` dependencies remain for P6.5.
P6.5-a adds direct characterization of `ModelComponents.to_system_model()`
identity, ordered axes, dimension, and metadata snapshot. The suite has 1226
passes and 10 optional-GPU skips; no implementation has changed yet.
P6.5-b/D-082 moves the byte-identical coupling/model definitions to
`core.model`. `dynamics.problem` re-exports the exact class objects, and
`models` imports core directly. The two reverse imports are eliminated; the
suite has 1227 passes, 10 optional-GPU skips, 77% branch coverage, and 43
strict-mypy modules. Real CUDA remains unverified.
P6.6-a freezes the remaining simulation-owned `FixedMLinMolBasis`: exact basis
order, index mapping, M values, Hamiltonian diagonal, and invalid-M errors.
The suite has 1232 passes and 10 optional-GPU skips; source is unchanged.
P6.6-b/D-083 moves the unchanged class to `models.linear_molecule.basis`.
Simulation retains only the D-017 M-block orchestration and incoherent
reduction. Phase 6 acceptance passes: one model-formula owner, no simulation
model classes, no duplicate factory, strict schemas, zero model reverse
imports, and all supported CPU dense/CSR references. The suite has 1232
passes, 10 optional-GPU skips, 77% branch coverage, and 43 strict-mypy modules.
P7.1-a/D-084 moves the unchanged generated-field sampling body to
`simulation.field_preparation`; runner imports the exact function. Private
decoder tests now import `io` directly. The suite has 1233 passes and 10
optional-GPU skips; strict mypy covers 44 modules.
P7.1-b freezes exact normal and M-average NPZ keys, one write per result,
stored/returned population identity, M-weight normalization, and unchanged
caller parameters in JSON. The suite has 1234 passes and 10 optional-GPU
skips; source is unchanged.
P7.1-c/D-085 moves exact payload assembly/writing to
`simulation.result_persistence`. Runner retains the save decision. NPZ/JSON
schemas, paths, single-write count, overwrite behavior, and D-017 block data
are unchanged. The suite has 1235 passes, 10 optional-GPU skips, 77% branch
coverage, and 45 strict-mypy modules.
P7.1-d freezes validation and immutable-case construction before propagation,
the two established nondimensionalization calls around propagation, regime
analysis, and the single explicit host conversion. Source is unchanged; the
suite has 1236 passes and 10 optional-GPU skips.
P7.1-e/D-086 moves the guarded preparation and propagation stages to
`simulation.execution`. Runner retains save policy, persistence dispatch, and
population projection. D-017 branching, nondimensionalization order, explicit
host conversion, and numerical behavior are unchanged. The suite has 1237
passes, 10 optional-GPU skips, 77% coverage, and 46 strict-mypy modules.
P7.1-f freezes OSError-only retries with one/two-second backoff, immediate
non-OS failure, exact traceback/parameter error files, and the current
every-second-or-final-batch checkpoint cadence. Source is unchanged; the suite
has 1240 passes and 10 optional-GPU skips.
P7.1-g/D-087 moves retry and failure-file handling to
`simulation.safe_execution`. `CaseRunOutcome` remains tuple-compatible and the
runner wrapper retains the multiprocessing call shape. Retry/backoff, prints,
traceback/parameter files, and checkpoint cadence are unchanged. The suite has
1241 passes, 10 optional-GPU skips, 77% coverage, and 47 strict-mypy modules.
P7.1-h freezes normal in-memory summary, resume file-backed summary,
checkpoint-completed case exclusion, sweep-path reconstruction, and the
completed/failed resume checkpoint totals. Source behavior is unchanged; the
suite has 1243 passes and 10 optional-GPU skips.
P7.1-i/D-088 gives fixed-size batch execution and checkpoint cadence to
`simulation.batch`. Runner retains case construction, process-count choice,
resume validation, and both distinct summary policies. Pool-per-batch
behavior is tested; the suite has 1244 passes, 10 optional-GPU skips, 78%
coverage, and 48 strict-mypy modules.
P7.1-j fixes insertion-ordered sweep paths and saved dry-run directory
creation before extraction. The source is unchanged; 1246 tests pass with
10 optional-GPU skips.
P7.1-k/D-089 moves normal/resume case-path materialization to
`simulation.case_paths`; pure sweep expansion stays separate. The full
suite has 1246 passes, 10 optional-GPU skips, 78% coverage, and 49
strict-mypy modules.
P7.1-l fixes returned-population scalar/vector/final-time summary projection,
first-five failure preview, and all-failed no-success-CSV behavior before
reporting extraction. Source is unchanged; 1248 tests pass with 10 optional-GPU
skips.
P7.1-m/D-090 moves normal completion reporting and returned-population CSV
writing to `simulation.reporting`. Resume still uses file-backed
`io.storage.update_summary`. The full suite has 1248 passes, 10 optional-GPU
skips, 78% coverage, and 50 strict-mypy modules.
P7.1-n fixes unreadable checkpoint/missing-parameter error order, all-complete
early exit without summary rewrite, and resume completion reporting before
the file-backed summary call. Source is unchanged; 1252 tests pass with
10 optional-GPU skips.
P7.1-o/D-091 moves resume preparation to `simulation.resume` and post-batch
reporting to `simulation.reporting`; runner retains the all-complete branch.
The suite has 1252 passes, 10 optional-GPU skips, 78% coverage, and 51
strict-mypy modules.
P7.1-p/D-092 requires an actual positive integer checkpoint interval at
both normal and resume entries. Valid cadence is unchanged; the suite has
1257 passes, 10 optional-GPU skips, 78% coverage, and 51 strict-mypy modules.
P7.1-q removes the unused private parallel wrapper and discarded resume
expression; the same full suite, coverage, and strict-mypy gates pass.
P7.1-r/D-093 accepts runner ownership and deterministic normal/resume
behavior under `PHASE7_RUNNER_ACCEPTANCE_AUDIT.md`. The suite has 1258
passes, 10 optional-GPU skips, 78% branch coverage, and 51 strict-mypy
modules. Phase 7 and final v0.3.0 remain open.
P7.1-s/D-094 sets the package version to `0.3.0.dev1` as a development
checkpoint only; no tag or publication is implied. P7.2-a inventories the
unversioned persistence boundary in `PHASE7_PERSISTENCE_BASELINE.md` and
adds exact-array/regime-format characterization. P7.2-b/D-095 writes a
versioned result manifest and supplies a strict opt-in loader; numeric
arrays remain unchanged, while duplicate pickle-only regime metadata moves
to its existing JSON sidecar. P7.2-c/D-096 routes resumed summaries through
the strict loader and surfaces invalid saved results before CSV writes;
normal in-memory summaries are unchanged. P7.2-d/D-097 atomically replaces
individual result NPZ/JSON/manifest files. It does not make the group
transactional: a failed rerun can leave a manifest/payload mismatch, which
the strict reader rejects. P7.2-e/D-098 atomically replaces each legacy
checkpoint JSON file without changing its contents or resume semantics. The
two checkpoint files are not yet a transaction. P7.2-f/D-099 writes new
normal results as immutable generations selected by one atomic
`result_current.json` pointer. Valid direct-layout manifest-v1 results are
readable but require explicit migration before overwrite. Calculation arrays
are unchanged. P7.2-g/D-100 publishes `checkpoint.json` and
`failed_cases.json` as one immutable generation through an atomic
`checkpoint_current.json` pointer. P7.2-h/D-101 adds strict checkpoint
payload schema v1 and binds resume to the complete ordered expanded-run
declaration with SHA-256. Invalid, unknown, unversioned, sidecar-mismatched,
or different-run checkpoints raise before filtering or execution; no implicit
upgrade or repair remains. The historical MD5 identity, cadence, and all
calculations are unchanged.
P7.2-i/D-102 routes all three standalone result-directory plotters through one
strict schema-v1 projection. They no longer inspect legacy NPY filenames or
print-and-return on missing data. The single new dependency edge is
`visualization.result_data -> io.result_schema`; plotting remains outside all
calculation paths.
P7.2-j/D-103 accepts the persistence phase after auditing the single schema
authorities, strict failure policy, preserved arrays/cadence, repository
documentation and workflows, complete CPU/coverage/quality/example gates, and
installed distributions. Full source/environment/content provenance,
directory durability, concurrent writers, migration, and garbage collection
remain explicit future work rather than implied guarantees.
P7.3-a/D-104 replaces the user-approved incorrect GRAPE time-local heuristic
with the exact discrete adjoint of the normalized dense NumPy RK4 objective.
An independent direct-RK4 central difference converges to `5.8e-9` relative
error under a fixed `1e-7` bound. GRAPE requires an explicit generated or
sampled seed and rejects custom propagators. That P7.3-a checkpoint left Krotov, Local, spectral constraints, Class-D
scales, and the no-op convergence predicate unchanged. P7.3-b/D-106 then split
the former Krotov calculation into `legacy_batch_overlap` and implemented the
independently referenced sequential interval-control standard route. P7.3-c
independently reproduces both Local update modes, and P7.3-d independently
verifies the legacy spectral mask, DFT, and periodic convolution
without numerical changes. P7.3-e1/D-107 gives all four solvers one typed
`OptimizationResult` while retaining the exact canonical RK4, Local legacy,
and standard-Krotov interval-control layouts. The suite has 1380 passes and 10
optional-GPU skips (1390 collected), 80% branch coverage, and strict mypy
covers 63 modules. P7.3-e objective/evaluator decomposition is next.

P7.3-e2/D-108 adds one target-population evaluator protocol while preserving
the direct indexed runner arithmetic and vector-vdot adjoint arithmetic as
distinct implementations. The accepted GRAPE discrete-L2 value has its own
typed objective. Local `weights` remains separate. The suite has 1384 passes
and 10 optional-GPU skips (1394 collected), 80% branch coverage, and strict
mypy covers 64 modules.

P7.3-e3/D-109 extracts the exact Local weights and target response arithmetic
into distinct typed evaluators. Lookahead, thresholds, seed signs, gain/shape,
clipping, field writes, and the frozen legacy grid remain in `local.py`. The
suite has 1387 passes and 10 optional-GPU skips (1397 collected), 80% branch
coverage, and strict mypy covers 64 modules. Constraint decomposition must
continue without broadening the legacy spectral filter to standard Krotov.

P7.3-e4/D-110 gives the legacy-only spectral filter one strict frozen
configuration and compiled-mask boundary. It removes duplicate runner parsing
and hidden defaults while preserving the exact Gaussian mask, rFFT grid,
``1+alpha`` denominator, field update, and standard-Krotov rejection. The suite
has 1390 passes and 10 optional-GPU skips (1400 collected), 80% branch
coverage, and strict mypy covers 64 modules. P7.3 acceptance/audit is next.

P7.3-f/D-111 accepts the optimization boundary after checking the exact
four-algorithm registry, single typed result and objective owners, strict
failure policy, dependency direction, independent references, stored
four-level transfer artifact, and all optimizer-facing type boundaries. No
calculation changes. The suite has 1396 passes and 10 optional-GPU skips (1406
collected), branch coverage is 80%, optimization modules are 72-100% covered,
and strict mypy covers 73 modules. P7.4 spectroscopy is next.

P7.4-a1/D-112 independently fixes the existing two-level absorption and
single-coherence radiation/PFID transform conventions before production code
moves. The test constructs Boltzmann populations directly but supplies them
through the existing caller-owned density boundary; production still has no
thermal-state constructor. Closed-form response discrepancies are below
`2.2e-16` and `1.6e-16`. Broadening, device, and normalization/area references
remain next. No production calculation changes. The suite has 1398 passes and
10 optional-GPU skips (1408 collected); branch coverage remains 80%.

P7.4-a2/D-113 completes the current spectroscopy reference set with direct
Doppler/Voigt and device convolutions, exact-route analytic comparisons, and
the weak-susceptibility limit. The resonant chunked route has a characterized
`1.51e-12` sparse/dense accumulation difference under a fixed `2e-12` bound;
no formula changes. The full suite has 1406 passes and 10 optional-GPU skips
(1416 collected), branch coverage is 80%, and the monolith reaches 94%.
Calculation-neutral responsibility moves are next.

P7.4-b1/D-114 moves `ExperimentalConditions` and its strict unit-label
validator into `spectroscopy.conditions`. Number-density and coherence-decay
bodies, facade identity, factory behavior, and all numerical paths are
unchanged. The suite has 1408 passes and 10 optional-GPU skips (1418 collected),
branch coverage remains 80%, spectroscopy remains 94% covered, and strict
mypy covers 73 modules.

P7.4-b2/D-116 moves the unchanged uniform-grid, complex Gaussian/Doppler, and
normalized Gaussian/sinc/sinc² device kernels to `spectroscopy.broadening`.
Calculator method routes remain available; D-113 references and architecture
guards freeze formulas and ownership. The suite has 1410 passes and 10
optional-GPU skips (1420 collected), coverage remains 80%, and strict mypy
covers 74 modules.

P7.4-b3/D-117 moves the byte-identical immutable
`SpectroscopyCalculationReport` to `spectroscopy.report`. Package and calculator
imports retain one class identity; fields and construction are unchanged. The
suite has 1411 passes and 10 optional-GPU skips (1421 collected), coverage
remains 80%, and strict mypy covers 75 modules.

P7.4-b4/D-118 moves the unchanged molecular-response-to-mOD body to
`spectroscopy.observables`. The calculator delegates with the same conditions;
D-113 references freeze the expression and weak-response behavior. The suite
has 1413 passes and 10 optional-GPU skips (1423 collected), coverage remains
80%, and strict mypy covers 76 modules.

P7.4-b5/D-119 moves the unchanged radiation/PFID response loop to
`spectroscopy.transform`. D-112 remains authoritative for sign, phase, indices,
and denominator. Public methods and conversions are unchanged. The suite has
1415 passes and 10 optional-GPU skips (1425 collected), coverage remains 80%,
and strict mypy covers 77 modules.

P7.4-b6/D-120 moves unchanged 2D preparation and exact 2D/matrix/loop
response kernels to `spectroscopy.response`. Route-specific accumulation, cache
lifetime, Doppler binding, and observable conversion are preserved. The suite
has 1417 passes and 10 optional-GPU skips (1427 collected), coverage remains
80%, and strict mypy covers 78 modules.

P7.4-b7/D-121 moves the unchanged CSR commutator, explicit exact/approximate
entry selection, and chunked accumulation to `spectroscopy.response`. Exact
mode retains every response-relevant nonzero; only explicit approximation uses
the threshold. Dispatch, report state, empty-response behavior, and observable
conversion remain unchanged. The suite remains 1417 passes with 10 optional-GPU
skips (1427 collected), coverage remains 80%, and strict mypy covers 78 modules.

P7.4-b8/D-122 accepts numerical spectroscopy ownership and adds executable
facade, owner, dependency, and failure-policy guards. P7.4-b9/D-123 begins the
approved O-014 migration: named standard absorption uses model-owned scalar
coupling or required typed Cartesian projection, and every exact route is
bitwise equal to the prior explicit same-polarization path. P7.4-b10/D-124
removes direct axes/pol_int defaults and case coercion. Explicit arbitrary
`pol_det` and the analyzer-observable split remain. The suite has 1427 passes
and 10 optional-GPU skips (1437 collected), coverage remains 80%, and strict
mypy covers 79 modules.


P7.4-b11/D-125 makes the internal response/observable boundary explicit without
changing calculation results. All four routes return their existing angular
frequency and complex molecular response; the facade applies the unchanged mOD
conversion once and retains device convolution afterward. Exact empty chunked
output remains float zeros. Analyzer response is not yet exposed. The suite has
1429 passes and 10 optional-GPU skips (1439 collected), coverage remains 80%,
and strict mypy covers 79 modules.

P7.4-b12/D-126 exposes the existing projected pre-mOD response through
`calculate_complex_response` and immutable `ComplexResponseSpectrum`. The result
owns read-only cm^-1 and C^2 m^2 / J arrays plus the exact calculation report.
It shares phase matching, Doppler, routing, approximation, and dispatch with
absorbance but never applies mOD or the device function. Existing absorbance
outputs are unchanged. Typed arbitrary analyzer construction and legacy
`pol_det` removal remain. The suite has 1432 passes and 10 optional-GPU skips
(1442 collected), coverage remains 80%, and strict mypy covers 79 modules.

P7.4-b13/D-127 adds `CartesianAnalyzerProjection` and the named
`analyzer_complex_response` path. Interaction and analyzer kets are normalized,
read-only, share exact axes, and must match Cartesian model coupling. Scalar
models reject analyzers. Typed analyzer calculators expose complex response and
reject mOD, and avoid a second Jones normalization. Direct legacy `pol_det`
remains for the next removal unit. The suite has 1435 passes and 10 optional-GPU
skips (1445 collected), coverage remains 80%, and strict mypy covers 79 modules.

P7.4-b14/D-128 completes O-014. Direct/factory construction requires one typed
standard or analyzer projection; raw axes and Jones arguments are gone. Standard
projections own all mOD entry points. Analyzer projections expose complex
response and are rejected by absorption, radiation, and PFID mOD methods.
Standard arrays and numerical kernels remain unchanged. The suite has 1435
passes and 10 optional-GPU skips (1445 collected), coverage remains 80%, and
strict mypy covers 79 modules. Final P7.4 acceptance is next.

P7.4-c/D-129 accepts the final spectroscopy boundary and completes Phase 7.
The 1435-pass CPU suite, 80% branch coverage, 94% calculator coverage, Ruff,
mypy, supported examples/index, build/Twine, and installed-wheel facade import
all pass. No calculation changes. Analyzer intensity/absorbance, thermal-state
construction, and real CUDA remain outside this acceptance.

P8.0-a/D-105 completes the pre-tag repository-tooling safety subset without
changing calculation behavior. Local release preparation is explicit and
read-only by default; it never commits, tags, pushes, or publishes. The release
workflow accepts final versions only and blocks build/publication until full
CPU gates and a real self-hosted CUDA reference pass. Jupyter is authenticated
and localhost-only by default. Supported examples and their generated index
are now CI contracts. External GPU and PyPI execution remain unverified.

P8.0-b/D-115 consolidates every tracked v0.2 artifact under
`examples/archives/v0_2/{scripts,optimization_configs}` without changing file
bodies. Current `configs/` remains exactly three supported documents. Contracts
forbid loose archived Python files and superseded archive directories; ignored
runtime/generated artifacts stay untracked. No calculation behavior changes.

P8.1-a/D-130 implements the exact eight-name D-073 package root through lazy
resolution to authoritative objects. Old root convenience exports are removed
without shims, and a plain package import loads no workflow, persistence,
optimization, spectroscopy, visualization, Pandas, or Matplotlib module. The
suite has 1438 passes and 10 optional-GPU skips (1448 collected), branch
coverage remains 80%, and strict mypy covers 80 modules. README and
workflow/documentation migration is next; no calculation behavior changes.

P8.1-b/D-131 types `run_simulation_case` as accepting the two already-supported
explicit routes: generated input through `field=None`, or externally sampled
scalar/Cartesian input. The keyword remains required; runtime logic is
unchanged. Strict mypy now covers 81 modules.

P8.2-a/D-132 rewrites the English/Japanese public READMEs from current typed
APIs and supported examples. Their quickstarts execute, every local link
resolves, and contracts reject removed APIs, stale badges/coverage, and
unqualified CUDA claims. The suite has 1446 passes and 10 optional-GPU skips
(1456 collected). Broader docs/workflow migration is next.

P8.2-b/D-133 rebuilds `docs/README.md` as the current documentation route/status
index. It removes obsolete runner, example, sparse, and unverified GPU advice;
parameter, sweep, propagation, and unit guides remain explicitly marked for
migration audit.

P8.2-c/D-134 rebuilds `docs/PARAMETER_REFERENCE.md` from the strict validators
and executable template. It covers every required generated/model key, both
field routes, units, applicability, and failure policy without recommending
physical scales. The template logic and values are unchanged.

P8.2-d/D-135 rebuilds the sweep guide around exact classifier precedence,
singleton scalarization, insertion-ordered Cartesian products, paths, and
checkpoint/resume identity. Dry-run help is corrected without changing its
count-only behavior. Strict mypy now covers 82 modules.

The user accepted D-071 through D-073 on 2026-09-16. CUDA is a supported v0.3
target and final release requires real-GPU evidence after device-native kernel
separation. Optimization and spectroscopy decomposition require independent
transparent test-only references; discrepancies must be presented before any
formula change. The Phase 8 root API is exactly the eight-name typed surface
recorded by D-073.
The next work is:

1. P7.1 is accepted and the `0.3.0.dev1` development checkpoint is recorded.
   P7.2-b disk schema v1, P7.2-c strict resumed summaries, and P7.2-d
   result individual-file atomic replacement, P7.2-e checkpoint
   individual-file atomic replacement, P7.2-f whole-result publication,
   P7.2-g checkpoint-pair publication, P7.2-h strict checkpoint schema plus
   validated declared-run resume provenance, and P7.2-i strict standalone
   visualization readers are implemented. P7.2-j accepts this persistence
   boundary. P7.3-a/D-104 completes the independent GRAPE reference and formula
   correction; P7.3-b/D-106 completes the standard Krotov oracle/correction and
   preserves the old route explicitly. P7.3-c completes the direct Local-control
   oracle, and P7.3-d completes the direct DFT/convolution spectral reference,
   both with no formula discrepancy. All required optimization oracles now pass.
   P7.3-e1/D-107 introduces the common typed result and P7.3-e2/D-108 introduces
   typed target evaluators plus the GRAPE discrete-L2 objective, and
   P7.3-e3/D-109 separates the two exact Local response evaluators, and
   P7.3-e4/D-110 types the legacy-only spectral constraint and compiled filter,
   all without changing any fixed calculation. P7.3-f/D-111 completes the
   bounded optimization ownership/acceptance audit. P7.4-a1/D-112 now fixes the
   analytic response and transform conventions; P7.4-a2/D-113 completes the
   remaining independent references. P7.4-b1/D-114 moves the unchanged
   experimental-condition boundary to its dedicated owner, and P7.4-b2/D-116
   moves unchanged broadening/device kernels to their owner. P7.4-b3/D-117 moves
   the immutable report to its owner, and P7.4-b4/D-118 moves absorbance
   conversion to its owner, and P7.4-b5/D-119 moves radiation/PFID response to
   its transform owner. P7.4-b6/D-120 moves the exact dense response kernels to
   their owner, and P7.4-b7/D-121 adds unchanged chunked exact/approximate
   kernels to that owner. P7.4-c/D-129 now accepts the complete spectroscopy
   boundary and closes Phase 7. P8.1-a/D-130 implements the exact lazy D-073
   root API, and P8.2-a/D-132 completes both public READMEs. Phase 8 broader
   documentation/workflow migration is next. Do
   not implement analyzer intensity or OD without its independent reference and
   explicit baseline contract. Complete source/environment and
   generated-array provenance remains separate and must not be overstated. Do
   not silently accept unversioned files.
   `DOCUMENTATION_WORKFLOW_AUDIT.md` inventories all Markdown/YAML/workflows;
   D-105 corrects the release workflow and repository tooling before any tag.
   Root README migration, Codecov disposition, actionlint, actual real-GPU and
   PyPI evidence, and the final version bump remain open Phase 8 gates.
   Preserve the distinct normal in-memory and resume file-backed summaries
   until an explicit tested policy decision changes them; do not conflate
   this with final v0.3.0 release.
2. Preserve the characterized `dynamics.utils.get_dipole_component_SI`
   fallback until a separately approved behavior change; preserve all unit
   conversion and unrelated persistence behavior.
3. Keep Phase 5 open until a real CUDA job can remove and verify the current
   RK4/split `device -> host -> device` round trip. Do not edit that path using
   skipped tests as evidence.
4. Preserve D-061 endpoint reuse. The explored full output-buffer rewrite was
   slower on representative dimensions and introduced sub-ulp differences;
   do not revive it without a separate reference and benchmark.
5. Keep CuPy density propagation unsupported.
6. Do not touch the Class-D `c_abs_min`, `drive_abs_min`, `shape_floor`,
   `learning_rate`, `lambda_a`, or convergence tolerances without the user-defined
   dimensions and independent references.
7. No recorded model-to-upper-layer reverse imports remain; preserve the
   architecture test that rejects their reintroduction.
8. Preserve the characterized visualization debts and fix them only in a
   separate behavior commit.
9. Defer persistence schema versioning and checkpoint-manager redesign to its
   separately tested persistence/API phase.
10. Preserve private optimization adapters, especially
   `LocalOptimizerLegacyGridV1`. All optimization references now pass; obtain
   the remaining spectroscopy references before its Phase 7 decomposition.
11. Preserve the D-044/D-115 support boundary: active examples, benchmarks,
   scripts, and exactly three current configs remain executable and tested;
   versioned archives remain historical until individually migrated and
   smoke-tested. Do not promote ignored runtime/generated artifacts.
