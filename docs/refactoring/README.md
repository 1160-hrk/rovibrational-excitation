# Refactoring source of truth

Last verified: 2026-09-30
Branch: `refactor/v0.3`
Behavioral baseline: `613ce93`

This directory is the authoritative planning and contract reference for the
v0.3 refactor. It is written for both maintainers and Codex. User-facing
documentation under `docs/` may describe the current released API and can lag
behind this refactor; conflicts must be resolved using implementation evidence,
tests, and the decision log.

## Documents

| Document | Purpose | Update trigger |
|---|---|---|
| `PHYSICS_CONTRACTS.md` | Equations, units, time grid, states, Morse rules, coupling, backend capabilities | Any physical or numerical contract change |
| `DECISIONS.md` | Accepted decisions and questions that still require the user | Every resolved or newly discovered ambiguity |
| `TARGET_ARCHITECTURE.md` | Target package tree, dependency rules, typed contracts, old-to-new mapping | Any architecture or ownership change |
| `EXECUTION_PLAN.md` | Ordered phases, task IDs, acceptance criteria, commit rules | At the start and completion of every phase |
| `API_INVENTORY.md` | Current exports, CLI/config routes, factories, examples, and v0.3 disposition | Any public/internal entry-point change |
| `UNIT_BOUNDARY_AUDIT.md` | Remaining explicit-unit debt, safe order, and user-confirmation items | Any physical input boundary migration |
| `VALIDATION_INVENTORY.md` | P1.2-B audit of standalone diagnostics, replacements, and unresolved scale-policy constants | Any legacy validation disposition or recovered scientific intent |
| `PHASE5_ACCEPTANCE_AUDIT.md` | Row-by-row CPU/CUDA numerical-engine acceptance status and transfer debt | Any solver capability, backend transfer, or Phase 5 status change |
| `PHASE7_RUNNER_ACCEPTANCE_AUDIT.md` | P7.1 runner ownership and regression evidence; deferred release gates | Any P7.1 status or runner contract change |
| `PHASE7_RESULT_DISK_SCHEMA_V1.md` | Manifest v1 and atomic result-generation publication | Any result disk layout, loader, or publication change |
| `PHASE7_CHECKPOINT_SCHEMA_V1.md` | Checkpoint payload v1, generation publication, and resume provenance | Any checkpoint schema, loader, or resume validation change |
| `PHASE7_PERSISTENCE_ACCEPTANCE_AUDIT.md` | P7.2 result/checkpoint acceptance evidence and guarantee limits | Any P7.2 completion or broader provenance/durability claim |
| `PHASE7_OPTIMIZATION_REFERENCES.md` | Independent P7.3 optimizer oracles, discrepancies, formulas, and tolerances | Every P7.3 reference or optimizer formula decision |
| `PHASE7_OPTIMIZATION_ACCEPTANCE_AUDIT.md` | P7.3 ownership, reference, fallback, typing, coverage, and explicit-exclusion evidence | Any P7.3 completion or optimizer capability claim |
| `PHASE7_SPECTROSCOPY_REFERENCES.md` | Independent P7.4 response, transform, broadening, and observable oracles | Every P7.4 spectroscopy formula or decomposition decision |
| `PHASE7_SPECTROSCOPY_ACCEPTANCE_AUDIT.md` | P7.4 owner, reference, fallback, and remaining constructor-debt evidence | Any P7.4 acceptance or spectroscopy public-constructor change |
| `PHASE8_RELEASE_READINESS_AUDIT.md` | Local release evidence, external blockers, and exact final transition sequence | Any release gate, CUDA closure, container validation, publication readiness, or version change |
| root `AGENTS.md` | Mandatory operating instructions and document routing | When workflow or required checks change |

## Mission

The refactor aims to make the package:

- physically auditable;
- explicit about units, time grids, normalization, algorithms, and backends;
- modular without hiding hot numerical loops behind costly abstractions;
- testable at model, kernel, workflow, and user API boundaries;
- safe to change without relying on backward compatibility;
- reproducible from configuration through serialized results.

“Complete” does not mean that every module is maximally abstract. It means that
ownership and dependencies are clear, unsupported states are rejected, and
physics changes are detected by tests.

## Current verified baseline

### Tests and quality

| Item | Baseline |
|---|---:|
| Pytest | 1523 passed, 16 skipped (1539 collected) |
| Measured branch coverage | 81% |
| Mandatory CI coverage floor | 47% |
| Ruff findings (active source, tests, examples, benchmarks, scripts) | 0 |
| Files failing format (same active scope) | 0 |
| Historical `examples/archives/` | Explicitly excluded by D-044 |
| Optimization module coverage | 72-100% |
| Strict mypy scope | 84 named modules |
| Spectroscopy coverage | 94% |
| `simulation/runner.py` coverage | 94% |
| RK4 Schrödinger coverage | CPU wrapper 19%; CuPy owner 97% via CPU-backed graph tests |

These rows were last verified locally on 2026-09-30. On 2026-09-16, D-071
through D-073 make device-native CUDA plus a real-GPU run mandatory for v0.3,
define independent optimization/spectroscopy reference construction, and fix
the exact minimal typed root API. P6.3-c implements D-075: `LinMolParameters`
is model-owned, the unused mapping wrapper is removed, and the stateless
dipole implementation is private while retaining exact parity coverage.
Strict mypy covers 41 named modules and P6.3 is complete. P6.3-b implements
D-074: the LinMol basis,
dipole implementation, and model builders now share
`models/linear_molecule/`, the former owners are removed, and the generic
dipole factory drops LinMol to avoid a reverse dependency. All D-070 values
and the M-average workflow remain unchanged. P6.3-a supplied the nine move
guards, while 29 existing physics cases retain independent authority over
selection rules, M averaging, Morse behavior, and propagation.

P6.4-a/D-076 audits the remaining experimental SymTop skeleton before removal.
It has no production caller, differs from the independently referenced D-053
model in basis order/filtering, anharmonic semantics, and transverse phases,
and its dense and CSR dipole routes both fail direct execution. Three new
guards record those differences; no production formula changes.
P6.4-b/D-077 removes that broken legacy basis/dipole, factory, and `jmk`
helper without a compatibility shim. Direct tests now use the production
`models.symmetric_top` owner; D-053 physics references remain unchanged.
The three legacy-only audit tests were replaced with two ownership guards and
one obsolete factory test was removed, yielding 1217 passing tests. Strict
mypy (41 modules), active example smoke tests, sdist/wheel build, Twine, and
isolated wheel import pass; optional CUDA tests remain unverified.
P6.4-c/D-078 assigns the sole production linear-rotor kernel to LinMol,
moves the independent Wigner reference under tests, and removes an unused
rotation export. The shared dipole base and vibration formulas remain separate
follow-up units; no transition formula or production matrix path changes.
P6.4-d/D-079 moves the shared harmonic/Morse function files unchanged to
`models.vibration`, retaining one function object for LinMol and VibLadder.
SymTop remains independent. The old path is removed; CUDA remains unverified.
P6.4-e/D-080 moves the unchanged concrete dipole cache/unit/persistence mixin
to `models.dipole_base` and adds a type-only `core.dipole.DipoleOperator` for
propagation and spectroscopy. Both dynamics reverse imports are gone. A
pre-move test freezes conversion, cache identity, SI views, and the legacy
fallback. Strict mypy now covers 42 modules; CUDA remains unverified.
P6.4-f/D-081 removes the now-empty `dipole` package and its eager root import.
The wheel contains no old package files; physics and numerical paths are
unchanged. Phase 6 remains open for the two exact `models -> dynamics.problem`
imports, to be characterized separately in P6.5.
P6.5-a adds direct model-component projection characterization without changing
source. Its identity, axis-order, and metadata-snapshot guards pass alongside
the full CPU suite; CUDA remains unverified.
P6.5-b/D-082 moves the byte-identical coupling/model contract definitions to
`core.model`. Dynamics keeps the same class objects; models has no upper-layer
imports. Strict mypy covers 43 modules, and active examples plus wheel checks
pass. Phase 6 acceptance audit and real CUDA verification remain separate.
P6.6-a characterizes the remaining simulation-owned fixed-M LinMol basis
before its ownership-only move. Basis order, mapping, Hamiltonian, and invalid
M behavior are frozen; the M-average workflow is unchanged.
P6.6-b/D-083 moves that unchanged class to `models.linear_molecule.basis`.
Simulation retains only the D-017 block workflow and incoherent reduction.
All Phase 6 CPU ownership, validation, and dense/CSR reference criteria pass;
Phase 6 is complete. Real CUDA remains a separate Phase 5 release gate.
P7.1-a/D-084 extracts the unchanged generated-field sampling body to
`simulation.field_preparation`. Existing waveform and helicity references pass;
strict mypy covers 44 modules. No public API or numerical behavior changes.
P7.1-b directly freezes the normal and M-average NPZ schemas, single-write
count, stored population, normalized weights, and caller-parameter JSON before
moving result persistence. Source behavior remains unchanged.
P7.1-c/D-085 extracts the fixed payload assembly and file writes to
`simulation.result_persistence`. Runner keeps the explicit save decision;
schemas and overwrite behavior are unchanged. Strict mypy covers 45 modules.
P7.1-d freezes the current one-case order across validation, immutable case
construction, propagation-time nondimensionalization, post-propagation regime
analysis, and explicit host conversion. Source behavior remains unchanged;
1236 tests pass with 10 optional-GPU skips.

P7.1-e/D-086 moves those guarded stages to `simulation.execution`. Runner now
coordinates preparation, propagation, save policy, and persistence without
owning model construction or numerical evolution. D-017 branching, explicit
host conversion, persistence, and numerical results remain unchanged; strict
mypy covers 46 modules.

P7.1-f freezes the exact `OSError` retry/backoff, immediate non-OS failure,
traceback/parameter error file, and every-second-or-final batch checkpoint
cadence. Source behavior remains unchanged; 1240 tests pass with 10
optional-GPU skips.

P7.1-g/D-087 extracts those retry and error-file operations to
`simulation.safe_execution`. A named tuple-compatible outcome makes failure
observable without changing existing unpacking or multiprocessing behavior.
Retry/backoff, traceback files, checkpoint cadence, and calculations remain
unchanged; strict mypy covers 47 modules.

P7.1-h freezes normal in-memory and resume file-backed summaries, plus
checkpoint-based case exclusion and sweep-path reconstruction. The two summary
sources remain distinct during batch-manager extraction. Source behavior is
unchanged; the full suite has 1243 passes and 10 optional-GPU skips.

P7.1-i/D-088 extracts fixed-size batch execution and checkpoint cadence to
`simulation.batch`. The runner retains case preparation, process-count
choice, resume validation, and separate normal/resume summary policies.
Parallel pool-per-batch behavior is now tested. The full suite has 1244
passes, 10 optional-GPU skips, 78% branch coverage, and 48 strict-mypy
modules.

P7.1-j freezes sweep-order case paths and saved dry-run directory creation.
The source is unchanged; 1246 tests pass with 10 optional-GPU skips.

P7.1-k/D-089 moves normal/resume case-path materialization to
`simulation.case_paths` while keeping sweep expansion pure. The full suite
still passes 1246 cases with 10 optional-GPU skips, 78% coverage, and 49
strict-mypy modules.

P7.1-l fixes normal reporting's scalar/vector/final-time projection,
failure previews, and the absence of a success-only CSV on all-failed
runs. The source is unchanged; 1248 tests pass with 10 optional-GPU skips.

P7.1-m/D-090 moves the guarded normal completion display and in-memory CSV
assembly to `simulation.reporting`. Resume keeps its file-backed summary
owner. The full suite has 1248 passes, 10 optional-GPU skips, 78% coverage,
and 50 strict-mypy modules.

P7.1-n fixes unreadable-checkpoint and missing-parameter error order,
all-complete early exit without summary rewrite, and the resume completion
message before the file-backed summary call. Source is unchanged; 1252
tests pass with 10 optional-GPU skips.

P7.1-o/D-091 moves resume entry preparation to `simulation.resume` and
post-batch reporting to `simulation.reporting`, preserving the all-complete
early return and file-backed summary source. The full suite has 1252
passes, 10 optional-GPU skips, 78% coverage, and 51 strict-mypy modules.

P7.1-p/D-092 requires a true positive-integer checkpoint interval at both
normal and resume entries, including rejection of booleans. Five new
parameterized cases pass; the suite has 1257 passes, 10 optional-GPU skips,
78% coverage, and 51 strict-mypy modules.

P7.1-q removes a verified-unused private parallel wrapper and a no-op
resume expression. The suite remains at 1257 passes, 10 optional-GPU skips,
78% branch coverage, and 51 strict-mypy modules.

P7.1-r/D-093 accepts the runner decomposition with a row-by-row audit in
`PHASE7_RUNNER_ACCEPTANCE_AUDIT.md`. The full suite has 1258 passes,
10 optional-GPU skips, 78% branch coverage, and 51 strict-mypy modules.
P7.1 is complete; result schema, optimization, spectroscopy, CUDA, and
public release remain open.

P7.1-s/D-094 marks the accepted runner checkpoint as package version
`0.3.0.dev1` without tagging or publishing a release. P7.2 schema and
persistence characterization is next; final `0.3.0` remains gated on
Phase 7, Phase 8, and real-CUDA validation.

P7.2-a records the existing disk format and resume failure boundaries in
`PHASE7_PERSISTENCE_BASELINE.md`. Its characterization test freezes numeric
arrays and the legacy pickle-only NPZ regime field before any schema change.
That P7.2-a record is the historical baseline. P7.2-b/D-095 adds a v1
result manifest and strict opt-in loader, documented in
`PHASE7_RESULT_DISK_SCHEMA_V1.md`. Numeric arrays are unchanged; duplicate
pickle-only NPZ regime metadata is removed in favor of its JSON sidecar.
P7.2-c/D-096 routes the resumed-run file-backed summary through the strict
loader, preserving valid CSV values while raising on legacy, corrupt, or
incomplete saved results. P7.2-d/D-097 atomically replaces each result
payload and manifest file without changing numerical arrays. P7.2-e/D-098
does the same for individual checkpoint JSON files. P7.2-f/D-099 publishes
complete normal results through an atomic generation pointer. Old
manifest-v1 direct-layout results remain readable but cannot be implicitly
overwritten. P7.2-g/D-100 atomically publishes each complete checkpoint and
failure-list pair. P7.2-h/D-101 adds strict checkpoint payload schema v1 and
requires the complete ordered expanded-run SHA-256 to match before resume
filtering or execution. Invalid, unversioned, unknown, sidecar-mismatched, or
different-run checkpoints raise explicitly and are never upgraded or repaired
implicitly. The exact contract is in `PHASE7_CHECKPOINT_SCHEMA_V1.md`.
P7.2-i/D-102 then moves all three standalone result-directory plotters to the
strict published-result loader. They use `t_E/E` and `t_p/pop`, reject legacy
NPY collections and incompatible scalar/Cartesian shapes explicitly, and do
not change any calculation or established plot order. The one new allowed
dependency edge is `visualization.result_data -> io.result_schema`.
At P7.2-i, the full suite had 1313 passes, 10 optional-GPU skips, 78% branch
coverage, and 58 strict-mypy modules.
P7.2-j/D-103 accepts that persistence boundary after the single-authority
wiring check, full CPU/coverage/quality/example gates, tracked Markdown/YAML
review, build/Twine validation, and installed-wheel schema smoke. The exact
guarantees and exclusions are recorded in
`PHASE7_PERSISTENCE_ACCEPTANCE_AUDIT.md`. The accepted full-suite baseline is
1314 passes, 10 optional-GPU skips (1324 collected), 78% branch coverage, and
58 strict-mypy modules.

P7.3-a/D-104 finds that the former GRAPE time-local heuristic is not the
gradient of terminal fidelity and differs from the independent finite
difference by approximately `1.79e4` relative error on the fixed diagnostic.
After explicit user approval, GRAPE now reverse-differentiates the exact
normalized dense NumPy RK4 map for the declared terminal-population plus
discrete-L2 objective. The independent direct-RK4 oracle converges to an
observed `5.8e-9` relative-error plateau under a fixed `1e-7` regression bound.
GRAPE now requires an explicit generated or sampled seed and rejects custom
propagators without a discrete derivative. That checkpoint otherwise leaves
Krotov, Local, spectral constraints, Class-D scales, and the historical no-op
convergence predicate unchanged.

P7.3-b/D-106 then finds that the former Krotov path normalized away the
terminal costate scale, used a batch old-state update, and inserted an extra
factor two. It is preserved as `legacy_batch_overlap`, including its stored
reference and spectral example. Standard `krotov` now uses sequential
piecewise-constant interval controls, `dH/dE=-mu`, overlap-scaled costates, and
a required inverse-field-squared-time penalty unit. A direct one-iteration
oracle agrees to `2e-15`; TwoLevel and `V=0..4` to `V=3` transfer and half-step
accuracy tests pass.

P7.3-c independently expands normalized RK4 and both Local update expressions
on the exact D-027 two-segment layout. `weights` and `target` fields agree with
production to relative array-norm error at most `1.22e-16`; trajectory error is
at most `2.23e-16`. No production formula or Local grid/index behavior changes.
P7.3-d independently constructs every Gaussian mask mode and solves the same
filter using an explicit dense DFT and a separate periodic-convolution matrix.
Odd and even lengths agree with production to at most `8.89e-16`; the active
wavenumber conversion is also directly referenced. This validates only the
historical `legacy_batch_overlap` filter, not standard Krotov monotonicity. No
numerical expression changes. All D-072 optimization references now pass. The
full suite has 1365 passes and 10 optional-GPU skips (1375 collected), 79%
branch coverage, and strict mypy covers 62 modules.

P7.3-e1/D-107 now gives all four optimizers one typed result while preserving
three distinct control layouts. It removes ambiguous dictionary keys and
standard-Krotov field aliases without copying, resampling, normalizing, or
reconstructing any result array. Local `weights` mode uses `None` for its valid
missing target. The full suite has 1380 passes and 10 optional-GPU skips (1390
collected), 80% branch coverage, and strict mypy covers 63 modules. Objective
and evaluator decomposition remains the next P7.3-e unit.

P7.3-e2/D-108 adds the typed target objective/evaluator boundary without
forcing one arithmetic expression. Indexed runner fidelity and vector ``vdot``
adjoint fidelity stay distinct; only GRAPE receives the dedicated discrete-L2
objective type. Local ``weights`` remains a separate diagonal-observable
functional. The full suite has 1384 passes and 10 optional-GPU skips (1394
collected), coverage remains 80%, and strict mypy covers 64 modules.

P7.3-e3/D-109 then extracts the exact Local weights and target responses into
distinct typed evaluators. Thresholds, seed signs, gain/shape, clipping, and
the frozen legacy grid remain in the unchanged orchestration. The full suite
has 1387 passes and 10 optional-GPU skips (1397 collected), coverage remains
80%, strict mypy remains at 64 modules, and the objective module is fully
covered.

P7.3-e4/D-110 then gives the historical spectral filter one strict typed
configuration and compiled-filter boundary. The legacy runner no longer
reinterprets validated input with hidden defaults or coercions. The exact
Gaussian mask, rFFT grid, direct ``1+alpha`` denominator, and field update are
unchanged, while standard Krotov continues to reject the option. The full suite
has 1390 passes and 10 optional-GPU skips (1400 collected), coverage remains
80%, strict mypy remains at 64 modules, and the spectral-constraint module is
86% covered.

P7.3-f/D-111 accepts the optimization boundary after auditing its exact
four-algorithm registry, single result/objective/configuration owners,
dependency direction, strict failures, independent optimizer references, and
stored four-level transfer artifact. No numerical branch, formula, grid,
threshold, or update order changes. The full suite has 1396 passes and 10
optional-GPU skips (1406 collected), branch coverage remains 80%, optimization
modules are 72-100% covered, and strict mypy covers all optimizer modules and
the high-level runner within its 72-module scope. P7.4 spectroscopy is next;
Class-D optimizer quantities and CUDA remain explicit open work.

P7.4-a1/D-112 begins spectroscopy decomposition with independent references,
not production movement. Direct Boltzmann populations feed the unchanged
caller-owned density input; a closed-form two-level resonant plus
counter-rotating response and a single-coherence radiation/PFID transform fix
the frequency, damping, phase, and sign conventions. Production agrees at
relative discrepancies below `2.2e-16` and `1.6e-16`, respectively. The
production package still has no thermal-state constructor, and none is
inferred. Broadening, device, area/sum, and remaining observable references are
next; no spectroscopy implementation has moved yet. The full suite has 1398
passes and 10 optional-GPU skips (1408 collected); branch coverage remains 80%.

P7.4-a2/D-113 completes the independent current-behavior reference set. A
direct normalized Gaussian convolution fixes Doppler/Voigt behavior; direct
impulse convolutions fix Gaussian, sinc, and sinc-squared device normalization.
All four exact routes now meet the analytic response, with a separately
recorded `1.51e-12` resonant chunked accumulation difference under a `2e-12`
bound. The weak-susceptibility and zero-response limits also pass. No production
calculation changes. The full suite has 1406 passes and 10 optional-GPU skips
(1416 collected), branch coverage remains 80%, and the monolith reaches 94%.
Structure-only spectroscopy decomposition is next.

P7.4-b1/D-114 moves the unchanged `ExperimentalConditions` and exact unit-label
validator to `spectroscopy.conditions`. The facade, monolith compatibility
name, and factory share the same class; numerical property bodies are unchanged.
The focused suite passes 45 tests, the full suite passes 1408 with 10
optional-GPU skips (1418 collected), total coverage remains 80%, spectroscopy
remains 94%, and strict mypy covers 73 modules.

P7.4-b2/D-116 moves unchanged uniform-grid, Gaussian/Doppler, and normalized
device-convolution kernels to `spectroscopy.broadening`. Existing calculator
methods remain delegates, while D-113 direct references and architecture guards
fix the formulas and dependency direction. The focused suite passes 47 tests;
the full suite passes 1410 with 10 optional-GPU skips (1420 collected), total
coverage remains 80%, and strict mypy covers 74 modules.

P7.4-b3/D-117 moves the byte-identical immutable calculation report to
`spectroscopy.report`. The facade and calculator compatibility name retain one
class identity; all fields and construction behavior are unchanged. The focused
suite passes 48 tests; the full suite passes 1411 with 10 optional-GPU skips
(1421 collected), coverage remains 80%, and strict mypy covers 75 modules.

P7.4-b4/D-118 moves the unchanged response-to-mOD conversion to
`spectroscopy.observables`. The calculator remains a delegate and D-113 fixes
the square-root, constants, and weak-response behavior. The focused suite passes
50 tests; the full suite passes 1413 with 10 optional-GPU skips (1423
collected), coverage remains 80%, and strict mypy covers 76 modules.

P7.4-b5/D-119 moves the unchanged radiation/PFID response loop to
`spectroscopy.transform`. D-112 fixes its sign, phase, indices, and denominator;
public methods and conversion boundaries remain unchanged. The focused suite
passes 52 tests; the full suite passes 1415 with 10 optional-GPU skips (1425
collected), coverage remains 80%, and strict mypy covers 77 modules.

P7.4-b6/D-120 moves unchanged 2D preparation and exact 2D/matrix/loop
response kernels to `spectroscopy.response`. Route-specific accumulation, cache
lifetime, Doppler binding, and all D-112/D-113 values are preserved. The focused
suite passes 54 tests; the full suite passes 1417 with 10 optional-GPU skips
(1427 collected), coverage remains 80%, and strict mypy covers 78 modules.

P7.4-b7/D-121 moves the unchanged CSR commutator, explicit exact/approximate
entry selection, and chunked accumulation to the same response owner. Exact
mode still prunes nothing response-relevant; threshold use and discarded L2 are
unchanged. The full suite remains 1417 passes with 10 optional-GPU skips (1427
collected), coverage remains 80%, and strict mypy covers 78 modules.

P7.4-b8/D-122 accepts numerical ownership through a dedicated audit and five
executable guards. Formula references, dependencies, and failure policy pass.

P7.4-b9/D-123 begins the approved constructor migration. The named
`standard_absorption` path uses model-owned scalar coupling without dummy
polarization and requires `CartesianProjection` otherwise. It preserves the
existing ordinary-absorption arrays bitwise, including the circular analyzer
bra. The legacy constructor and arbitrary-analyzer observable split remain to
be completed. P7.4-b10/D-124 then requires direct `axes` and `pol_int` and
rejects case coercion; scalar standard absorption still requires neither. The
full suite passes 1427 with 10 optional-GPU skips (1437 collected), coverage
remains 80%, and strict mypy covers 79 modules.


P7.4-b11/D-125 then makes the internal observable boundary explicit. All four
response routes return the same complex molecular response and angular-frequency
arrays, while `calculate` owns the one unchanged mOD conversion followed by the
optional device function. Empty chunked output remains exact float zero. No
public analyzer observable or calculation formula changes in this checkpoint;
the full suite passes 1429 with 10 optional-GPU skips (1439 collected), coverage
remains 80%, and strict mypy covers 79 modules.


P7.4-b12/D-126 publishes the approved pre-mOD observable as immutable
`ComplexResponseSpectrum`. `calculate_complex_response` reuses the exact
phase-matching, Doppler, routing, approximation, and report policies, while
instrument convolution remains specific to scalar mOD. The public arrays are a
read-only cm^-1 grid and projected response in C^2 m^2 / J. The typed analyzer
constructor and legacy `pol_det` removal remain next. The full suite passes 1432
with 10 optional-GPU skips (1442 collected), coverage remains 80%, and strict
mypy covers 79 modules.


P7.4-b13/D-127 adds the typed arbitrary-analyzer path. One immutable
`CartesianAnalyzerProjection` carries normalized interaction and analyzer kets;
its axes must exactly match a Cartesian model and it is rejected for scalar
coupling. The named calculator permits complex response and rejects mOD. The
legacy direct `pol_det` surface remains only for the next migration unit. The
full suite passes 1435 with 10 optional-GPU skips (1445 collected), coverage
remains 80%, and strict mypy covers 79 modules.


P7.4-b14/D-128 completes the constructor migration and resolves O-014. One typed
projection replaces raw axes and interaction/detection arrays in direct and
factory construction. Standard projections own every mOD path; analyzer
projections expose only complex response and are rejected by absorption,
radiation, and PFID mOD methods. Standard calculations remain unchanged. The
full suite passes 1435 with 10 optional-GPU skips (1445 collected), coverage
remains 80%, and strict mypy covers 79 modules. Final P7.4 acceptance is next.


P7.4-c/D-129 accepts the final spectroscopy boundary and completes Phase 7.
The complete CPU/coverage/quality/example/build/installed-wheel gates pass;
standard absorption remains numerically unchanged and analyzer measurements are
truthfully complex-only. Analyzer intensity/absorbance, a thermal-state
constructor, and real-CUDA evidence remain outside this acceptance.

P8.0-a/D-105 completes the pre-tag repository-tooling safety subset early.
Local release preparation is read-only by default and never commits, tags,
pushes, or publishes. The tag workflow accepts final versions only and blocks
build/publication until the complete CPU gates and a real self-hosted CUDA
reference pass. Jupyter retains standard authentication and binds only to
localhost by default. The supported examples, parameter template, and generated
index are executable CI contracts; archived examples are not scanned. The full
suite has 1327 passes and 10 optional-GPU skips (1337 collected). Calculation
behavior is unchanged; external GPU and PyPI execution remain unverified.

P8.0-b/D-115 gives all tracked v0.2 artifacts one versioned archive root:
`scripts/` and `optimization_configs/` now live below
`examples/archives/v0_2/`. The three supported optimization configs and every
archived file body are unchanged. Contracts reject loose archived Python files
and the superseded directory names; generated/runtime artifacts remain
untracked. No calculation behavior changes.

P8.1-a/D-130 implements the exact eight-name D-073 package root. All runtime
exports resolve lazily to their authoritative objects; the old model, core, and
spectroscopy convenience names are removed without shims. A fresh root import
loads no workflow or optional heavy dependency. The full suite has 1438 passes
and 10 optional-GPU skips (1448 collected), branch coverage remains 80%, and
strict mypy covers 80 modules. README migration is next; no calculation
behavior changes.

P8.1-b/D-131 corrects the public `run_simulation_case` annotation to its two
existing explicit routes: `field=None` for generated input, or a sampled scalar
or Cartesian field. Runtime logic is unchanged; strict mypy now covers 81
modules.

P8.2-a/D-132 rewrites both public READMEs from the exact root, supported
examples, and verified CPU capability matrix. Seven contracts execute both
quickstarts, resolve links, and reject stale API/evidence. The suite has 1446
passes and 10 optional-GPU skips (1456 collected); broader docs/workflow audit
work remains.

P8.2-b/D-133 rebuilds `docs/README.md` as a current route/status index and
contract-tests its links and rejection of obsolete recommendations. Parameter,
sweep, propagation, and unit guides remain explicitly labeled migration-audit
work rather than being presented as current.

P8.2-c/D-134 rebuilds the parameter guide from authoritative strict schemas and
corrects only comments in the executable template. Required generated/model
keys and stale-name rejection are contract-tested; the template calculation
still passes end to end. Sweep, propagation, and unit guides remain.

P8.2-d/D-135 rebuilds the sweep guide around its actual classifier, insertion
order, singleton scalarization, Cartesian product, paths, and resume identity.
CLI help and a stale test comment are corrected without changing behavior;
strict mypy now covers 82 modules.

P8.2-e/D-136 rebuilds the time-propagation guide from `TimeGrid`, typed
propagation options/results, capability validation, and the actual RK4/split
implementations. It removes unimplemented FFT/Suzuki/adaptive recommendations,
records dense conversion for spectral split, and keeps real CUDA explicitly
unverified. The suite has 1466 passes and 10 optional-GPU skips (1476
collected). No calculation code changes.

P8.2-f/D-137 rebuilds the unit guide from the converter tables and frozen
quantity boundaries. It records preserved caller provenance, canonical internal
views, 2π semantics, direct-field versus intensity acceptance, optimizer units,
spectroscopy exact labels, and unresolved Class-D quantities without inference.
The suite has 1481 passes and 10 optional-GPU skips (1491 collected). No
calculation code changes.

P8.3-a/D-138 removes the development image's persistent tokenless/wildcard
Jupyter configuration, installs from `pyproject.toml` extras, and rebuilds the
container guide around the actual Dev Container and launcher. Static safety,
JSON, shell syntax, link, and repository contracts pass. Docker is unavailable
here, so clean image build and VS Code attach remain external release checks.
The suite has 1486 passes and 10 optional-GPU skips (1496 collected); no
calculation behavior changed.

P8.3-b/D-139 removes the unused Codecov config and false setup guide. GitHub
Actions still enforces the same 47% branch threshold and uploads report/XML
artifacts; no external coverage upload or badge is claimed. Two contracts make
that single authority and absence of stale service files explicit. The suite
has 1488 passes and 10 optional-GPU skips (1498 collected). No calculation or
CI execution behavior changed.

P8.3-c/D-140 turns the clean documentation/workflow audit into mandatory
gates. Repository contracts cover current and archived Markdown local links,
code fences, and YAML/YML syntax. The required quality job runs pinned,
checksum-verified actionlint v1.7.12, with only the release runner's `gpu` label
declared as custom. Both workflows pass locally. The suite has 1492 passes and
10 optional-GPU skips (1502 collected); no calculation behavior changed.

P8.4-a/D-141 publishes the explicit v0.2-to-v0.3 migration contract. It maps
current owners and strict input replacements, distinguishes the old batch
overlap from standard Krotov, rejects inferred conversion of ambiguous legacy
modulation, and requires new schema-v1 runs instead of implicit disk upgrades.
Five contracts verify the guide and its public links. The suite has 1497 passes
and 10 optional-GPU skips (1507 collected); no implementation changed.

P8.4-b/D-142 makes `pyproject.toml` the sole dependency manifest, removes false
spectroscopy subpackage version/contact metadata, and replaces the obsolete
test guide with current commands, markers, evidence rules, and links. Three
contracts guard these authorities. The suite has 1500 passes and 10 optional-GPU
skips (1510 collected); no calculation behavior changed.

P8.4-c/D-143 records the complete local release rehearsal: CPU/coverage/quality,
examples, actionlint, read-only version transition, build/Twine, isolated wheel
import, entry points, and clean payload all pass. The release-readiness audit
keeps v0.3.0 blocked on device-native CUDA plus real-GPU evidence, an actual
container build/attach, publication infrastructure, and the final explicit
version transition. No source or calculation behavior changed.

P5.5-a/D-144 replaces the incorrect fused CuPy RK4 kernel with a separate
CPU-consistent device-native implementation. CPU-backed graph tests cover
sign, stages, trajectory, stride, and renormalization; three real-GPU parity
cases are collected. The suite has 1506 passes and 13 optional-GPU skips (1519
collected). Split device residency and all real-CUDA evidence remain open.

P5.5-b/D-145 moves all three split CuPy modes to a device-native owner without
changing formulas, thresholds, phases, stride, or renormalization. CPU-backed
mode comparisons pass and three real-GPU cases are collected. The suite has
1514 passes and 16 optional-GPU skips (1530 collected); strict mypy covers 84
modules. Only real-CUDA transfer, parity, and performance evidence remains.

P5.5-c/D-146 adds a strict schema-v1 real-CUDA evidence recorder plus manual
pre-tag and mandatory tag-time workflows. It covers RK4 final/trajectory and
all three split modes, checks both device and host input routes against NumPy,
and records synchronized timing without a speed gate. The suite has 1521 passes
and 16 optional-GPU skips (1537 collected). Phase 5 remains open because no
actual-hardware report has been accepted locally.

P8.5-a/D-147 adds one required normal/release container smoke without changing
package behavior. Its minimal context builds the image, verifies non-root
unmounted-image import, and exercises token-authenticated Jupyter through a
read-only checkout. The suite has 1523 passes and 16 optional-GPU skips (1539
collected). Docker execution and manual VS Code attach remain external.

P6.2-c
implements D-069:
the frozen schema now belongs to `models/vib_ladder`, and the unused mapping
and stateless dipole wrappers are removed after a complete caller audit. The
remaining typed builders use the same validation, unit conversion, basis,
Hamiltonian, and dipole implementation. All D-067 values remain unchanged,
P6.2 is complete. P6.2-b implements
D-068:
the VibLadder basis, dipole class, stateless builder, and production builders
now share `models/vib_ladder/`; the three former owners are removed and the
generic dipole factory rejects the moved model rather than creating a new
reverse dependency. All D-067 values remain unchanged. P6.2-a implements D-067:
nine new contracts freeze VibLadder parameter/unit projection, basis/state
order, Hamiltonian, scalar-z coupling, harmonic dipoles, dense/CSR storage,
cache identity, and all current builder paths. Together with the existing
independent Morse and propagation references, 61 focused cases pass without a
source implementation change. P6.1-d implements D-066:
the frozen schema joins `models/two_level`, TwoLevel is removed from the legacy
generic dipole factory, and the unused mapping/stateless builders are deleted.
All D-063/D-064 numerical references remain unchanged, completing P6.1.
P6.1-c implements D-065:
the basis, dipole, stateless dipole builder, and production builders now have
one owner under `models/two_level/`; all three superseded paths are removed and
the D-063/D-064 numerical references remain unchanged. P6.1-b implements D-064:
reduced Planck's constant is derived once as `H/(2*pi)`, all runtime aliases
share it, and Hamiltonian conversions use the central converter. The approved
correction changes the active typed TwoLevel population by at most
`1.142e-13`; dense/CSR disagreement remains `6.78e-20`. Four new contracts
cover constant authority and energy/dipole round trips. P6.1-a implements D-063:
the TwoLevel production parameter projection, basis/state mapping, Hamiltonian,
dense/CSR dipoles, scalar-x coupling, and builder parity are fixed by eight
pre-move characterization cases. Its O-013 finding is resolved by D-064.
P5.4-a implements D-062:
all CPU-verifiable Phase 5 rows now have executable evidence. NumPy CSR split
is exactly equal to dense spectral execution, prepared split kernels exactly
match public results, and an existing device state crosses result finalization
by identity. The dead missing-Numba fallback is removed because Numba is a
required dependency. D-144 and D-145 make CuPy RK4 and split source paths
device-native, and D-146 fixes the evidence schema and workflows. Phase 5
remains open only because an accepted actual-CUDA report has not been produced.
P5.1-c implements D-061:
the dense Liouville RK4 kernel reuses the exactly shared right/next-left
endpoint Hamiltonian. Multiple dimensions and both output modes are bitwise
equal to the retained pre-change Numba loop. The single-thread benchmark
records modest 1.019x-1.044x median speedups and explicitly distinguishes
analytical eliminated allocation traffic from process RSS. A slower
sub-ulp-different full-buffer experiment was rejected. The complete suite is
1188 passed with 10 optional-GPU skips and branch coverage remains 75%.
P5.1-b implements D-060:
validated Liouville wrappers now delegate to a prevalidated dense NumPy/Numba
kernel. Field-stage indices, RK4 order, stride writes, shapes, and caller-array
immutability are fixed. This move makes no speed claim. Strict mypy covers 37
named modules. P4.3-q implements D-059: Local control requires
an explicit `seed_field` or `none` initialization. The seed branch requires a positive direct-amplitude value/unit
pair and segment count while preserving the active 1000 V/m, five-segment
field, trigger, clipping, grid, endpoint, and RK4 behavior. The no-seed branch
injects nothing and raises before propagation when the existing mode-specific
initial trigger identifies a zero-control fixed point; it never falls back.
Strict mypy now covers 36 named modules. P4.3-p implements D-058:
Local control gain now requires a finite positive value/unit pair and converts
to canonical `(V/m)^2 fs` before the unchanged update equations. The active
example preserves `1e21` canonically as `1000 (GV/m)^2 fs`. Result
diagnostics distinguish the field-fluence proxy from the objective and expose
field scale and clipping without changing the frozen Local grid or control
logic. P4.3-o implements D-057:
optimizer option values now reject implicit type conversion, mode fallback,
duplicate axes, invalid spectral constraints, and suppressed Local failures.
Nonnegative spectral alpha uses exact `1+alpha` division, while valid Local
time/index/update behavior and Krotov references remain unchanged. Strict mypy
now covers 35 named modules. P4.3-n implements D-056:
optimization documents now use closed root and per-algorithm schemas, require
explicit ordered control axes, honor required YAML output/plot policy with
documented API precedence, and raise top-level requested plotting failures.
Thirteen incomplete v0.2 YAML files are archived without invented dipoles;
three current-schema configs remain active with explicit dipole units and axes.
All optimizer kernels and local time/index contracts remain unchanged, and
strict mypy covers 34 named modules. P4.3-m implements D-055 by moving the
incomplete v0.2 optimization YAML files to an explicit archive without
inventing missing dipoles. P4.3-l implements D-054:
optimization now consumes the same frozen LinMol, VibLadder, and TwoLevel
parameters and model-owned basis/Hamiltonian/dipole builders as normal
simulation. Strict value/unit pairs and exact quantum-number state tuples
replace legacy optimizer model names and implicit M repair. Basis order, H0,
SI dipoles, every optimizer time/index contract, and the stored Krotov result
remain characterized; optimizer kernels are unchanged. Incomplete tracked
legacy optimizer configs receive no invented dipole values. P4.3-k implements
D-053:
normal simulation now has an independent rigid parallel-band SymTop model with
signed `|v,J,K,M>` ordering, explicit CH3F ortho/para filtering, two rotational
constants, two vibration-rotation couplings, and Cartesian rank-one Wigner
dipoles. NumPy dense/CSR RK4 and strict scaling are verified; CuPy, split
operator, all-isomer pure states, and optimization raise explicitly. The
legacy direct SymTop route and every optimizer numerical kernel are unchanged.
P4.3-j began D-052 with a strict model-layer symmetry foundation and named
presets for `H2`, `D2`, `T2`, `HD`, and `CH3F` without inferred constants or
weights. P4.3-i implements D-051:
the local optimizer now names its direct field limit and seed amplitude
`field_max_v_per_m` and `seed_amplitude_v_per_m`. Exact characterization keeps
the defaults, seed-before-componentwise-clipping order, stored and propagated
field arrays, odd grid, shared endpoints, slices, indices, and RK4-consumed
prefix unchanged. Class-D optimizer quantities remain unresolved. P4.3-h
implements D-050:
Krotov initial fields require an explicit generated/sampled source, physical
value/unit pairs, and strict exact-grid sampled injection. Frozen seed samples,
update indices, and V=0 to V=3 fidelities are unchanged. P4.3-g unit 3 implements
D-049 and completes the low-level field-unit boundary: generated pulses require
time, carrier, and direct amplitude units, while GDD/TOD are complete optional
pairs or exact zero. Frozen nonzero-dispersion samples remain unchanged and
cross-unit waveforms agree within conversion roundoff. Unit 2 implements D-048
for direct construction and canonical fs/V/m storage; unit 1 implements D-047
for arbitrary arrays. Optimizer grids, indices, and values remain unchanged.
P4.3-f implements D-046:
spectroscopy conditions, wavenumber grids, and device resolution now require
exact unit labels and numerical code consumes frozen canonical values. All
spectroscopy formulas and routing remain unchanged. P4.3-e implements D-045:
every normal-simulation scalar physical input now has an explicit paired unit,
and frozen quantity schemas convert once to fs, V/m, C*m, fs^2, fs^3, or
rad/fs. Direct mappings, Python files, batch expansion, and saved parameter
JSON retain the caller values and labels. The stale-label `ParameterProcessor`
and its route-dependent value mutation are deleted. Equivalent ps/MV/cm and
canonical fs/V/m inputs retain sampled-field and population agreement.
P4.3-d implements D-043:
spectral modulation uses a unit-aware physical
delay and explicit phase/amplitude multipliers, GDD/TOD use physical Taylor
coefficients, intensity is cycle-averaged input to peak field, and unit
validation has no heuristic warning or raw-attribute fallback. D-044 reduces
the supported example set to three typed smoke-tested scripts; former v0.2
scripts are explicit archives. Active source, tests, examples, benchmarks, and
scripts are Ruff-clean and format-clean. The preceding D-041 unit 7 added
explicit, report-only convergence assessment.
The caller supplies both grids, the named observable, and tolerance; maximum
absolute difference is reported without changing either calculation. Unit 6
made the final normal-simulation mapping strict. Unknown
keys, model/field/algorithm-inapplicable keys, dummy scalar polarization, and
misplaced split selectors now fail before allocation. Imported Python modules
are excluded from parameter mappings, while non-module helper values remain
visible to strict validation. Generated and externally injected fields converge
to the immutable `SimulationCase`; normal model and fixed-M construction consume
the frozen model schema directly.
Exact field/population references remain unchanged, and strict mypy now covers
23 named modules. The preceding D-042 unit introduced frozen LinMol,
VibLadder, and TwoLevel parameter schemas with neutral frequency names and
required paired units. Model input is validated before allocation and projected
once to the unchanged low-level rad/fs constructors. PHz, THz, wavenumber, and
rad/fs forms retain Hamiltonian, dipole, and fixed-M propagation equivalence.
Generated and injected field parity remains exact. Direct basis, propagation
kernels, optimization paths, and `LocalOptimizerLegacyGridV1` remain unchanged.
Phase 3 completed under P3.2-b; all discovered modules import and the top-level
import graph has no cycles.
P3.2-a had audited the Phase 3 structure and removed the empty
simulation manager, obsolete writable-array time-grid adapter, and two uncalled
`ParameterProcessor` construction helpers. The cleanup routes callers through
the already canonical immutable `TimeGrid`, removes the last `core -> fields`
reverse dependency, and changes no numerical kernel or physical formula.
P3.1-h had moved all plotting helpers unchanged from `plots/` to
`visualization/`, while root import remained independent of optional Matplotlib.
P3.1-g had moved checkpoint, serialization, and summary persistence
unchanged from `simulation/` to `io/`. File formats, filenames, deduplication,
and overwrite behavior remain fixed by contract tests. P3.1-f had moved model
construction unchanged to `models/` and kept the unchanged M-average
propagation workflow at `simulation/m_average.py`.
P3.1-e had moved generic validation unchanged to `core/validation.py`, P3.1-d
had moved nondimensionalization unchanged to `dynamics/scaling/`, and P3.1-c had moved propagation unchanged to `dynamics/`, P3.1-b had moved the unchanged electric-field implementation to
`fields/`, and P3.1-a had moved the unchanged `Hamiltonian` implementation to
`core/operators.py`. These checkpoints built and checked the distributions and
verified the new and removed module paths in isolated wheel installs. The
preceding P2.5 checkpoint reran the seven 4001-point CPU workloads through the
typed public boundary: every final state was exactly equal to the committed
Numba-CSR reference. The fixed 0.10 ms result-contract overhead is material
only for the two-level dense micro-workload and is documented in D-039. The 47%
gate intentionally starts at the accepted Phase 0 baseline; raise it in a
dedicated coverage checkpoint after target ownership and omit policy are
stable. The old README claim of 63% coverage is stale.

### Largest source hotspots

| File | Physical lines | Main concern |
|---|---:|---|
| `spectroscopy/absorbance_calculator.py` | 983 | Conditions moved; response, broadening, transform, and observable responsibilities remain despite 94% coverage |
| `simulation/runner.py` | 628 | Construction, execution, multiprocessing, output, error handling |
| `dynamics/scaling/converter.py` | 562 | Strict scaling, array conversion, and object preparation |
| `dynamics/algorithms/rk4/schrodinger.py` | 478 | Dense, sparse, CPU, GPU, validation paths in one module |
| `fields/field.py` | 478 | Field state, pulse construction, polarization, unit conversion |
| `dynamics/utils.py` | 416 | Backend, units, field mapping, nondimensional preparation |

Line count alone does not require splitting; mixed responsibility and poor
testability do.

## Known repository hygiene problems

- P1.1 removed tracked coverage databases, historical runner-test output, and
  the Notebook checkpoint; saving runner tests now use pytest temporary paths.
- P1.2-A removed the approved root print scripts, two empty RK4 files, the
  disabled old-API split test, and one obsolete migration validator after
  migrating or confirming overlap for their useful assertions.
- P1.2-B removed the approved standalone diagnostics and redundant pytest
  wrapper after recording every collected replacement in
  `VALIDATION_INVENTORY.md`; the ignored generated PNG files were preserved.
- P1.3 removed `dipole/rot/jm_old.py` after Wigner-3j equivalence testing,
  removed unused deprecated builder wrappers and the redundant old-basis demo,
  and retained one spectroscopy archive as explicit Phase 7 migration evidence.
- The competing nondimensional compatibility modules have been removed; the
  strict converter remains pending its target-package move.
- P1.5-B corrected spectroscopy Jones-bra detection, enabled every selected
  Cartesian component, removed the implicit post-projection orientational
  `1/3`, and restricted Doppler broadening to the two transition-specific
  routes.
- P1.5-C replaced the implicit vibrational mask with required `pump_probe`
  (`V_i == V_j`) and `unfiltered` modes. Same-V rotational/M coherence is
  retained, discarded density norm is reported, missing labels raise, and
  post-probe radiation/PFID remains unfiltered. The combined polarization,
  broadening, pathway, and failure contracts are anchored by 26 focused tests.
- P1.6 consolidated duplicate CI workflows and validated mandatory Ruff, limited
  mypy, Python 3.10-3.13 pytest, physics/contracts, 47% branch coverage, build,
  and clean-wheel import gates locally and in GitHub Actions run #47.
  Refactor-branch pushes and explicit manual runs execute that same policy;
  `main` requires the aggregate `Required CI gates` check with strict branch
  synchronization and administrator enforcement.
- Several README examples reference removed or moved APIs.

Each item must be classified as migrate, replace, archive outside the package,
or delete. Do not delete a legacy implementation until unique formulas have
been compared with the replacement.

## Completed preparatory work

| Commit | Result |
|---|---|
| `7ce9419` | Modularized simulation construction and hardened propagation contracts |
| `af21fbe` | Aligned density propagation options, backend honesty, time return, validation |
| `613ce93` | Unified `-mu E` sign and added physical density matrix validation |
| `3b081e1` | Added the reproducible, non-blocking Phase 0 benchmark recorder |
| `6e154ec` | Replaced the Python/SciPy sparse RK4 loop with explicit Numba CSR propagation |
| `93ee9eb` | Made Cartesian and helicity-projected split interactions explicit and physically tested |
| `7d14fda` | Required physical inputs and consolidated strict nondimensionalization |
| `e4102d0` | Made spectroscopy conditions, exact/approximate/auto routes, broadening, and execution reports explicit |
| `82c8d76` | Enforced mandatory quality, physics, coverage, build, and wheel-import CI gates locally |
| `d5c56fc` | Enabled pre-PR refactor-branch and explicit manual CI execution |
| `62e6bfd` | Pinned Ruff and passed every remote required gate in Actions run #47 |
| `834f8ef` | Corrected complex spectroscopy polarization and passed Actions run #49 |
| `874b1c4` | Made pump-probe V-pathway selection explicit and passed Actions run #51 |
| `53bfb2c` | Introduced the typed TimeGrid for normal simulation and passed Actions run #53 |
| `965dcda` | Preserved the exact local-optimizer legacy time layout |
| `b211610` | Made dimensional NumPy RK4 backward direction explicit for Krotov |
| `873ad6e` | Made model coupling and complete physical propagation input one immutable `PropagationProblem` |

These commits are the starting point, not the final architecture.

## Phase status

| Phase | Name | Status |
|---|---|---|
| 0 | Physics characterization baseline | Complete — P0.1-P0.7 CPU baseline recorded; CUDA remains unverified |
| 1 | Repository and CI normalization | Complete — local and GitHub gates pass; `main` requires `Required CI gates` |
| 2 | Typed propagation contracts | Complete — P2.1-P2.5; one typed problem/options input and one backend-explicit endpoint-complete result |
| 3 | Target package migration | Complete — P3.1-a through P3.2-b establish target owners, remove superseded paths, and eliminate top-level cycles |
| 4 | Units and nondimensionalization | Complete for decided contracts — Class-D optimizer values and adaptive integration explicitly deferred |
| 5 | Numerical dynamics engine | In progress — CPU acceptance verified by P5.4-a; backend-native CuPy execution and real-CUDA parity remain |
| 6 | Model consolidation | Complete — P6.1-P6.6-b; model formulas have one owner and supported CPU dense/CSR references pass |
| 7 | Simulation, optimization, spectroscopy decomposition | Complete — P7.1 through P7.4 accepted by D-093, D-103, D-111, and D-129 |
| 8 | Public API, documentation, and release | In progress — P8.0-a tooling and P8.0-b archive consolidation complete; root API/docs, external release evidence, and final bump remain |

Status must be updated only when the acceptance criteria in
`EXECUTION_PLAN.md` are met.

## Conflict resolution

When code, tests, released documentation, and these documents disagree:

1. Determine which behavior the current tests actually enforce.
2. Compare the behavior with `PHYSICS_CONTRACTS.md`.
3. Check `DECISIONS.md` for an explicit user decision.
4. If the answer changes physics or scientific interpretation, ask the user.
5. Add the resolution to `DECISIONS.md` before or with implementation.
6. Update stale public documentation after tests pass.
