# Executable refactoring plan

Last updated: 2026-08-27
Working branch: `refactor/v0.3`
Starting baseline: `613ce93`

## 1. Execution policy

This plan is intentionally sequential. A later phase may be investigated, but
source migration does not begin until the preceding phase acceptance criteria
are satisfied.

Every task follows:

~~~text
characterize -> change one concern -> focused tests -> full tests
             -> lint/diff checks -> docs -> commit
~~~

A phase is not complete because files moved or tests happened to pass once. It
is complete only when its listed artifacts and gates exist in the repository.

## 2. Baseline snapshot

At the start of this plan:

| Metric | Value |
|---|---:|
| Full pytest | 360 passed, 9 skipped |
| Total measured coverage | 47% |
| Ruff findings | 1,143 |
| Ruff auto-fixable | 925 |
| Ruff format failures | 63 files |
| Python versions declared | 3.10 through 3.13 |
| Python versions in current CI | 3.10 through 3.12 |
| Optimization measured coverage | 0% |
| Spectroscopy measured coverage | 11% |

Completed preparatory commits:

- `7ce9419 refactor: modularize simulation and harden propagation contracts`
- `af21fbe fix: align density propagation contracts`
- `613ce93 fix: enforce physical density matrix evolution`

## 3. Phase 0 — physics characterization baseline

Goal: make structural regressions detectable before package movement.

### P0.1 Inventory public and internal entry points

Status: Complete on 2026-07-31. See `API_INVENTORY.md`. This checkpoint changed
documentation only; package migration and API deletion remain forbidden until
the rest of Phase 0 is complete.

Tasks:

- Enumerate root exports from `rovibrational_excitation.__init__`.
- Enumerate subpackage `__all__` exports.
- Record CLI entry points from `pyproject.toml`.
- Record runner configuration loading paths.
- Record all factories and registries.
- Identify examples importing removed or nonexistent APIs.
- Classify each entry as target public, temporary, internal, or delete.

Artifact:

- `docs/refactoring/API_INVENTORY.md` with current path, callers, target path,
  and disposition.
- Decision O-008 updated with the proposed v0.3 root namespace.

Acceptance:

- Every current root export and console script has a disposition.
- Internal source no longer relies on root convenience imports in newly edited
  modules.

### P0.2 Create physics test layout

Status: Complete on 2026-07-31. The ownership directories and markers exist,
six physics modules are reserved in `tests/physics/README.md`, and legacy test
scripts have explicit dispositions. Empty placeholder modules are forbidden;
P0.3-P0.6 create each listed module together with its first real reference
test.

Create:

~~~text
tests/
├── unit/
├── contracts/
├── physics/
│   ├── test_two_level_reference.py
│   ├── test_vib_ladder_reference.py
│   ├── test_linear_molecule_reference.py
│   ├── test_dipole_selection_rules.py
│   ├── test_solver_invariants.py
│   └── test_dimensional_equivalence.py
├── integration/
└── performance/
~~~

The tree above is the Phase 0 target, not a requirement to add empty Python
files in P0.2. The six `physics/test_*.py` files materialize in their owner
tasks P0.3-P0.6.

Do not move all existing tests immediately. Add the new structure and migrate
tests incrementally so collection remains stable.

Add marker definitions:

- `physics`: trusted scientific reference/invariant;
- `gpu`: requires actual CuPy/CUDA execution;
- `performance`: benchmark; Phase 1 excludes it from ordinary CI;
- `slow`: long deterministic correctness test.

Acceptance:

- `pytest --collect-only` lists every new reference test.
- No “test” script remains uncollected without an explicit archival decision.

P0.2 validation retained all 369 collected items, the moved subset passed 89
tests, the full suite passed 360 with 9 GPU skips, and `gpu`/`performance`
each select exactly 9 tests.

### P0.3 TwoLevel reference cases

Status: Implemented on 2026-07-31 in
`tests/physics/test_two_level_reference.py`. The real NumPy/CuPy parity case is
collected but remains unverified because this environment has no CUDA device;
capability wording therefore remains conditional.

Validation collected 380 tests: 370 passed and 10 GPU tests skipped. The
`physics` and `gpu` markers each select 10 cases. The largest CPU absolute
tolerance is `2e-12` for nondimensional population equivalence; the driven
matrix-exponential reference uses `2e-13` with a 0.002 fs propagation step.

Required tests:

1. Free evolution of a superposition with analytic phase:
   `psi_n(t) = exp(-i E_n t) psi_n(0)`.
2. Density evolution equals the outer product of pure evolution.
3. Constant-drive small-step RK4 agrees with a matrix-exponential reference.
4. Scalar coupling produces identical results for normalized x, y, and mixed
   polarization inputs at the workflow boundary.
5. Dimensional and nondimensional population/time results agree.
6. Dense and supported backend paths agree.

Record all parameter values directly in the test fixture. Do not load an
example configuration whose defaults may change.

Acceptance tolerances:

- analytic free evolution: scale-aware near machine precision;
- RK4 reference: tolerance derived from step size and fourth-order convergence;
- no unexplained absolute tolerance above `1e-8`.

### P0.4 VibLadder and Morse reference cases

Status: Complete on 2026-07-31. All 16 collected cases in
`tests/physics/test_vib_ladder_reference.py` pass. Stored and override
Hamiltonian construction share the confirmed `omega01` formula; two distinct
Morse parameter pairs cover derivation, bounds, and instance isolation.

Validation collected 396 tests: 386 passed and 10 GPU tests skipped. The
`physics` marker selects 26 cases. Energy references use `atol=2e-15`,
scalar-polarization parity uses `2e-14`, and nondimensional population parity
uses `2e-12`.

Required tests:

- harmonic energies and adjacent-level spacings;
- anharmonic energy formula for at least three levels;
- harmonic transition selection rule;
- Morse `N` derivation;
- zero anharmonicity rejection;
- maximum bound-level acceptance and next-level rejection;
- no Morse state leakage between two constructed models;
- scalar-polarization independence;
- dimensional/nondimensional agreement.

The test must use at least two different Morse parameter pairs so a hidden
global value would fail.

### P0.5 LinMol reference cases

Status: Complete on 2026-08-01. All 23 cases in
`tests/physics/test_linear_molecule_reference.py` pass. D-017 replaces the
old implicit `use_M=False -> M=0` interpretation with fixed-linear,
separate-|M|-block propagation and a normalized incoherent population sum.
The reduced workflow agrees with a full M-resolved reference at
`atol=3e-13`; energy and Hermiticity references use near-roundoff absolute
tolerances. Explicit-M x/z response, dense/CSR propagation, coherent cross
terms, reduced work, polarization validation, result serialization, and
ambiguous cross-J rejection are covered. The CuPy selection mask was aligned
with CPU J=0 and Morse behavior, but real GPU parity remains pending CUDA.
The full suite collects 419 tests: 409 pass and 10 GPU tests skip. The
`physics` marker selects 49 cases.

Required tests:

- state index to quantum numbers and reverse round trip;
- exact basis size for `use_M=False` and `use_M=True`;
- known low-lying rovibrational energies;
- x/y/z dipole Hermiticity;
- documented rotational/vibrational selection rules;
- response difference between physically distinct Cartesian polarizations;
- dense/sparse dipole and propagation agreement for a small basis;
- coherent superposition includes cross terms;
- incoherent ensemble omits cross terms.

The user must provide or approve any domain-specific reference value that
cannot be derived unambiguously from the implemented formula.

### P0.6 Solver invariant and convergence suite

Status: Complete on 2026-08-01. All 11 deterministic cases in
`tests/physics/test_solver_invariants.py` pass. RK4 fourth-order convergence,
the analytic RK4 norm-amplification polynomial, explicit renormalization,
left/mid/right sampling, `-mu E`, trajectory/final equivalence, split
unitarity and diagonal-H0 validation, factor-of-two physical time, current
stride endpoint behavior, Liouville trace/Hermiticity, the D-012 threshold
boundary, and unsupported capability errors are covered.

The full suite collects 430 tests: 420 pass and 10 GPU tests skip. The
`physics` marker selects 60 cases. No solver source change was required.
O-001, O-003, and O-004 remain open; their current behavior is characterized
without choosing a future API policy.

Required tests:

- RK4 fourth-order convergence trend on a small analytic system;
- norm drift measurement with `renorm=False`;
- explicit behavior with `renorm=True`;
- split-operator norm conservation;
- split-operator rejection of non-diagonal `H0`;
- left/mid/right electric-field sampling;
- trajectory/final-state equality;
- time-grid endpoint and stride behavior;
- Liouville trace and Hermiticity;
- density validation threshold boundary;
- unsupported capability errors.

Acceptance:

- All tests are deterministic.
- Random inputs use explicit seeds.
- Tests fail when the interaction sign or factor-of-two time rule is reversed.

### P0.7 Record benchmark baseline

Status: Complete on 2026-08-01. `benchmarks/baseline-v0.2.10.json` was
measured from clean source commit `3b081e1` with one excluded warmup and the
median of seven runs. The timed region contains propagation only.

The report covers dimensions 2 (TwoLevel), 16 (VibLadder), and 18
(M-resolved LinMol) through NumPy dense and SciPy sparse RK4, plus dense
TwoLevel Liouville propagation. Pure-state norm errors are at most `2.34e-15`,
the Liouville trace error is `2.95e-18`, and dense/sparse final-state L2
differences are at most `1.58e-16`. Returned trajectory allocations range
from 64,032 to 576,288 bytes for pure states and are 128,064 bytes for the
density case.

The full suite collects 437 tests: 427 pass and 10 GPU tests skip. Marker
selection finds 60 physics, 10 GPU, and 16 performance cases. CuPy is not
installed in the measurement environment, so no GPU performance result is
claimed. Absolute runtime remains non-blocking and environment-specific.

Create a non-blocking benchmark report with:

- environment and dependency versions;
- JIT warmup excluded;
- median of repeated runs;
- TwoLevel, VibLadder, and small LinMol dimensions;
- NumPy dense and sparse;
- CuPy only on a real CUDA environment;
- final norm/trace error;
- peak trajectory memory estimate.

Artifact:
`benchmarks/baseline-v0.2.10.json` plus a human-readable README describing the
measurement command.

### Phase 0 acceptance

Status: Complete on 2026-08-01 for the CPU baseline. Required physics cases,
solver invariants, numerical tolerances, and the non-blocking performance
artifact are recorded without moving the package tree. Real-CUDA parity and
performance remain explicitly unverified because this environment has no
CuPy/CUDA; no GPU capability claim is inferred from skipped tests.

- Required physics matrix is implemented or explicitly blocked by an open
  decision.
- Full test suite passes.
- Baseline numerical outputs and tolerances are documented.
- No package directory migration has started.
- Phase status in `docs/refactoring/README.md` is updated.

## 4. Phase 1 — repository and CI normalization

Goal: make automated quality signals truthful before architectural movement.

### P1.1 Classify and remove generated artifacts

Status: Complete on 2026-08-03. Two tracked coverage databases, 19 historical
runner-test output files, and one Notebook checkpoint were removed after
confirming that they contained repeated runtime metadata and mock tracebacks,
not unique scientific reference data. Ignore rules now cover coverage shards,
tool caches, and Notebook checkpoints. The two runner tests that previously
wrote to the repository now patch their output root to pytest's `tmp_path`.
Focused validation passed all 44 runner and simulation-contract tests. The
full suite passed 432 tests with 10 GPU skips, and no `tests/results/`
directory was recreated.

Inspect, then remove from Git and add ignore rules for:

- root `.coverage`;
- `tests/.coverage`;
- historical `tests/results/`;
- notebook checkpoint files;
- transient build, cache, and result directories.

Do not delete a file containing unique reference data until it is migrated to a
fixture or archive with a documented purpose.

Acceptance:

- `git ls-files` contains no coverage database, runtime result, cache, or
  notebook checkpoint.
- Tests write only to pytest temporary directories.

### P1.2 Normalize test collection

Status: Complete on 2026-08-10. P1.2-A migrated the useful root assertions
and removed disabled, empty, and obsolete test scripts. P1.2-B compared every
remaining standalone diagnostic with collected tests, recorded the evidence in
`VALIDATION_INVENTORY.md`, and removed the approved scripts plus the redundant
`tests/run_tests.py` wrapper. The retained `validation/README.md` redirects
to authoritative pytest and benchmark locations. Ignored diagnostic PNG files
were preserved. `pytest --collect-only -q` finds 505 tests without warnings;
the full suite passes 495 tests with 10 GPU skips.

- Convert useful assertions in root `test_basis_validation.py` to pytest.
- Replace or delete print-based `test_new_api.py`.
- Remove its self-deleting behavior immediately.
- Decide whether the empty detailed RK4 files should be deleted.
- Convert `test_splitop_advanced.py.disabled` or record why it is archived.
- Classify `validation/` scripts as physics test, diagnostic tool, benchmark, or
  delete.
- Ensure all retained correctness checks run through pytest.

Acceptance:

~~~bash
pytest --collect-only -q
pytest -q
~~~

Both commands complete with no collection warnings or hidden root test suite.

### P1.3 Classify legacy implementation files

Status: Complete on 2026-08-10. The competing nondimensional implementation
was removed after strict dimensional-equivalence tests identified the
production path. The analytic rotational `jm_old.py` implementation was
removed after the independent Wigner-3j equivalence suite passed. Unused
deprecated dipole builder wrappers and the redundant old-basis API demo were
also removed; authoritative dipole classes and stateless builders remain.

For each legacy file, identify unique logic and callers:

- `dipole/rot/jm_old.py`;
- `validation/core/*old*.py`;
- archived example scripts;
- deprecated dipole wrapper builders.

If unique logic exists, add an equivalence test before removing it. If no caller
or unique formula exists, delete it in a cleanup commit.

The archived `absorbance_from_density_matrix.py` is intentionally retained as
non-executable migration evidence until Phase 7. It contains legacy PFID,
Doppler, and response formulas that must be characterized against
`AbsorbanceCalculator` before removal. Its duplicated approximate constants
and hard-coded thresholds are evidence to review, not accepted defaults.

Acceptance:

- No file named `old` remains in importable package source.
- Deprecated modules have a scheduled removal task or are removed.
- No import emits a deprecation warning for an API that the target design will
  not retain.

### P1.4 Repository-wide formatting commit

Status: Complete on 2026-08-10. Ruff reformatted 33 files under `src/` and
`tests/`; a second format check changed zero files. The full suite passed 495
tests with 10 GPU skips. No source, test, API, or physics behavior was edited
manually in this commit.

Run only after a clean worktree:

~~~bash
ruff format src tests
~~~

Include supported root tools only if they remain.

This commit contains no semantic edits. Review generated changes and run all
tests.

### P1.5 Ruff lint normalization

Status: Complete on 2026-08-10. Ruff applied 59 safe fixes, then the remaining
ten intensity names, four unused spectroscopy temporaries, and two exact-type
comparisons were resolved manually without changing formulas or method
thresholds. Import sorting exposed and removed a latent `dipole.factory`
package cycle. Ruff now reports zero findings, formatting is stable, the 32
unit-conversion tests pass, and the full suite passes 495 tests with 10 GPU
skips.

First run safe fixes on a clean dedicated branch/commit, then review manual
issues:

~~~bash
ruff check --fix src tests
ruff check --no-fix src tests
~~~

Manually resolve unused variables, ambiguous names, multiple statements, and
import-order issues. Do not use unsafe fixes without reviewing each affected
rule.

Acceptance: zero Ruff findings in configured source and test paths.

### P1.5-A Spectroscopy numerical-policy checkpoint

Status: Complete on 2026-08-10. D-023 replaced the ignored
`sparse_threshold`, fixed absolute response/Doppler cutoffs, implicit optimized
routing, and duplicated constants with explicit tested contracts. Exact routes
retain response-relevant nonzero elements; approximation requires a relative
threshold and reports its discarded commutator norm; automatic selection
requires a memory budget and reports the executed route. Experimental
conditions are required, device broadening is applied when requested, and
Doppler width is derived from the actual uniform grid.

A realistic two-level reference now compares the `loop`, `matrix`, `2d`, and
`chunked` exact paths and exposed a chunked transition-frequency orientation
error plus catastrophic pruning of physical dipoles near `1e-30 C m`. The
focused spectroscopy/unit suite passes 42 tests. The complete suite passes 505
tests with 10 GPU skips, and Ruff remains clean.

This checkpoint resolves the numerical-policy portion of O-007. Trusted
experimental spectra, sum rules, and FFT/broadening reference conventions
remain Phase 7 prerequisites before decomposing the spectroscopy monolith.

### P1.5-B Spectroscopy polarization-response checkpoint

Status: Complete on 2026-08-11. D-024 fixes the
complex Jones contraction: interaction uses ket coefficients and detection
uses the conjugate analyzer coefficients. The response is now invariant under
a global Jones-vector phase, all one to three selected Cartesian axes
contribute, and malformed or zero polarization vectors fail before matrix
construction.

The unconditional post-projection `1/3` susceptibility factor was removed;
orientation averaging is no longer silently applied twice to an M-resolved
state. Doppler broadening is limited to `matrix` and `loop`, which share
transition-specific widths. The mean-transition `2d` broadening and
post-absorbance `chunked` convolution were deleted rather than retained as
nominally exact alternatives.

The 20 focused reference tests cover global-phase invariance, helicity
selection, equal left/right response for an M-symmetric state, sign reversal
under M-orientation reversal, linear-polarization regression, third-axis
participation, strict vector validation, susceptibility conversion,
transition-specific Doppler parity, and unsupported-route errors.

The complete local suite passes 519 tests with 10 GPU skips. GitHub Actions run
#49 passes Ruff/mypy, Python 3.10-3.13, physics/contracts, branch coverage,
build/clean-wheel import, and the protected aggregate `Required CI gates` job.

### P1.5-C Pump-probe pathway-selection checkpoint

Status: Complete on 2026-08-11 under D-025. Every calculator now requires
`phase_matching="pump_probe"` or `"unfiltered"`. Pump-probe selection retains
exactly the pre-probe `V_i == V_j` blocks, including all same-V rotational and
M coherences, because V represents the current workflow's net vibrational
absorption/emission order. A basis without correctly shaped V labels raises;
there is no implicit fallback.

The selection is applied once before dispatch to `matrix`, `loop`, `2d`, or
`chunked`, so numerical route choice cannot change it. Reports expose the mode
and discarded density Frobenius-norm fraction. Radiation and PFID bypass this
pre-probe selection and retain post-probe cross-V optical coherence.

The old `use_v_mask` and `abs(delta_v) < 2` behavior are deleted. The focused
spectroscopy suite now has 26 passing tests, including same-V retention,
cross-V removal, exact-route parity after selection, explicit-mode failures,
observable pump-probe/unfiltered differences, density validation, and a
nonzero PFID/radiation regression. The complete suite collects 535 tests:
525 pass and the same 10 GPU tests skip. Ruff, formatting, and the named strict
mypy modules are clean. GitHub Actions run #51 passes Ruff/mypy, Python
3.10-3.13, physics/contracts, branch coverage, build/clean-wheel import, and
the protected aggregate `Required CI gates` job. Implementation commit:
`874b1c4`.

### P1.6 CI truthfulness

Status: Complete on 2026-08-11. The duplicate test workflow was removed and one
CI workflow now has
mandatory Ruff, a Python 3.10-3.13 full-test matrix, an independent
`tests/physics tests/contracts` job, branch coverage, distribution build, and
clean-wheel import jobs. A final aggregate job rejects a failed, skipped, or
cancelled prerequisite so branch protection has one stable required check.
Pushes to `refactor/**` run the same gates before a pull request is opened, and
`workflow_dispatch` provides an explicit rerun path; neither route changes the
numerical test policy.

The initial branch-coverage floor is the accepted Phase 0 value of 47%; the
current local measurement is 59%. Mypy is mandatory in strict mode only for
three named typed modules, while imported legacy modules are followed silently;
expanding that list is an explicit ratchet instead of a repository-wide
allowed failure. Test XML, coverage reports, distributions, and committed
benchmark summaries are uploaded as artifacts. GPU skips remain explicitly
unverified rather than being reported as backend validation.

Local evidence: the full suite passes 509 tests with 10 GPU skips (519
collected); the independent physics/contracts job passes 167 tests with one GPU
skip; Ruff and the named mypy set are clean; sdist/wheel pass Twine; and the
wheel installs, imports, and passes `pip check` in a fresh environment. The
SPDX MIT metadata and required `sympy` runtime dependency were also corrected
when the clean build exposed the obsolete license form and undeclared import.

Remote evidence: GitHub Actions run #47 passed Ruff/mypy, Python 3.10, 3.11,
3.12, and 3.13, physics/contracts, branch coverage, build/clean-wheel import,
and the aggregate `Required CI gates` job. The `main` branch protection rule
requires that app-bound check with strict synchronization, applies to the
administrator, and rejects force pushes and branch deletion.

Implementation commits: `82c8d76`, `d5c56fc`, and `62e6bfd`.

Update workflows:

- use Ruff formatter and linter; remove redundant Black;
- test Python 3.10, 3.11, 3.12, and 3.13 because all are declared supported;
- make build and wheel import tests mandatory;
- make physics tests fail the job;
- start coverage floor at the measured Phase 0 value and prohibit reduction;
- upload test and benchmark summaries;
- remove `continue-on-error` from gates labeled validation;
- introduce mypy gradually, initially on new typed modules;
- keep GPU support conditional until a real CUDA runner exists.

Acceptance:

- A deliberately failing unit test fails CI.
- A deliberately failing physics test fails CI.
- A formatting or lint error fails CI.
- Coverage below the configured floor fails CI.
- Built wheel installs into a clean environment and imports.

### Phase 1 acceptance

- Full tests pass on declared Python versions.
- Ruff lint and format checks pass repository-wide.
- CI gates reflect actual pass/fail state.
- Generated artifacts and uncollected tests are resolved.
- Coverage is measured consistently and documented.

## 5. Phase 2 — typed propagation contracts

Goal: replace implicit `**kwargs` and variable return types before moving
packages.

Status: in progress. D-026 fixes the typed endpoint, initial-state, density
trace, incoherent split, renormalization, execution-policy, and backend-native
result contracts. Legacy kernels remain unchanged while typed boundaries are
introduced.

Implementation status:

- P2.1-a is complete for the normal simulation path: the frozen `TimeGrid`,
  legacy adapter, validation, and `ElectricField.from_time_grid` are tested;
  every remote required gate passed in Actions run #53.
- P2.1-b is complete for time-array construction. Local optimization keeps
  the D-027 legacy layout. GRAPE and Krotov use canonical `TimeGrid`, explicit
  field spacing, full internal trajectories, output-only thinning, and the
  D-028 backward direction under D-029. Independent objective and gradient
  references remain open under O-006.

### P2.1 Introduce TimeGrid

- Move validated time-grid semantics into a frozen type.
- Keep `FIELD_INTERVALS_PER_PROPAGATION_STEP = 2` as a named invariant.
- Construct `ElectricField` from the TimeGrid rather than reconstructing time
  separately.
- Add dimensional and nondimensional time tests.

Do not remove old calls until each workflow either uses `TimeGrid` directly or
has an accepted, characterized adapter. Under D-027 the local optimizer keeps
its versioned storage layout and exposes only its legacy-consumed odd prefix to
the solver boundary. Under D-029, GRAPE and Krotov construct the canonical grid
from `field_dt_fs`; their optimization calculations always use the full
trajectory and apply `output_stride` only to returned output.

### P2.2 Introduce explicit state kinds

Add distinct input types:

- `PureState`;
- `IncoherentEnsemble`;
- `DensityState`.

P2.2-a is complete: immutable NumPy-host value types validate normalized pure
states, norm-encoded incoherent ensembles, and trace-one density states without
repair. P2.2-b is complete: `MixedStatePropagator` requires
`IncoherentEnsemble | DensityState`, dispatches by type, and unwraps immediately
before the unchanged Schrodinger or Liouville solver. Raw list and square-array
inputs are rejected. P2.2-c is complete: `SchrodingerPropagator` requires
`PureState` and `LiouvillePropagator` requires `DensityState`. Both unwrap the
typed value through a thin facade into the byte-for-byte unchanged array
calculation body. Optimization retains an internal `_propagate_array` migration
bridge so intermediate RK4 vectors are neither validated nor repaired between
segments or forward/backward passes. The bridge is not public API.

### P2.3 Introduce ExecutionPolicy and capabilities

P2.3-a is complete: `ExecutionPolicy` requires typed `ArrayBackend` and
`MatrixStorage` choices with no defaults, and the capability registry encodes
the accepted state/algorithm/backend/storage matrix. Structural incompatibility
is rejected before optional-backend availability and before allocation. P2.3-b
is complete: normal simulation requires `backend`, `storage`, and `algorithm`,
rejects legacy `dense`/`sparse` booleans, and passes one validated policy to
model/dipole construction and pure or fixed-M propagation. NumPy TwoLevel and
VibLadder CSR construction now returns actual SciPy CSR matrices with dense
element parity. P2.3-c is complete: `PropagatorFactory` requires typed state
path, algorithm, execution policy, and renormalization choice; it performs
capability preflight and contains no polarization or sparsity heuristic. P2.3
is complete.

One policy controls backend and storage for model construction and propagation.
A capability registry rejects unsupported combinations before matrix
allocation.

Add parameterized tests for every advertised combination.

### P2.4 Introduce PropagationProblem and PropagationOptions

P2.4-a is complete: immutable `PropagationOptions` requires algorithm,
execution policy, trajectory selection, positive sample stride, scaling mode,
and renormalization policy. Normal simulation validation constructs one object
with no defaults and shares it with ordinary or fixed-M propagation. The typed
factory consumes that object.

P2.4-b is complete: every public solver requires `PropagationOptions`, accepts no
unrestricted `**kwargs`, and rejects solver/options or split-interaction
conflicts before numerical work. P2.4-c is complete: `SystemModel` owns the
basis, Hamiltonian, dipole, and exclusive typed coupling; `PropagationProblem`
owns that model, the exact `ElectricField`/`TimeGrid` pair, and one typed initial
state. Public solvers accept only this problem plus options and temporary result
controls. The private array adapters and all numerical kernels remain frozen for
optimization migration. P2.5 now supplies the final non-conditional result object
without changing that seam.

The field TimeGrid is the only timestep source. No solver-level `dt` override.

### P2.5 Introduce PropagationResult

Status: Complete on 2026-08-14 under D-039.

All public solvers now return one immutable-contract `PropagationResult` with
time, backend-native state, state kind, trajectory flag, backend, and recursively
immutable JSON metadata. Public `return_times` and conditional array/tuple
returns are removed. Final-only results carry one endpoint time; trajectories
carry exact start/end times and append an already computed endpoint after regular
stride samples without changing integration. Normal and fixed-M workflows use
explicit `to_numpy()` conversion at their host analysis/storage boundary.

Private array adapters and kernels remain the optimizer/numerical migration seam.
The boundary requests stride one and thins only the returned output; stride one
reuses the full state array without copying. The temporary full-trajectory cost
for stride greater than one is recorded for Phase 5 rather than changing the
characterized kernel during Phase 2. Metadata records the actual same-call
nondimensional scales and an explicitly scoped deterministic configuration hash.
The baseline recorder itself now executes all seven workloads through this typed
public boundary.

### Phase 2 acceptance

Status: Complete on 2026-08-14.

- No high-level propagator public method accepts unrestricted `**kwargs`.
- Return type no longer changes according to booleans.
- Backend/storage/algorithm errors occur before expensive work.
- The 4001-point seven-workload baseline has exact final-state parity with the
  committed Numba-CSR artifact; the documented 0.10 ms fixed two-level dense
  result-contract overhead is the only greater-than-10% timing exception.
- Full validation is 698 passed, 10 optional-GPU skips, 67% branch coverage,
  clean Ruff/format/diff checks, and strict mypy success for 15 modules.
- API inventory, architecture, decision, module, and baseline documents are
  updated.

## 6. Phase 3 — target package migration

Goal: establish dependency direction using mechanical movement before redesign.

Status: Complete on 2026-08-23. P3.1-a completed on 2026-08-15 under D-040. The generic
`Hamiltonian` implementation moved unchanged from `core/basis/hamiltonian.py`
to `core/operators.py`; all direct imports moved to the target owner, the old
path was removed without a compatibility shim, and an explicit empty
`core/__init__.py` plus AST dependency-boundary tests were added. The legacy
basis/state class consolidation is deliberately deferred to Phase 6 because it
is a redesign rather than a mechanical move. P3.1-b then moved the unchanged
electric-field package to `fields/`, renamed only its generic `core.py` module
to `field.py`, repaired imports and public examples, and removed the old path.
P3.1-c moved the unchanged propagation package to `dynamics/`, repaired direct
imports, and removed the old path without a compatibility shim. All numerical
kernels are exact renames. P3.1-d then moved the unchanged strict scaling
package to `dynamics/scaling/`; only the relative constants import required
repair. P3.1-e moves generic numerical validation unchanged to
`core/validation.py`, eliminating the reverse `core -> dynamics` dependency.
P3.1-f moves construction modules to flat `models/` and keeps M-average
propagation in `simulation/m_average.py`; all computational bodies are exact
renames. One exact `core -> fields` dependency, two `dynamics -> dipole.base`
dependencies, and three higher-layer dependencies from the temporary flat
`models` facade/factory remain exact-allowlisted. P3.1-g moves the unchanged
persistence implementations from `simulation/` to `io/`, repairs only imports,
and removes the old paths. The existing unversioned schemas and checkpoint
manager design are deliberately preserved for a later, separately tested
redesign. P3.1-h moves all five unchanged plotting implementations from `plots/`
to `visualization/`, repairs the optimization runner and root package imports,
and removes the old namespace. The package initializer intentionally exports no
functions so root import keeps optional Matplotlib unloaded and same-named
submodules cannot shadow function aliases.

Suggested movement order:

1. generic states, operators, units, and TimeGrid into target `core`;
2. electric-field modules into `fields`;
3. propagation wrappers/kernels into `dynamics`;
4. simulation model builders into target `models`;
5. persistence modules into `io`;
6. plotting into `visualization`.

For each move:

1. add/import smoke test;
2. `git mv` the file;
3. repair direct imports;
4. run focused and full tests;
5. commit;
6. only then redesign internals in a later commit.

Add an import-boundary test that rejects forbidden dependencies.

P3.1-a verification: 90 focused tests; 700 passed and 10 optional-GPU skips in
the full suite; 67% branch coverage; clean Ruff, format, strict mypy, and diff
checks; successful sdist/wheel build, Twine checks, and isolated wheel import
with the new module present and old module absent.

P3.1-b verification: 154 focused tests; 702 passed and 10 optional-GPU skips in
the full suite; 67% branch coverage; clean Ruff, format, strict mypy, and diff
checks; successful sdist/wheel build, Twine checks, wheel-content audit, and
isolated import with new fields modules present, root identity preserved, and
the old subpackage absent.

P3.1-c verification: 184 focused tests with 6 optional-GPU skips; 704 passed
and 10 optional-GPU skips in the full suite; 67% branch coverage; clean Ruff,
format, strict mypy, and diff checks; successful sdist/wheel build, Twine
checks, wheel-content audit, and isolated import with `dynamics` present and
`core.propagation` absent.

P3.1-d verification: 122 focused tests with 1 optional-GPU skip; 705 passed
and 10 optional-GPU skips in the full suite; 67% branch coverage; clean Ruff,
format, strict mypy, and diff checks; successful sdist/wheel build, Twine
checks, wheel-content audit, and isolated import with `dynamics.scaling` present
and `core.nondimensional` absent.

P3.1-e verification: 128 focused tests with 6 optional-GPU skips; 706 passed
and 10 optional-GPU skips in the full suite; 67% branch coverage; clean Ruff,
format, strict mypy, and diff checks; successful sdist/wheel build, Twine
checks, wheel-content audit, and isolated import with `core.validation` present
and `dynamics.algorithms.validation` absent.

P3.1-f verification: 183 focused tests with 1 optional-GPU skip; 708 passed
and 10 optional-GPU skips in the full suite; 67% branch coverage; clean Ruff,
format, strict mypy, and diff checks; successful sdist/wheel build, Twine
checks, wheel-content audit, and isolated import with `models` and
`simulation.m_average` present and `simulation.models` absent.

P3.1-g verification: 73 focused tests; 714 passed and 10 optional-GPU skips in
the full suite; 67% branch coverage; clean Ruff, format, strict mypy, and diff
checks; successful sdist/wheel build, Twine checks, wheel-content audit, and
isolated import with `io` present and the three old simulation persistence
modules absent. Three moved implementation files are 100% exact renames.

P3.1-h verification: 18 visualization/time-contract tests and 39 broader
optimization, CLI, reference, and architecture tests pass; 723 passed and 10
optional-GPU skips in the full suite. Branch coverage rises to 69%; Ruff, format,
strict mypy, diff, sdist/wheel, Twine, wheel-content, and isolated import checks
all pass. Five moved implementation files are 100% exact renames, `plots` is
absent, and root import does not load Matplotlib.

P3.2-a performs the first acceptance cleanup after the mechanical moves. It
removes an empty `simulation/manager.py`, the unreferenced
`simulation/timegrid.py` writable-array wrapper around the already canonical
`TimeGrid.from_bounds`, and two uncalled construction helpers on
`ParameterProcessor`. Tests now call the typed TimeGrid owner directly while
retaining the same values, midpoint layout, exact endpoints, and rejection
conditions. Removing the field-construction helper eliminates the final
`core -> fields` reverse dependency. No formula, integration step, indexing,
threshold, or backend behavior changes.

The acceptance audit loads all 96 discovered source modules and finds one
remaining top-level mutual dependency: `models <-> simulation`, caused by
`models.factory` importing simulation-owned input validation while
`simulation.runner` imports model construction. The three factories are not
duplicates: model construction, dipole construction by basis, and propagator
selection have different responsibilities. P3.2-b should move unchanged model
validation ownership to `models.validation`; model/dipole consolidation stays
in Phase 6.

P3.2-a verification: 97 focused tests pass; 726 tests pass with 10 optional-GPU
skips in the full suite; branch coverage remains 69%. Ruff, formatting, strict
mypy for 14 modules, diff, sdist/wheel, Twine, wheel-content, all-module import,
dependency, and isolated-install checks pass. The wheel contains
`core/time.py`, excludes both removed simulation modules, and exposes neither
uncalled construction helper.

P3.2-b extracts only model selection, required model-key sets, and potential
name validation from `simulation.validation` to `models.validation`. Predicate
order, lower-case normalization, implicit defaults, required-key sets, accepted
potential names, and messages remain unchanged. Direct model construction now
raises `ModelConfigurationError`; the simulation boundary translates it to its
existing `SimulationConfigurationError` with the same message. Time-grid,
field, polarization, execution, capability, split-mode, and M-average validation
remain simulation-owned. No numerical or physical code moves.

This removes `models.factory -> simulation.validation`, the last top-level
mutual dependency. The import audit reports no top-level cycle; all 97
discovered modules import successfully. The four remaining exact transition
debts are two `models -> dynamics.problem` and two
`dynamics -> dipole.base` imports, all explicitly deferred to Phase 6 because
resolving them requires model/operator ownership consolidation.

P3.2-b verification: 74 focused model, simulation, and physics tests pass; the
full suite passes 732 tests with 10 optional-GPU skips; branch coverage remains
69%. Ruff, formatting, strict mypy for 15 modules, diff, sdist/wheel, Twine,
wheel-content, dependency, all-module import, and isolated-wheel checks pass.
Phase 3 acceptance is complete.

### Phase 3 acceptance

Status: Complete on 2026-08-23 under D-040.

- Source tree matches `TARGET_ARCHITECTURE.md` at the package level.
- No circular imports.
- Internal modules do not import root convenience exports.
- Old empty directories and duplicate factories are removed.
- Wheel includes every target package.

## 7. Phase 4 — units and nondimensionalization

Goal: perform unit conversion exactly once and reduce overlapping policy.

Status: P4.1 strict scale/fallback contract completed early on 2026-08-06
(implementation commit `4c33359`). Complete-generator scaling now uses the
centered eigenspectrum, active dipole operator norms, and peak field-vector
magnitude. ZeroField and inactive scale provenance are explicit. Absolute
Schrodinger phase is restored after centering. Heuristic auto-timestep and
invented zero scales now raise.
P4.2 completed on 2026-08-10 under D-022 (`7d14fda`). The 25-name public
scaling surface was reduced to strict transformation, scale metadata, exact
conversions, and neutral reporting. `analysis.py`, `strategies.py`, `impl.py`, automatic
timestep wrappers, heuristic strength verification, and demo factories were
removed. Raw-array units and object coupling semantics are now required. P3.1-d moved
this implemented policy unchanged to its target `dynamics/scaling` owner. Scale
provenance has already been attached to same-call `PropagationResult` metadata
under P2.5.

The 2026-08-09 explicit-fallback audit also rejects removed and unknown solver
options and prevents dipole CuPy requests from becoming NumPy arrays. Remaining
P1/P2 findings and physics-facing default decisions are in FALLBACK_AUDIT.md.

P4.3-a implemented the first explicit quantity boundary on 2026-08-27 under
D-042. Generated carrier input now requires neutral `carrier_frequency` plus
`carrier_frequency_units`, is validated by immutable `Frequency`, and is
normalized once to `rad/fs`. PHz, THz, Hz, wavenumber, and angular-frequency
forms have field-level equivalence tests. The legacy ambiguous key is rejected.

P4.3-b implemented frozen model schemas on 2026-08-27 and requires neutral
model-frequency names with paired units. LinMol, VibLadder, TwoLevel, and fixed-M construction validate
finite scalar input before allocation. PHz, THz, wavenumber, and canonical
`rad/fs` forms produce equivalent Hamiltonian/dipole arrays; fixed-M propagation
also has cross-unit final-population equivalence. Old unit-encoded runner keys
raise, while direct low-level basis APIs and all propagation formulas remain
unchanged. The `ParameterProcessor` no longer pre-converts typed model frequency
fields or `energy_gap`, correcting the old noncanonical TwoLevel
double-conversion path. The complete CPU suite passes 834 tests with 10
optional-GPU skips; branch coverage is 70%, and Ruff, formatting, strict mypy
for 17 modules, all 100 module imports, build, and Twine gates pass.

P4.3-c characterized the remaining legacy conversion boundary on 2026-08-28.
Every advertised direct frequency, energy, dipole, electric-field, time, GDD,
and TOD unit has a canonical round-trip test. Frequency-to-Hamiltonian
conversion covers every advertised frequency/energy pair, and intensity aliases
share one peak-field convention. The complete CPU suite passed 904 tests with
10 optional-GPU skips and measured branch coverage was 71%.

P4.3-d implements D-043 on 2026-08-29. Spectral modulation now takes an
explicit physical delay, uses the accepted phase/amplitude multipliers, and
makes zero depth an exact identity. GDD/TOD use physical Taylor factors, and
intensity is documented and tested as cycle-averaged input to peak field.
The warning/range validator, fixed 1000 fs estimate, raw dipole fallback,
exception downgrade, and non-strict parameter conversion path are deleted.
Strict propagation validation checks canonical accessors and structural
consistency without guessing a physical scale. Local optimizer source, grids,
indices, and numerical kernels remain unchanged. The complete CPU suite passes
924 tests with 10 optional-GPU skips; measured branch coverage is 72%, and
active Ruff, formatting, mypy, smoke-example, build, and Twine gates pass.

P4.3-e implements D-045 in bounded units beginning 2026-08-30. Unit 1 adds
frozen scalar boundaries for dipole, peak field, GDD, and TOD without changing
converter formulas. Unit 2 replaces the unit-encoded normal-simulation
`mu0_Cm` input with required `dipole_scale/dipole_scale_units`, converts
once to C*m in the frozen model schema, and keeps low-level model and dipole
formulas unchanged. Unit 3, completed on 2026-08-31, adds frozen
`GeneratedFieldParameters`. Every generated-field time and amplitude has a
required paired unit; optional GDD/TOD require both members or neither.
Validation converts once to fs, V/m, rad/fs, fs^2, and fs^3 before calling the
unchanged waveform formulas. Unit 4 removes the general
`ParameterProcessor`; direct mappings, Python files, batch expansion, and
saved parameters retain the original value/unit pairs. Cross-unit waveform and
population references pass, the full CPU suite passes 937 tests with 10
optional-GPU skips, and strict mypy passes for all 23 configured modules.

P4.3-f implements D-046 on 2026-08-31. Spectroscopy conditions now require
explicit K, Pa, m, ps, and kg-per-molecule labels and expose frozen canonical
fields to the unchanged formulas. Every absorbance, radiation, PFID, direct 2D,
and device-function spectral grid requires an explicit cm^-1 label. Device
resolution is a conditional value/unit pair. Missing or unsupported labels
raise without fallback. All exact/approximate routing, polarization,
phase-matching, Doppler, response, and Beer-Lambert calculations are unchanged.
The complete CPU suite passes 944 tests with 10 optional-GPU skips.

P4.3-g begins D-047 on 2026-08-31. Unit 1 requires a direct electric-field
amplitude unit on every `add_arbitrary_Efield` array and converts it once to
V/m. Signed arbitrary arrays reject intensity units. Existing GRAPE, Krotov,
and local-optimizer arrays are explicitly labeled V/m without changing any
value, time point, endpoint slice, segment index, or propagation call. The
complete CPU suite passes 945 tests with 10 optional-GPU skips. Constructor
defaults and generated low-level pulse quantities remain later P4.3-g units.

P4.3-g unit 2 implements D-048 on 2026-08-31. Direct `ElectricField` and
`ZeroField` construction now requires `time_units`, converts once to internal
fs, and accepts no meaningless constructor `field_units`. Stored fields remain
V/m. `from_time_grid` is explicitly canonical; requested field-scale output
requires a direct amplitude unit; two uncalled alternate constructors are
removed. Seconds/fs construction parity passes. Every optimizer-created grid
is labeled fs without changing local-optimizer points, endpoints, slices,
indices, values, or RK4 calls. The complete CPU suite remains 945 passed with
10 optional-GPU skips.

P4.3-g unit 3 implements D-049 on 2026-08-31 and completes the low-level
field-unit boundary. Generated pulses require duration, center, carrier, and
amplitude units; amplitude accepts direct field units only. GDD and TOD require
a complete value/unit pair when present and otherwise retain exact zero.
Canonical nonzero-dispersion samples are frozen, and cross-unit pulse outputs
agree within double-precision conversion roundoff. No waveform formula,
polarization path, optimizer grid, update index, or Krotov reference changes.
The complete CPU suite passes 957 tests with 10 optional-GPU skips.

P4.3-h implements D-050 on 2026-09-01. Krotov initial fields require the
explicit `generated` or `sampled` source. Generated Gaussian-FWHM seeds require
duration, center, carrier, direct amplitude, and polarization with a unit for
every physical scalar; GDD/TOD are complete optional pairs or exact zero.
Sampled two-component fields require a direct amplitude unit, are copied and
converted once to V/m, and must match the canonical odd field grid exactly.
Legacy names, source conflicts, unknown initial-field keys, intensity labels,
complex/nonfinite arrays, and length mismatch raise without fallback. Frozen
seed samples and the V=0 to V=3 fidelities are unchanged. The Krotov update
loop, objective, time grid, factor-of-two index, and endpoints are untouched.
The complete CPU suite passes 972 tests with 10 optional-GPU skips.

P4.3-i implements D-051 on 2026-09-01. The local optimizer's unit-ambiguous
`field_max` and `seed_amplitude` keys are replaced by
`field_max_v_per_m` and `seed_amplitude_v_per_m`; the old keys raise with the
required replacement. Defaults remain `1e12 V/m` and `1e3 V/m`, and exact
characterization preserves seed-before-componentwise-clipping behavior. The
legacy odd grid, tail endpoints, shared-boundary ownership, segment midpoint,
slices, lookahead index, field samples, and RK4-consumed prefix do not change.
Class-D optimizer quantities remain untouched. The complete CPU suite passes
975 tests with 10 optional-GPU skips.

P4.3-j begins the D-052 SymTop preparation on 2026-09-01 without changing a
numerical model. A strict `models.symmetry` layer represents point-group
families, optional permutation-inversion labels, rotational symmetry states,
nuclear-spin policies, and source-versioned molecule-name presets. `H2`,
`D2`, `T2`, and `HD` provide explicit linear-rotor weights. `CH3F`
provides only its validated K-modulo-three ortho/para sectors and refuses a
weight until signed-K symmetry adaptation is implemented. Presets contain no
physical constants and unknown names never fall back. The complete CPU suite
passes 983 tests with 10 optional-GPU skips.

Remaining Phase 4 work:

- add independent SymTop Hamiltonian and dipole primitive references, then
  implement the accepted rigid parallel-band model without reusing the broken
  legacy formulas by assumption;
- connect symmetry filters to model basis construction only after exact basis
  ordering and unfiltered parity are characterized;
- consolidate optimization model construction with the frozen model schemas
  only after exact Hamiltonian, dipole, basis-ordering, and result parity;
- defer Class-D optimizer penalties and tolerances until independent references
  and user-defined dimensions exist;
- define an error-controlled adaptive integrator separately, if wanted.
Tasks:

- define explicit quantity/unit types or validated value-plus-unit dataclasses;
- keep pure conversion functions in `core/units`;
- preserve complete-problem scaling in `dynamics/scaling` without introducing
  competing policy;
- preserve same-call scale serialization in results;
- add property-style round-trip tests.

Acceptance:

- numerical kernels contain no unit conversion calls;
- no mutable object parameter changes during conversion;
- dimensional and nondimensional reference observables/time agree;
- one documented internal dimensional unit exists per quantity;
- all current supported direct input units have round-trip tests (completed by
  P4.3-c).

## 8. Phase 5 — numerical dynamics engine

Goal: make solver contracts common while retaining efficient backend-specific
kernels.

### P5.1 RK4

Status: P5.1-a completed early on 2026-08-03 in `6e154ec`. NumPy dense and
CSR propagation now use separate allocation-stable Numba kernels. CSR input
is canonicalized without approximate truncation, `sparse=True` is explicit,
and final-only propagation allocates one returned state.

`benchmarks/numba-csr-v0.2.10.json` records 100.6x, 50.4x, and 44.8x
speedups over the former Python/SciPy sparse paths for TwoLevel, 16-level
VibLadder, and 18-state LinMol. Dense/sparse final differences are at most
`1.11e-16`, and final norm errors are at most `2.34e-15`. A separate
final-only tridiagonal diagnostic measured 5.67x at dimension 64 and 24.77x
at dimension 256. Full validation collected 442 tests: 432 passed and 10 GPU
tests skipped.

CuPy dense and Liouville kernel separation remain part of the later P5.1
completion; this early unit does not claim all of P5.1 complete.

Separate:

- validation and preparation;
- NumPy dense kernel;
- NumPy sparse kernel;
- CuPy dense kernel;
- Liouville NumPy kernel.

Unify field-stage indexing, interaction sign, stride semantics, and result
construction through tests, not through runtime abstraction inside hot loops.

### P5.2 Split operator

- require diagonal `H0` explicitly;
- sample both Cartesian field components at propagation midpoints;
- use a static eigensystem for fixed direction and M-diagonal rotations for changing xy direction;
- expose `cartesian` and `helicity_projected` as distinct physical models;
- validate component Hermiticity and xy rotation covariance without silent repair;
- accept sparse inputs but state explicitly that spectral eigenvectors are dense;
- keep NumPy and CuPy construction and final-state shape aligned;
- compare Cartesian propagation against RK4 at two step sizes;
- benchmark setup and propagation separately before making a speed claim.

### P5.3 Backend transfer policy

Decide whether public results are host arrays or backend-native arrays and
encode it in `PropagationResult`. Eliminate repeated transfer.

### Phase 5 acceptance

- every advertised capability has an executing test;
- CPU/GPU parity is tested where infrastructure permits;
- no silent fallback;
- physics baselines pass;
- median performance regression is below 10% or explicitly approved;
- memory use is documented for trajectories.

## 9. Phase 6 — model consolidation

Goal: co-locate model formulas, parameters, basis, Hamiltonian, dipole, and
coupling.

Order:

1. TwoLevel;
2. VibLadder;
3. LinMol;
4. SymTop only after O-005 is resolved.

For each model:

- add frozen parameter schema;
- move basis and state mapping;
- move Hamiltonian formula;
- move dipole/selection-rule code;
- expose coupling capability;
- eliminate duplicate simulation builder;
- update registry and reference tests.

Under D-041, each frozen schema validates required keys, numeric types,
finiteness, ranges, units, and model-specific constraints before allocating a
basis or operator. The migration first proves that a valid schema projects to
the exact existing constructor values and results; strict rejection replaces
the mapping path only after parity is established.

Morse `N` remains derived instance-local data.

### Phase 6 acceptance

- one model package owns every model-specific formula;
- simulation contains no model physics;
- no duplicate model/dipole factory path;
- model construction validates all required parameters;
- reference tests pass for dense/sparse/backend combinations supported.

## 10. Phase 7 — workflows, optimization, and spectroscopy

### P7.1 Simulation runner

Split current runner into:

- configuration parsing;
- typed case construction;
- one-case execution;
- sweep expansion;
- process management;
- persistence/checkpoint service;
- progress/reporting.

The D-041 migration is divided into separately testable units:

1. require `basis_type` and `initial_states` — complete on 2026-08-24;
2. replace normal-simulation `use_M` with required `m_resolved` or
   `m_incoherent_average` — complete on 2026-08-24;
3. introduce scalar and Cartesian field values plus exact external-sample
   injection — complete on 2026-08-24;
   Verification: exact generated/injected population parity covers scalar,
   M-resolved Cartesian, and fixed-M incoherent-average paths; generated
   helicity metadata is preserved and arbitrary Cartesian fields are never
   decomposed by inference. The full suite is 765 passed and 10 skipped with
   70% branch coverage; Ruff, format, 16-module strict mypy, build, and Twine
   checks pass;
4. require generated-envelope and modulation discriminators — complete on
   2026-08-27;
   Verification: four supported legacy envelope functions and both sinusoidal
   modulation types produce exactly equal sampled arrays; missing, removed,
   unknown, and inapplicable selectors fail before construction. Custom and
   two-width Voigt waveforms use external sampled-field injection. The full
   suite is 785 passed and 10 skipped;
5. consume the Phase 6 frozen model schemas in a typed `SimulationCase` —
   complete on 2026-08-28;
   Verification: both field routes converge to one immutable case before
   allocation; model and fixed-M builders consume its frozen schema directly;
   836 tests pass with 10 optional-GPU skips and branch coverage remains 70%;
6. reject every unknown and inapplicable key at the final schema boundary —
   complete on 2026-08-28;
   Verification: model, generated/external-field, polarization, and algorithm
   applicability are discriminated before allocation; imported modules are not
   mistaken for Python parameters; 852 tests pass with 10 optional-GPU skips
   and branch coverage remains 70%;
7. add an opt-in convergence-report service that never changes the requested
   time grid — complete on 2026-08-28;
   Verification: generated and external scalar routes require identical
   endpoints and a strictly smaller fine field-grid step; calculation
   parameters otherwise match; the caller supplies the named observable and
   finite nonnegative tolerance; reports use maximum absolute difference and
   read-only observable values. No result is written and no grid is modified.
   The full suite is 872 passed with 10 optional-GPU skips, branch coverage
   remains 70%, and strict mypy covers 21 named modules.

Every unit characterizes the old valid-case projection first. None may route
normal-simulation time construction through `LocalOptimizerLegacyGridV1` or
alter the local optimizer frozen arrays and indices.

One-case execution must be a deterministic pure application service aside from
explicit result writing.

### P7.2 Result schema and I/O

- add schema version;
- serialize model, field, time, solver, backend, and scaling metadata;
- atomic result/checkpoint writes;
- validated resume;
- migration error for unknown schema.

### P7.3 Optimization

Before changes, add one trusted objective and gradient/reference test for each
supported algorithm. Introduce common Objective, Evaluator, OptimizationResult,
and constraint interfaces only after behavior is characterized.

### P7.4 Spectroscopy

Before splitting the 898-line module, characterize:

- thermal state;
- response function;
- FFT sign/frequency convention;
- broadening;
- absorption/PFID/emission observables;
- normalization or sum rules.

Then split by scientific responsibility.

### Phase 7 acceptance

- runner modules are individually testable;
- failed cases and resume behavior are deterministic;
- result schema is versioned;
- optimization and spectroscopy no longer have zero/near-zero critical
  coverage;
- no broad catch suppresses physics errors.

## 11. Phase 8 — public API and release

Tasks:

- decide O-008 and reduce root exports;
- rewrite README and Japanese README against the actual API;
- execute documentation code snippets;
- update every supported example — completed early under D-044 with three
  typed smoke examples;
- move unsupported examples to an explicit archive or delete them — completed
  early under D-044 for the former v0.2 script set;
- update version to 0.3.0;
- produce migration notes stating that backward compatibility is intentionally
  broken;
- build sdist and wheel;
- test clean installation;
- run all quality, physics, and benchmark gates;
- tag only after the refactor branch is clean.

### Phase 8 acceptance

- documented examples execute;
- README contains no removed name;
- public API inventory matches exports;
- wheel smoke test passes;
- known limitations and backend matrix are published;
- changelog and result schema version are current.

## 12. Suggested immediate commit sequence

The first work after these documents should use approximately this sequence:

1. `test: establish two-level physics references`
2. `test: establish vibrational and Morse references`
3. `test: establish linear-molecule and dipole references`
4. `test: establish solver convergence and invariant suite`
5. `test: migrate uncollected root validation scripts`
6. `chore: remove generated repository artifacts`
7. `chore: classify and remove verified legacy files`
8. `style: apply repository-wide Ruff formatting`
9. `fix: resolve repository-wide Ruff findings`
10. `ci: enforce truthful quality and physics gates`

Actual boundaries may be smaller. Never combine steps 1–4 with steps 8–9.

## 13. Mandatory checks per change class

| Change class | Required checks |
|---|---|
| Documentation only | link/path check, `git diff --check` |
| Formatting only | full pytest, Ruff format/lint |
| File move | import smoke, focused tests, full pytest, wheel build |
| Public API | contract tests, examples, full pytest, docs |
| Physics formula | analytic/golden test, focused convergence, full pytest |
| Backend | capability test, parity test, unavailable-backend error |
| Serialization | round trip, schema mismatch, atomic write/resume |
| Performance kernel | correctness, convergence, benchmark, memory |

## 14. Stop and ask conditions

Codex must stop and ask the user when:

- a formula or sign is not covered by an accepted decision;
- a “magic number” cannot be derived from documented parameters;
- a legacy and new implementation disagree scientifically;
- a required reference value cannot be obtained from an analytic relation or
  existing trusted test;
- normalization or clipping would alter user data;
- a directory move changes scientific ownership in a way not covered by the
  target architecture;
- a backend implementation would require a materially different numeric type;
- optimization or spectroscopy behavior has no trusted reference;
- a destructive cleanup contains potentially unique scientific data.

## 15. Phase completion record

When completing a phase, append or update a record with:

~~~markdown
### Phase N completion

- Commit(s):
- Date:
- Tests:
- Coverage:
- Lint/format:
- Performance:
- Documentation:
- Accepted deviations:
- Remaining open decisions:
~~~

Do not mark a phase complete while required work is merely deferred without an
open decision or follow-up task.
