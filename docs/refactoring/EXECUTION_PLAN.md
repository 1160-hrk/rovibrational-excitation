# Executable refactoring plan

Last updated: 2026-08-11
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
result contracts. P2.1 is the first implementation checkpoint; legacy kernels
remain unchanged while typed boundaries are introduced.


### P2.1 Introduce TimeGrid

- Move validated time-grid semantics into a frozen type.
- Keep `FIELD_INTERVALS_PER_PROPAGATION_STEP = 2` as a named invariant.
- Construct `ElectricField` from the TimeGrid rather than reconstructing time
  separately.
- Add dimensional and nondimensional time tests.

Do not remove old calls until all workflows use TimeGrid.

### P2.2 Introduce explicit state kinds

Add distinct input types:

- `PureState`;
- `IncoherentEnsemble`;
- `DensityState`.

Remove final public reliance on list/square-array inference. Temporary adapters
may live at the old boundary during this phase.

### P2.3 Introduce ExecutionPolicy and capabilities

One policy controls backend and storage for model construction and propagation.
A capability registry rejects unsupported combinations before matrix
allocation.

Add parameterized tests for every advertised combination.

### P2.4 Introduce PropagationProblem and PropagationOptions

Replace solver `**kwargs` with typed fields. Required choices are explicit;
unsupported combinations fail during construction or preflight.

The field TimeGrid is the only timestep source. No solver-level `dt` override.

### P2.5 Introduce PropagationResult

All high-level solvers return a result object containing time, state, state
kind, trajectory flag, backend, and metadata.

Temporary old-array adapters are tested and then removed because compatibility
is not required.

### Phase 2 acceptance

- No high-level propagator public method accepts unrestricted `**kwargs`.
- Return type no longer changes according to booleans.
- Backend/storage/algorithm errors occur before expensive work.
- Current physics and performance baselines pass.
- API inventory and architecture documents are updated.

## 6. Phase 3 — target package migration

Goal: establish dependency direction using mechanical movement before redesign.

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

### Phase 3 acceptance

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
scaling surface was reduced to strict transformation, scale metadata, exact conversions, and
neutral reporting. `analysis.py`, `strategies.py`, `impl.py`, automatic
timestep wrappers, heuristic strength verification, and demo factories were
removed. Raw-array units and object coupling semantics are now required.


The 2026-08-09 explicit-fallback audit also rejects removed and unknown solver
options and prevents dipole CuPy requests from becoming NumPy arrays. Remaining
P1/P2 findings and physics-facing default decisions are in FALLBACK_AUDIT.md.

Remaining Phase 4 work:

- move the implemented policy into the target dynamics/scaling package;
- attach scale provenance to the unified PropagationResult rather than
  recomputing it in simulation/runner.py;
- finish explicit quantity types and property-style unit round trips;
- define an error-controlled adaptive integrator separately, if wanted.
Tasks:

- define explicit quantity/unit types or validated value-plus-unit dataclasses;
- keep pure conversion functions in `core/units`;
- move complete-problem scaling to `dynamics/scaling`;
- move the consolidated converter, scales, reporting, and conversion helpers
  without reintroducing competing policy;
- serialize scales in results;
- add property-style round-trip tests.

Acceptance:

- numerical kernels contain no unit conversion calls;
- no mutable object parameter changes during conversion;
- dimensional and nondimensional reference observables/time agree;
- one documented internal dimensional unit exists per quantity;
- all current supported input units have round-trip tests.

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
- update every supported example;
- move unsupported examples to an explicit archive or delete them;
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
