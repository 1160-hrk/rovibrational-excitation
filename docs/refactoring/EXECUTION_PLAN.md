# Executable refactoring plan

Last updated: 2026-10-01
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

The archived
`examples/archives/v0_2/scripts/absorbance_from_density_matrix.py` is retained
as non-executable migration evidence. Its legacy PFID, Doppler, and response
formulas have now been independently characterized by P7.4-a1/a2 against
`AbsorbanceCalculator`; its duplicated approximate constants and hard-coded
thresholds remain historical evidence, not accepted defaults.

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
  the D-027 legacy layout. GRAPE and the former Krotov route (now explicitly
  `legacy_batch_overlap` under D-106) use canonical `TimeGrid`, explicit field
  spacing, full internal trajectories, output-only thinning, and the D-028
  backward direction under D-029. Independent objective and gradient
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
the solver boundary. Under D-029, GRAPE and the route now named `legacy_batch_overlap` construct
the canonical grid from `field_dt_fs`; standard Krotov instead uses the D-106
interval grid and `control_dt_fs`; their optimization calculations always use the full
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

P4.3-k implements D-053 on 2026-09-02. A new production
`models.symmetric_top` package owns the signed `|v,J,K,M>` basis, independent
integer Wigner-3j rotation primitive, parallel-band Cartesian dipole, and model
builder. CH3F ortho/para filtering is applied before basis allocation; all
constants remain explicit quantities. Normal NumPy dense/CSR RK4 and strict
scaled propagation are supported. CuPy, split operator, coherent all-isomer
construction, and optimization raise explicitly. Independent physics tests and
the complete suite pass 1016 tests with 10 optional-GPU skips; measured branch
coverage remains 73%.

P4.3-l implements D-054 on 2026-09-02. Optimization now reuses frozen
LinMol, VibLadder, and TwoLevel parameter schemas plus the same model-owned
basis/Hamiltonian/dipole builders as normal simulation. The strict optimization
projection requires value/unit pairs, rejects legacy and inapplicable keys, and
requires exact quantum-number tuples without silently adding or dropping M.
LinMol accepts only `representation=m_resolved`; SymTop and M-incoherent
optimization remain explicit unsupported routes. Basis ordering, H0, SI
dipoles, all optimizer time/index contracts, and the stored Krotov result are
characterized. No optimizer numerical kernel changes. The complete suite passes
1025 tests with 10 optional-GPU skips, branch coverage is 74%, and strict mypy
covers 32 named modules.

P4.3-m implements D-055 on 2026-09-03. Thirteen incomplete v0.2 optimization
YAML files move with history to an explicit archive. No physical value is
invented and no source or numerical behavior changes.

P4.3-n implements D-056 on 2026-09-03. Optimization documents now have closed
root, algorithm, plot, output, time, and spectral-constraint key sets.
`control_axes` is required; GRAPE explicitly accepts only `xy`; unsupported or
misspelled values never fall back. Krotov initial-field branches are validated
before model construction, including exact sampled-grid length. Required
`output.dir` and `plot.enabled` are honored with explicit API/CLI precedence,
and top-level requested plotting failures raise. Thirteen incomplete v0.2 YAML
files are archived without invented dipoles; three current-schema configs with
explicit dipole units and axes remain active. Optimizer kernels and all local
time/index contracts are unchanged. The complete suite passes 1045 tests with
10 optional-GPU skips, and strict mypy covers 34 named modules.

P4.3-o implements D-057 on 2026-09-03. Optimizer values now use exact boolean,
integer, finite-real, enum, and distinct-axis contracts without Python coercion
or fallback. Local evaluates only explicit `target` or `weights` modes, uses the
separate `weight_reverse` flag, resolves eigenvalues only for requested
lookahead, and surfaces weight/running-cost failures. Spectral constraints are
fully value-validated; finite nonnegative alpha uses exact `1+alpha` division
without a repair floor. Existing Local grid/index/endpoint references, Krotov
fidelity, and current YAML smoke runs remain unchanged. The complete suite
passes 1146 tests with 10 optional-GPU skips; strict mypy covers 35 named
modules.

P4.3-p implements D-058 on 2026-09-07. Local `gain` and `gain_units` are
required, accept four exact field-squared-femtosecond labels, and convert to
canonical `(V/m)^2 fs` before the unchanged target/weights update equations.
The active example uses `1000 (GV/m)^2 fs`, exactly preserving its former
`1e21` canonical multiplier. Positive validation removes the unreachable
reciprocal repair. The former `running_cost` output becomes the explicitly
diagnostic `field_fluence_proxy`; canonical gain, vector field maximum/RMS,
segment clipping fraction, and separate per-axis gain-dipole reference scales
are reported without changing any control value, segment, index, endpoint, or
RK4 call. `c_abs_min`, `drive_abs_min`, and `shape_floor` remain Class D.
The complete CPU suite passes 1168 tests with 10 optional-GPU skips.

P4.3-q implements D-059 on 2026-09-08. Local initialization is now an explicit
required `seed_field` or `none` branch. `seed_field` requires a positive direct
electric-field amplitude value/unit pair and positive maximum segment count;
the active `1000 V/m`, five-segment configuration preserves the existing seed
trigger, signs, clip order, field arrays, shared endpoints, and odd RK4 prefix.
`none` injects no field and uses the existing mode-specific predicate as a
first-segment preflight: a weights response below `drive_abs_min` or target
overlap below `c_abs_min` raises before propagation with measured diagnostics.
It never falls back to a seed. Initialization method and actual seeded segment
count are reported. No new meaning is assigned to the three Class-D thresholds.

Phase 4 is complete for every currently decided contract. The following are
explicitly deferred extensions rather than blockers for Phase 5:

- defer the remaining Class-D optimizer thresholds, penalties, and tolerances
  until independent references and user-defined dimensions exist;
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

P5.1-b implements D-060 on 2026-09-10. The validated `rk4_lvne` and
`rk4_lvne_traj` wrappers retain density/field/step/stride validation, canonical
dtype and contiguous-array preparation, legacy output shapes, and final-state
unwrapping. They now delegate to a dedicated prevalidated
`liouville_numpy.py` NumPy/Numba kernel. The numerical body, field-stage
indices, interaction sign, commutator, operation order, output allocation,
stride writes, Numba signature, and fastmath setting moved unchanged. A frozen
complex reference agrees within `1e-17`, caller arrays remain unchanged, 71
focused Liouville tests pass with 6 optional skips, and the complete suite
passes 1184 tests with 10 optional-GPU skips. Strict mypy covers 37 named
modules.

No allocation or speed claim is made: inherited per-step stage intermediates
remain. An allocation-stable replacement requires a separate measured unit.
CuPy dense separation and real-GPU validation remain before P5.1 can be called
complete.

P5.1-c implements D-061 on 2026-09-13. The dense Liouville kernel reuses the
right-endpoint Hamiltonian as the next step's exactly identical left-endpoint
Hamiltonian. Field indices, the `-mu E` sign, both Cartesian components,
commutator expressions, RK4 stages, stride, output shapes, and `fastmath`
remain unchanged. Direct comparison with the retained pre-change Numba loop is
bitwise exact for multiple dimensions and both output modes.

`benchmarks/liouville-endpoint-reuse-v0.3.json` records single-thread median
speedups of 1.019x to 1.044x for dimensions 4 to 64, with zero final-state
difference. It also records the analytical source-level Hamiltonian-array
traffic removed, explicitly not process RSS. A full output-buffer experiment
was slower for representative nontrivial dimensions and changed sub-ulp
results, so it was rejected rather than weakening the exact contract.
Remaining stage and commutator intermediates are intentionally preserved.
The complete suite passes 1188 tests with 10 optional-GPU skips; branch
coverage remains 75%.

Separate:

- validation and preparation;
- NumPy dense kernel;
- NumPy sparse kernel;
- CuPy dense kernel;
- Liouville NumPy kernel.

Unify field-stage indexing, interaction sign, stride semantics, and result
construction through tests, not through runtime abstraction inside hot loops.

### P5.2 Split operator

CPU status: accepted under D-062 on 2026-09-13. All rows below execute on
NumPy, including exact CSR-to-dense-spectral parity. The benchmark now
separates public end-to-end, spectral setup, and prepared inner propagation;
prepared and public final states are exactly equal. CuPy source paths are
device-native; numerical, transfer, and timing evidence remains pending a real
CUDA job.

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

Policy status: decided by D-026/D-039 and CPU/device-like boundary behavior
accepted by D-062. `PropagationResult` retains an existing device state and
converts only through explicit `to_numpy()`. P5.5-a/D-144 makes CuPy RK4 return
device-native arrays, and P5.5-b/D-145 does the same for all three split modes.
Real-GPU transfer and numerical evidence remains before acceptance.

### Phase 5 acceptance

- every advertised capability has an executing test;
- CPU/GPU parity is tested where infrastructure permits;
- no silent fallback;
- physics baselines pass;
- median performance regression is below 10% or explicitly approved;
- memory use is documented for trajectories.

P5.4-a records the row-by-row disposition in
`PHASE5_ACCEPTANCE_AUDIT.md`. D-144 and D-145 complete source-level
device-native ownership. Phase 5 remains in progress only for an accepted
real-CUDA parity, backend, transfer, and timing artifact. D-071 makes this a
mandatory v0.3 release gate. It does not block independent Phase 6 CPU model
consolidation, but v0.3.0 cannot be tagged until the implementation passes on a
GPU-equipped runner.

P5.5-a/D-144 replaces the user-approved incorrect fused CuPy RK4 kernel with
a separate dense CuPy implementation of the CPU graph. It uses `H0 - mu E`,
the exact left/mid/mid/right stages, the standard RK4 update, trajectory/stride,
and per-step renormalization, and returns CuPy arrays without `.get()` or
`cp.asnumpy`. A NumPy-backed CuPy double verifies the graph on CPU; three real
GPU cases are collected for backend identity, shape, dtype, norm, and parity.
The full CPU suite has 1506 passes and 13 optional-GPU skips (1519 collected).
No CUDA speed claim is made. Split device residency and all real-GPU evidence
remain open.

P5.5-b/D-145 moves the unchanged static Cartesian, rotating Cartesian, and
helicity-projected CuPy split calculations to a separate device-native owner.
The shared tolerance factor, Hermiticity/covariance checks, midpoint indices,
zero-amplitude branch, phases, stride, and renormalization rules are preserved.
CPU-backed graph comparisons pass for all modes; three real-GPU identity/parity
cases are collected. The full suite has 1514 passes and 16 optional-GPU skips
(1530 collected), and strict mypy covers 84 modules. Source-level residency is
complete; actual CUDA transfer/timing evidence remains mandatory.

P5.5-c/D-146 adds a hard-failing real-CUDA evidence recorder and a manual
pre-tag workflow without changing either propagator. Its schema requires RK4
final/trajectory and all three split modes, checks host-input and device-input
results against NumPy, and records backend, dtype, shape, error, norm,
synchronized timing, transfer volume, software, hardware, and source identity.
Timing has no pass threshold. The release workflow repeats and attaches the
accepted JSON; CUDA absence writes a diagnostic artifact and fails. The local
suite has 1521 passes and 16 optional-GPU skips (1537 collected). Phase 5 stays
open until actual hardware produces an accepted artifact.

## 9. Phase 6 — model consolidation

Goal: co-locate model formulas, parameters, basis, Hamiltonian, dipole, and
coupling.

Order:

1. TwoLevel;
2. VibLadder;
3. LinMol;
4. SymTop only after O-005 is resolved.

### P6.1 TwoLevel

P6.1-a completed the pre-move characterization on 2026-09-13 under D-063.
Eight executable cases freeze the parameter projection, basis order and state
mapping, Hamiltonian formula and current storage conversion, coherent state,
scalar-x capability, exact Cartesian dipoles, dense/CSR parity, cache reuse,
and stateless-builder parity. No source implementation changed.

P6.1-b resolves O-013 under D-064 in a separate approved numerical-behavior
commit. Reduced Planck's constant is derived once from exact Planck's constant,
all runtime aliases consume it, and all Hamiltonian conversion methods use the
central converter. The typed TwoLevel example changes by at most `1.142e-13`
in population; dense/CSR and dimensional/nondimensional checks continue to
pass. P6.1-c may now move ownership while preserving these new references.

P6.1-c completes the structural move under D-065. `TwoLevelBasis`,
`TwoLevelDipoleMatrix`, the stateless dipole builder, and model construction now
share `models/two_level/`; the three superseded owners are removed. Active
imports use the model package, while the transitional generic dipole factory is
addressed explicitly as `dipole.factory`. The D-063/D-064 numerical references
remain exact. P6.1-d may next move `TwoLevelParameters` out of the shared schema
module and settle the transitional factory without combining that interface
work with this file move.

P6.1-d completes TwoLevel consolidation under D-066. The frozen schema moves
to `models/two_level/parameters.py`; unchanged shared schema validators move to
private `models/_parameter_validation.py`; the unused mapping and stateless-dipole
wrappers are removed; and the transitional generic dipole factory no longer
has a TwoLevel branch or model-layer dependency. Its remaining vibrational
route requires `potential_type` in the signature. All D-063/D-064 physics and
numerical references pass unchanged. P6.2 begins VibLadder with a separate
pre-move characterization unit.

### P6.2 VibLadder

P6.2-a completes the pre-move characterization on 2026-09-14 under D-067.
Nine new contracts freeze the frozen-parameter projection, caller-unit
provenance and canonical conversion, basis/state order, anharmonic
Hamiltonian, coherent state, scalar-z capability, exact harmonic dipoles,
dense/CSR storage, cache identity, and parity among every current construction
path. The existing independent physics suite remains authoritative for the
Morse formula, instance-local derived N, maximum bound level, zero-shift
rejection, and propagation parity. No source implementation changed.

P6.2-b should next move `VibLadderBasis`, `VibLadderDipoleMatrix`, and the
production builders into `models/vib_ladder/` as one structural ownership
unit. Active imports move in the same commit and the superseded owners are
removed without compatibility shims. The frozen schema and generic factory
cleanup remain a separate P6.2-c interface unit. The shared `dipole/vib`
harmonic and Morse functions remain in place during P6.2-b because LinMol and
legacy SymTop still consume them; their final owner is decided from all actual
Phase 6 consumers rather than inferred during a VibLadder file move.

P6.2-b completes the structural ownership move on 2026-09-15 under D-068.
The basis, stateful dipole, transitional stateless builder, and model builders
now share `models/vib_ladder/`; all active imports use that path and the three
former owners are absent. The moved bodies change only the imports required by
their new location. The generic dipole factory drops VibLadder rather than
adding a forbidden reverse dependency on the model package. Every D-067
numerical and physical reference remains unchanged.

P6.2-c completes VibLadder consolidation on 2026-09-15 under D-069.
`VibLadderParameters` now belongs to `models/vib_ladder/parameters.py` with
unchanged validation and conversion. Caller audit found no production use of
the redundant mapping builder or stateless dipole wrapper, so both are removed.
The package exposes only its schema, basis, stateful dipole, and typed builders.
Shared `dipole/vib` transition-element functions stay in place until the
remaining LinMol and legacy SymTop consumers are consolidated. All D-067
references pass unchanged and P6.2 is complete.

### P6.3 LinMol

P6.3-a completes pre-move characterization on 2026-09-15 under D-070. Nine new
cases freeze current ownership, parameter/unit projection, signed
`|v,J,M>` order, Hamiltonian construction, Cartesian coupling, coherent
basis-index state construction, stateful/stateless dipole parity, cache
identity, dense/CSR parity, and mapping/typed builder parity. The 29 existing
independent physics cases retain authority for selection rules, M-incoherent
averaging, Morse behavior, and propagation. No source implementation changes.

P6.3-b completes the structural ownership move on 2026-09-16 under D-074. The
basis, stateful/stateless dipole implementation, and mapping/typed model
builders now live in `models/linear_molecule/`; all former owner paths are
removed without shims. The generic dipole factory drops LinMol rather than
adding a reverse dependency on the model package. Formulae, signed basis order,
selection rules, state indices, M averaging, dense/CSR behavior, and all D-070
references remain unchanged.

P6.3-c completes LinMol consolidation on 2026-09-16 under D-075. The frozen
schema moves unchanged into the model package. The unused mapping wrapper is
removed, and the dipole implementation function becomes the private
`_build_mu` kernel used by the stateful class. The package exports only its
schema, basis, stateful dipole, and typed builders. The production mapping
entry retains exact parity with typed construction, strict mypy covers 41
modules, and all D-070 references remain unchanged. P6.3 is complete.
`simulation/m_average.py` remains a workflow owner outside the model package.

### P6.4 SymTop and shared dipole debt

P6.4-a should first characterize and audit the remaining experimental
`core/basis/symtop.py` and `dipole/symtop/` callers against the D-053 production
`models/symmetric_top/` owner. No legacy formula is merged into production by
inference. If the two implementations disagree physically, present the exact
formulae and outputs to the user before changing either calculation.

After that audit, remove unused legacy owners or migrate only independently
protected behavior, then resolve `dipole.factory`, `dipole.base`, and shared
rotational/vibrational kernel ownership from their actual remaining consumers.

P6.4-a completes on 2026-09-16 under D-076. Three guards record that legacy
and production owners are distinct, their basis/anharmonic conventions are not
equivalent, and their transverse Cartesian phases differ. The caller audit
finds no production consumer of the legacy skeleton; direct execution also
shows both legacy dense and CSR dipole construction are broken. The D-053
production implementation and its independent references remain unchanged;
the complete suite passes 1220 tests with 10 optional-GPU skips.

P6.4-b removes the legacy basis/dipole/factory and legacy-only `jmk` helper
on 2026-09-16 under D-077. The direct tests and package/docs exports now
point to production `models.symmetric_top`; no legacy formula was migrated.
The D-053 physics and propagation matrix remains the acceptance gate.

P6.4-c should audit the actual consumers of shared `dipole.base`,
`dipole.rot.jm`, and `dipole.vib` before assigning their final owner. Preserve
each transition element and all model arrays; do not merge distinct Morse or
rotational formulas by inference.

P6.4-c completes the consumer audit and the unambiguous rotational move on
2026-09-16 under D-078. The analytic `J,M` kernel now belongs to LinMol;
its independent Wigner reference is test-only. The unused `J`-only helper and
old rotation package are removed. Full tests pass 1218 cases with 10
optional-GPU skips. `dipole.base` stays as a transitional shared class until
an operator protocol and concrete cache/unit/persistence owner are separately
characterized. P6.4-d should first freeze both consumers of `dipole.vib`,
including Morse boundaries and CPU/GPU modes, then move the exact shared
functions to a neutral `models/vibration` owner. SymTop's independently
referenced functions must not be merged by inference.

P6.4-d completes that exact-file shared-vibration move on 2026-09-16 under
D-079. Both model consumers import the same functions, CPU physics and Morse
boundary references pass, and SymTop is unchanged. The full suite has 1219
passes and 10 optional-GPU skips; CUDA remains unverified. P6.4-e should
characterize and split `dipole.base`: put only the minimum operator protocol
below dynamics, retain cache/unit/persistence implementation with the models,
and eliminate the two documented reverse dependencies without changing
runtime fallback or conversion behavior.

P6.4-e completes on 2026-09-16 under D-080. A separate pre-move test commit
freezes conversion, cache identity, SI views, and the legacy fallback. The
concrete base moves unchanged into `models.dipole_base`; a type-only
`core.dipole.DipoleOperator` expresses the four required accessors. Both
`dynamics -> dipole.base` reverse imports are eliminated. CPU tests pass 1223
cases with 10 optional-GPU skips; branch coverage remains 77%, strict mypy
covers 42 modules, and active examples plus wheel checks pass. P6.4-f should
audit and remove the now code-empty `dipole` package shell, then evaluate the
Phase 6 acceptance list without broad numerical cleanup. Do not change the
legacy `get_dipole_component_SI` fallback in that structural unit.

P6.4-f removes the now code-empty `dipole` package on 2026-09-16 under
D-081, including its obsolete README and root convenience import. The
historic archived examples are unchanged. Full tests pass 1224 cases with
10 optional-GPU skips, coverage stays 77%, and the wheel has no old package
files. Phase 6 is still open: `models/__init__.py` and `models/factory.py`
import `dynamics.problem`, exactly matching the two recorded upper-layer
dependencies. P6.5 should characterize `CouplingSpec`/`SystemModel` ownership
and remove those edges without altering coupling semantics, model component
values, or projection behavior. Do not fold this into a numerical change.

P6.5-a freezes `ModelComponents.to_system_model()` identity, ordered Cartesian
and scalar-axis projection, model dimension, and defensive metadata snapshot
before moving contract ownership. No source implementation changes; the full
suite passes 1226 cases with 10 optional-GPU skips. P6.5-b may move the
unchanged coupling/model contracts to a lower neutral owner, retaining the
existing dynamics access path while eliminating exactly the two reverse imports.

P6.5-b/D-082 moves the byte-identical `Axis`, `CouplingMode`, `CouplingSpec`,
and `SystemModel` definitions to `core.model`. Dynamics re-exports the exact
objects and models imports them directly; no model-to-upper-layer imports
remain. Full tests pass 1227 cases with 10 optional-GPU skips, coverage is
77%, strict mypy covers 43 modules, and active example/build/isolated-wheel
checks pass. P6.6 should audit the Phase 6 acceptance matrix, especially
supported dense/CSR references and the separately unverified real-CUDA path.

P6.6-a freezes the simulation-owned `FixedMLinMolBasis` before correcting its
owner. Five direct cases record fixed-M basis order, index mapping, M array,
Hamiltonian diagonal, and invalid-M rejection. No source implementation
changes; the full suite passes 1232 cases with 10 optional-GPU skips. P6.6-b
may move only that class to `models.linear_molecule`; block creation, weights,
propagation, and incoherent reduction remain in `simulation.m_average`.

P6.6-b/D-083 moves the unchanged fixed-M basis class to
`models.linear_molecule.basis`. The D-017 block workflow remains in simulation
with identical construction arguments, weights, indices, and reduction. The
acceptance audit passes all five criteria below on CPU: model formulas have one
owner; simulation defines no model class/formula; duplicate factories are
gone; frozen schemas validate required inputs; and all supported NumPy
dense/CSR references pass. Phase 6 is complete. Optional CuPy skips are not
acceptance evidence and remain under the separate Phase 5 real-CUDA gate.

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

Accepted on 2026-09-17 under D-083. The D-017 fixed-M averaging algorithm is a
simulation workflow, not a second LinMol model owner: it imports the model-owned
basis and operators and owns only block orchestration and observable reduction.

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

P7.1-a/D-084 begins the decomposition on 2026-09-17. The byte-identical
generated-field sampling body moves to `simulation.field_preparation`; the
runner imports that exact function. Existing direct unit/frequency/envelope,
modulation, scalar/Cartesian, M-average, and helicity references pass. Private
polarization-decoder tests now use its existing `io` owner. The full suite has
1233 passes and 10 optional-GPU skips, and strict mypy covers 44 modules.
Next isolate one-case result assembly/writing without changing the unversioned
schema, overwrite behavior, or the current single-write M-average NPZ path.

P7.1-b freezes one-case persistence before extraction on 2026-09-20. Direct
normal and D-017 M-average cases assert one compressed-NPZ write, exact key
sets, returned/stored population identity, absence of a fictitious aggregate
M-average wavefunction, normalized M weights, and unchanged caller parameters
in JSON. No source implementation changes; the full suite passes 1234 cases
with 10 optional-GPU skips. The next unit may move payload assembly/writing
without changing any key, array, path, count, encoding, or overwrite behavior.

P7.1-c/D-085 extracts that exact payload assembly/writing to
`simulation.result_persistence`. Runner retains only the `save` decision and
passes computed results to the new owner. Normal and M-average schemas, one
write per case, JSON and conditional regime file behavior remain fixed. The
full suite passes 1235 cases with 10 optional-GPU skips, branch coverage stays
77%, strict mypy covers 45 modules, and active examples pass. P7.1-d should
separate one-case preparation from propagation without changing call order.

P7.1-d freezes that order before extraction on 2026-09-20. A direct normal-case
contract records validation, immutable case construction, propagation-time
nondimensionalization, propagation completion, post-propagation scale analysis,
and the explicit `to_numpy()` boundary in their current order. No source
implementation changes; the full suite passes 1236 cases with 10 optional-GPU
skips. P7.1-e may now separate preparation and propagation under this guard.

P7.1-e/D-086 moves the guarded preparation and propagation stages to
`simulation.execution`. Preparation still validates before sampling and freezes
one `SimulationCase`; propagation consumes only that case and returns either
the unchanged D-017 M-average result or an internal typed wavefunction bundle.
Runner retains save policy, persistence dispatch, and the population return.
The full suite passes 1237 cases with 10 optional-GPU skips, branch coverage is
77%, strict mypy covers 46 modules, and all three active examples pass. Next
characterize retry/error and batch/checkpoint orchestration before extraction.

P7.1-f freezes retry, failure-file, and checkpoint cadence behavior before
batch-service extraction on 2026-09-21. `OSError` alone receives the existing
one- and two-second retries; other exceptions fail on the first attempt. The
returned traceback is written verbatim before caller parameters, including
the existing JSON-safe complex representation. Five cases in batches of two
save checkpoints after batch two and the final third batch. No source behavior
changes; the full suite passes 1240 cases with 10 optional-GPU skips. P7.1-g
may now extract safe-case execution under these guards.

P7.1-g/D-087 moves the guarded retry and failure-file body to
`simulation.safe_execution`. Its `CaseRunOutcome` remains tuple-compatible,
and runner passes the existing `_run_one` callable through a top-level wrapper
that remains safe for multiprocessing. OSError-only retry counts/backoff,
immediate other failures, traceback/parameter file content, prints, and batch
checkpoint cadence remain fixed. The full suite passes 1241 cases with 10
optional-GPU skips, branch coverage is 77%, and strict mypy covers 47 modules.
P7.1-h freezes the two existing summary paths and resume case reconstruction on
2026-09-21. A normal batch summarizes the returned population even when a
different result file exists. Resume expands the saved Python parameters,
rebuilds the same sweep directories, skips checkpoint-completed cases, and
updates all-case summary rows from saved NPZ files, not the newly returned
population. The test also fixes the current failed-case and completed-hash
checkpoint totals. No source behavior changes; the full suite passes 1243
cases with 10 optional-GPU skips. P7.1-i may now extract batch management
under these guards without changing calculation or persistence policy.

P7.1-i/D-088 moves the duplicated normal/resume batch loops to
`simulation.batch.execute_case_batches` on 2026-09-21. Runner still owns
case construction, resume validation, process-count choice, the
multiprocessing-safe case callable, and the two intentionally different
summary paths. The batch owner preserves case order, a fresh process pool
per batch, progress labels, every-second-or-final checkpoint cadence,
completed/failed hashing, and the original result projections. The new
parallel-path test checks two pools for three cases split 2+1. The full suite
passes 1244 cases with 10 optional-GPU skips; branch coverage is 78%, and
strict mypy covers 48 modules. Next isolate reporting and case-path
reconstruction under the same no-behavior-change policy.

P7.1-j freezes case-path construction before extraction on 2026-09-21.
Normal sweep cases retain insertion-ordered Cartesian expansion and
`key_label` nested paths; saved directories are created before execution.
A saved dry run still creates those directories but executes no case and
creates no checkpoint or summary. Resume reconstruction was already fixed
by P7.1-h. No source behavior changes; 1246 tests pass with 10 optional-GPU
skips. P7.1-k may give both routes one case-path owner without changing
these side effects.

P7.1-k/D-089 gives normal and resumed case-path materialization one owner,
`simulation.case_paths`, on 2026-09-21. The existing pure sweep expansion
and label formatting remain in `simulation.sweep`; the new owner adds only
the unchanged `save` flag, nested result path, eager mkdir, and `outdir`
field. Runner still decides the results root, dry-run behavior, and resume
filtering. The full suite passes 1246 cases with 10 optional-GPU skips,
branch coverage stays 78%, and strict mypy covers 49 modules. Next freeze
and extract normal-run reporting without conflating it with resume's
file-backed summary.

P7.1-l freezes normal-run reporting before extraction on 2026-09-21.
Returned scalar, vector, and time-series populations project to the current
`pop_i` CSV columns; the last time row is used for multi-dimensional
results, and missing columns remain missing. Failure rows retain their
errors, previews show only the first five of seven failures with a remainder
count, and an all-failed run writes no `summary_success.csv`. No source
behavior changes; 1248 tests pass with 10 optional-GPU skips. P7.1-m may
move only this normal reporting body, leaving resume's file-backed summary
in its existing IO owner.

P7.1-m/D-090 moves only normal-run completion reporting and in-memory CSV
generation to `simulation.reporting` on 2026-09-21. The runner still owns
the save decision and result projection; resume still calls
`io.storage.update_summary` and reads existing NPZ files. The exact
first-five failure preview, final-population projection, status/error rows,
and conditional success-only CSV remain unchanged. The full suite passes
1248 cases with 10 optional-GPU skips; branch coverage stays 78%, and
strict mypy covers 50 modules. Next characterize resume validation and
reporting before deciding whether their application-level coordination
should be separated from runner; persistence schema redesign remains P7.2.

P7.1-n freezes resume entry/report ordering before ownership extraction on
2026-09-21. An unreadable checkpoint errors before parameter loading. A valid
checkpoint with missing `params.py` prints prior progress, then raises
`FileNotFoundError`. If all cases are complete, resume creates the existing
case directories, returns an empty list, and does not rewrite summary.
Otherwise the new-completion message precedes the file-backed summary call.
No source behavior changes; 1252 tests pass with 10 optional-GPU skips.
P7.1-o may isolate resume preparation/reporting while preserving these
observable details and deferring validation-policy changes to P7.2.

P7.1-o/D-091 moves guarded resume preparation to `simulation.resume` and
the post-batch completion report to `simulation.reporting` on 2026-09-21.
The runner still decides process count, executes remaining cases, handles
all-complete early return, and supplies the existing checkpoint/parameter
loader and file-backed summary callback. Validation/error order, eager case
paths, checkpoint contents, return values, and summary timing are unchanged.
The full suite passes 1252 cases with 10 optional-GPU skips; branch coverage
stays 78%, and strict mypy covers 51 modules. Next perform a P7.1 acceptance
audit. In particular, `resume_run` does not currently validate
`checkpoint_interval` like normal batch execution; address this in a
separate tested validation-policy unit, not this ownership commit.

P7.1-p/D-092 closes that input-boundary gap on 2026-09-21. Both normal and
resume entry points now require an actual positive Python integer
`checkpoint_interval` before file I/O or case execution. Zero, negative,
non-integer values, and booleans raise the same precise `ValueError`; valid
integer batch cadence, checkpoint contents, and calculations are unchanged.
Five parameterized tests first failed on the old resume route or boolean
normal route, then passed after the shared validator was added to
`simulation.batch`. The full suite passes 1257 cases with 10 optional-GPU
skips, branch coverage remains 78%, and strict mypy covers 51 modules.
The P7.1 acceptance audit is next.

P7.1-q removes two verified-unused private artifacts during the acceptance
cleanup: `runner._parallel_run_safe` had no callers, and the resume
`base_dict.get("description", "resumed_run")` expression discarded its
result. The existing `run_all`, CLI, case executor, and multiprocessing
path remain untouched. All 1257 CPU tests pass with 10 optional-GPU skips,
78% branch coverage, and 51 strict-mypy modules. The acceptance audit is
still required before assigning a P7.1 completion status.

P7.1-r/D-093 completes the simulation-runner acceptance audit on
2026-09-21. `PHASE7_RUNNER_ACCEPTANCE_AUDIT.md` checks each application
owner, exact normal/resume failure and summary behavior, import/architecture
wiring, the full 1258-pass CPU suite with 10 optional-GPU skips, 78% branch
coverage, 51 strict-mypy modules, all active examples, repository-wide Ruff,
sdist/wheel construction, Twine, and installed-wheel import outside the
workspace. P7.1 is complete. This does **not** complete Phase 7: schema
versioning/atomic persistence, independent optimization/spectroscopy
references and decomposition, and the Phase 5 real-CUDA gate remain open.

P7.1-s/D-094 records `0.3.0.dev1` as a development-version checkpoint on
2026-09-21. Only package metadata, its contract test, and release/planning
documentation change; numerical logic does not. No tag or publication is
implied, and neither Phase 7 nor final `0.3.0` is complete. The full suite
passes 1259 tests with 10 optional-GPU skips and 78% branch coverage; wheel
build, Twine validation, and installed-wheel import pass. P7.2 is next,
followed by the remaining Phase 7, Phase 8, and real-CUDA acceptance gates.

### P7.2 Result schema and I/O

P7.2-a starts on 2026-09-21 with
`PHASE7_PERSISTENCE_BASELINE.md`. The current disk artifacts, writer/reader
ownership, missing schema and provenance checks, non-atomic publication,
and optional pickle-only NPZ regime metadata are inventoried before any
writer or resume policy changes. A new contract test preserves the exact
legacy numerical arrays and separate JSON regime data. No disk schema or
calculation changes in this unit.

P7.2-b/D-095 introduces the result-disk schema v1 manifest and strict
opt-in loader, documented in `PHASE7_RESULT_DISK_SCHEMA_V1.md`. It
preserves ordinary and M-average numeric arrays exactly, removes only the
pickle-only duplicate NPZ regime metadata in favor of its existing JSON
sidecar, and rejects unversioned/unknown/inconsistent results explicitly.
The manifest records stored-file digests, dtypes, shapes, canonical units,
and selected raw model/execution declarations. The old summary reader,
atomic writes, checkpoint schema, and full provenance remain later units.
The full suite passes 1267 tests with 10 optional-GPU skips, branch coverage
is 78%, strict mypy covers 52 modules, and the installed-wheel import passes.

P7.2-c/D-096 migrates resumed-run file-backed summaries to the strict
v1 loader. Valid result populations and CSV columns remain unchanged.
Existing but unversioned, corrupt, or incomplete results now raise before
summary overwrite; genuinely absent results remain `failed`. The prior
print-only outer failure catch is removed. Normal returned-result summaries
and all-complete resume behavior are untouched. Full suite: 1270 passed,
10 optional-GPU skipped; strict mypy now covers 53 modules.

P7.2-d/D-097 replaces individual normal-result NPZ/JSON files atomically.
A temporary file is written and synced in the destination directory before
`os.replace`; the v1 manifest remains last. Failed individual writes preserve
the prior destination and remove the temporary file. Stored arrays, JSON
semantics, and propagation are unchanged. This does not guarantee a coherent
multi-file snapshot when replacing a prior result: strict validation reports
an interrupted or mismatched group. Checkpoint atomicity and provenance
remain separate P7.2 work. Full suite: 1275 passed, 10 optional-GPU
skipped; branch coverage: 78%; strict mypy covers 54 modules.

P7.2-e/D-098 atomically replaces each checkpoint JSON file while preserving
its key set, hash/deduplication semantics, write order, and resume behavior.
Failure injection proves that each existing destination is retained and its
temporary file removed when replacement fails. The checkpoint/failure-list
pair is not transactional; format versioning, validation, and provenance
remain separate P7.2 units. Full suite: 1277 passed, 10 optional-GPU
skipped; branch coverage: 78%; strict mypy covers 54 modules.

P7.2-f/D-099 publishes a writer-completed immutable result generation with one
atomic `result_current.json` pointer replacement. The strict reader resolves
it once, validates the unchanged manifest v1, and never falls back from an
invalid pointer. Legacy direct-layout v1 results remain readable but require
explicit migration before overwrite. Normal/resumed summaries retain their
respective population sources; only the result file layout changes. The
full suite passes 1289 tests with 10 optional-GPU skips; branch coverage is
78%, strict mypy covers 54 modules, and build/Twine checks pass.

P7.2-g/D-100 publishes `checkpoint.json` and `failed_cases.json` together
inside an immutable generation, selected by one atomic
`checkpoint_current.json` replacement. Payload or pointer failure keeps the
previous pair selected. Legacy direct-layout checkpoints remain readable and
upgrade on the next successful save. Case hashes, deduplication, cadence, and
resume filtering are unchanged. Invalid pointers never fall back or get
silently repaired. Strict schema/provenance validation remains separate.
Full suite: 1298 passed, 10 optional-GPU skipped; branch coverage: 78%;
strict mypy covers 54 modules. Build/Twine checks and an installed-wheel
checkpoint round trip pass.

P7.2-h/D-101 adds checkpoint payload schema v1 and validates every known field,
count, MD5 identifier, and the exact failure sidecar. Each save binds the
checkpoint to the complete ordered expanded case declaration with a named
SHA-256 scope. Resume reconstructs that declaration from `params.py` and
requires the digest, case count, and completed/failed membership to match
before filtering or execution. Existing invalid data raises
`CheckpointFormatError`; only an absent pair returns `None`. Unversioned
and unknown payloads require an explicit migration or new run and cannot be
silently upgraded. D-100 publication remains unchanged, as do MD5
deduplication, cadence, valid-run filtering, and all calculations. Full suite:
1309 passed, 10 optional-GPU skipped (1319 collected); branch coverage: 78%;
strict mypy covers 54 modules. Build, Twine, and installed-wheel checkpoint v1
round-trip/provenance rejection pass.

P7.2-i/D-102 migrates the three standalone result-directory plotters from
unversioned `tlist.npy`, `Efield_real.npy`, `Efield_vector.npy`, and
`population.npy` probes to one typed `visualization.result_data` projection
over `io.result_schema.load_simulation_result`. Field plots use `t_E/E`;
population uses `t_p/pop`. Leading dimensions must match their time axes, and
the vector plot requires exactly two Cartesian components. Invalid schema,
publication, payload, legacy-only input, and plot shapes raise before figure
creation. Plot series, filenames, all-state population behavior, show/save
order, and calculations remain unchanged. The only application-layer
dependency allowed from visualization is this strict schema reader. Full suite:
1313 passed, 10 optional-GPU skipped (1323 collected); branch coverage remains
78%, and strict mypy covers 58 modules. Build, Twine, and installed-wheel strict
reader round-trip/legacy-rejection validation pass.

P7.2-j/D-103 accepts P7.2 after the single-authority wiring audit, exact
persisted-array and valid-resume characterization, strict invalid-data policy,
complete CPU/coverage/quality/example gates, tracked Markdown/YAML checks, and
installed-distribution validation. The result/checkpoint disk contracts and
their exact non-guarantees are fixed by
`PHASE7_PERSISTENCE_ACCEPTANCE_AUDIT.md`. P7.2 is complete; P7.3 is next.

Complete source, environment, indirect-external-file, Hamiltonian, dipole,
generated-field, and numerical-input content provenance beyond the declared
expanded-run digest remains future separately versioned work, not a P7.2
guarantee.

### P7.3 Optimization

Implement D-072 before changes: central finite-difference GRAPE gradients with
step-size convergence, a direct one-iteration Krotov oracle, direct local
updates on the frozen legacy grid, and direct DFT/convolution spectral
constraints. Introduce common Objective, Evaluator, OptimizationResult, and
constraint interfaces only after these independent references pass or any
discrepancy is explicitly resolved by the user.

P7.3-a/D-104 completes the GRAPE reference and its user-approved correction.
The former time-local heuristic differs from the terminal-fidelity central
finite difference by approximately `1.79e4` relative error on the fixed
diagnostic. Production now reverse-differentiates the exact normalized dense
NumPy RK4 graph for
`1-fidelity+(lambda_a/2)*sum(E**2)`. The independent direct-RK4 oracle reaches
an observed `5.8e-9` relative-error plateau and fixes a `1e-7` bound. GRAPE
requires an explicit generated or sampled seed and rejects custom propagators
without a matching derivative. The unweighted discrete L2 penalty, update
direction, `target_fidelity` check, time grid, output sampling, and Class-D
scales are otherwise retained. The existing no-op `convergence_tol` branch is
recorded but unchanged. Full details are in
`PHASE7_OPTIMIZATION_REFERENCES.md`.

P7.3-b/D-106 completes the independent Krotov reference and the
user-approved correction. The former normalized-costate, batch-overlap,
factor-two calculation is frozen behind `legacy_batch_overlap`, including its
stored result and spectral example. Standard `krotov` now uses midpoint
piecewise-constant controls, overlap-scaled unnormalized costates,
`dH/dE=-mu`, no extra factor two, and sequential updated-state propagation.
Its required penalty unit is `1 / ((V/m)^2 fs)`; old field-grid and seed keys,
custom propagators, spectral constraints, and plotting raise. The independent
direct-RK4 one-iteration oracle agrees to `2e-15`; TwoLevel and five-level
VibLadder transfer plus half-step repropagation references pass.

P7.3-c completes the independent direct Local-control reference without a
production change. A slow test-only normalized RK4 calculation reproduces the
frozen two-segment D-027 layout and directly evaluates both `weights` and
`target` updates, including shaped seed, lookahead, shared endpoint,
componentwise clipping order, and final odd prefix. Field relative array-norm
errors are at most `1.22e-16` and trajectory differences are at most
`2.23e-16`. No formula discrepancy is found.

P7.3-d completes the independent spectral-kernel reference without changing a
numerical expression. Direct Gaussian construction covers pass/stop, max/sum,
FWHM/sigma, weights, clipping, scale, and the active wavenumber conversion.
Explicit dense DFT and periodic-convolution solves cover odd and even lengths;
production differs from the direct DFT by at most `8.89e-16`. This establishes
only the historical `legacy_batch_overlap` filter algebra, not standard
Krotov monotonicity. All D-072 optimization references now pass. P7.3-e may
introduce common optimization contracts and decompose orchestration while
retaining the independently fixed calculations.

P7.3-e1/D-107 replaces four partial result dictionaries with one typed
`OptimizationResult`. An exact `ControlLayout` distinguishes GRAPE/legacy
canonical RK4 samples, Local legacy samples, and standard Krotov midpoint
interval controls; only sampled-field layouts carry an `ElectricField`.
Arrays, metrics, time grids, and controls are retained by identity without
repair or reinterpretation. Active tests, plotting orchestration, and the
legacy benchmark consume the typed fields. Full verification passes 1380 CPU
tests with 10 optional-GPU skips (1390 collected), 80% branch coverage, and
strict mypy for 63 modules. P7.3-e2 should next introduce objective/evaluator
interfaces in a bounded unit without changing any fixed formula.

P7.3-e2/D-108 introduces that typed target-evaluation boundary. Direct indexed
population and vector ``vdot`` evaluation satisfy one protocol while retaining
their existing arithmetic and overlap reuse. The GRAPE discrete-L2 value has a
dedicated objective type; Krotov, legacy, and Local reporting retain their
existing expressions. Local ``weights`` is explicitly not folded into target
population. Full verification passes 1384 CPU tests with 10 optional-GPU skips
(1394 collected), 80% branch coverage, and strict mypy for 64 modules. P7.3-e3
may next separate Local response evaluators or constraint orchestration, one
referenced behavior at a time.

P7.3-e3/D-109 separates the two Local response calculations into distinct typed
evaluators. Weights mode retains `Im(<psi|A(-mu_a)|psi>)`; target mode retains
its overlap, derivatives, and `Im(conj(c)*d_a)` responses. Local orchestration
still owns lookahead, thresholds, seed signs, gain/shape, clipping, field
writes, and every frozen grid/index decision. The missing-target zero branch is
also characterized. Full verification passes 1387 CPU tests with 10
optional-GPU skips (1397 collected), 80% branch coverage, and strict mypy for
64 modules; `optimization.objective` is 100% covered. P7.3-e4 may next type the
legacy-only spectral constraint boundary without exposing it as standard
Krotov behavior.

P7.3-e4/D-110 moves the already strict legacy spectral-constraint schema into
one frozen ``LegacySpectralConstraint`` and compiles one
``LegacySpectralFilter`` on the unchanged runner rFFT grid. The runner no
longer repeats parsing with implicit defaults or coercions. Constraint absence
still selects the unfiltered legacy update; standard Krotov still rejects the
option. Mask and solve formulas, update addition, field grid, and all Class-D
values are unchanged. Full verification passes 1389 CPU tests with 10
optional-GPU skips (1399 collected), 80% branch coverage, and strict mypy for
64 modules; the spectral-constraint module now reaches 86% coverage. One P7.3
acceptance/audit unit should next verify all optimizer interface owners and
close the phase before P7.4.


P7.3-f/D-111 accepts the optimization boundary. Exact registry and owner
identity, dependency direction, broad-catch absence, stored legacy artifact
consistency, every independent scientific oracle, all four typed results, and
all strict schemas are executable contracts. Every optimization module and the
high-level runner now passes mandatory strict mypy. Full verification passes
1396 CPU tests with 10 optional-GPU skips (1406 collected), 80% branch
coverage, and strict mypy for 72 modules; optimization coverage is 72-100%.
The no-op GRAPE/legacy convergence condition, Class-D values, Local legacy
layout/defaults, standard-Krotov limitations, SymTop rejection, and absence of
an optimizer disk schema are explicit exclusions. No calculation changes.
P7.3 is complete; P7.4 is next.

### P7.4 Spectroscopy

Before splitting the current 1,049-line module, characterize:

- thermal state;
- response function;
- FFT sign/frequency convention;
- broadening;
- absorption/PFID/emission observables;
- normalization or sum rules.

Use D-072 analytic/direct references, then split by scientific responsibility.

P7.4-a1/D-112 adds the first independent references before moving production
code. A direct partition sum supplies a caller-owned two-level thermal density;
the resulting resonant plus counter-rotating Lorentzian response agrees with
the public loop result below `2.2e-16` relative discrepancy. A separately
derived single-coherence transform fixes the radiation/PFID frequency, phase,
and sign conventions below `1.6e-16`. Production has no thermal-state
constructor, so this checkpoint does not invent one. Doppler/device
broadening, normalization/area, and remaining observable references are next.
No production calculation changes. The full suite has 1398 passes and 10
optional-GPU skips (1408 collected); branch coverage remains 80%.

P7.4-a2/D-113 completes the current spectroscopy reference set. Direct
normalized convolution covers transition Doppler/Voigt behavior and all three
device functions; all exact response routes meet the analytic two-level answer.
The resonant chunked route has an observed `1.51e-12` accumulation difference
under a fixed `2e-12` bound. The weak-susceptibility and zero-response limits
also pass. No production calculation changes. P7.4-b may now move one scientific
responsibility at a time with before/after parity. The full suite has 1406
passes and 10 optional-GPU skips (1416 collected), branch coverage remains 80%,
and the spectroscopy monolith reaches 94%.

P7.4-b1/D-114 moves the existing experimental-condition value/unit boundary
into `spectroscopy.conditions`. Property formulas, strict validation, facade
identity, calculator use, and factory construction are unchanged. Architecture
guards fix the owner and dependency direction. The focused suite passes 45
tests; the full suite passes 1408 with 10 optional-GPU skips (1418 collected),
total coverage remains 80%, spectroscopy remains 94%, and strict mypy covers
73 modules.

P7.4-b2/D-116 moves uniform-grid validation, Gaussian filtering,
transition-specific Doppler broadening, and normalized Gaussian/sinc/sinc²
device convolution to `spectroscopy.broadening`. Calculator method names remain
delegates, and every D-113 formula, boundary mode, sample grid, normalization,
and error remains unchanged. The focused suite passes 47 tests; the full suite
passes 1410 with 10 optional-GPU skips (1420 collected), total branch coverage
remains 80%, and strict mypy covers 74 modules.

P7.4-b3/D-117 moves the byte-identical immutable
`SpectroscopyCalculationReport` to `spectroscopy.report`. The facade and
calculator compatibility name share the same class object; fields, order,
construction, and public behavior are unchanged. The focused suite passes 48
tests; the full suite passes 1411 with 10 optional-GPU skips (1421 collected),
coverage remains 80%, and strict mypy covers 75 modules.

P7.4-b4/D-118 moves the unchanged response-to-mOD conversion to
`spectroscopy.observables`. The calculator delegate supplies the same condition
properties; square-root branch, constants, operation order, and units remain
fixed by D-113. The focused suite passes 50 tests; the full suite passes 1413
with 10 optional-GPU skips (1423 collected), coverage remains 80%, and strict
mypy covers 76 modules.

P7.4-b5/D-119 moves the unchanged post-probe radiation/PFID response loop to
`spectroscopy.transform`. Validation, frequency conversion, observable
conversion, and public methods remain in the calculator; D-112 freezes the
sign, phase, indices, and denominator. The focused suite passes 52 tests; the
full suite passes 1415 with 10 optional-GPU skips (1425 collected), coverage
remains 80%, and strict mypy covers 77 modules.

P7.4-b6/D-120 moves unchanged 2D denominator preparation and the exact 2D,
matrix, and loop response bodies to `spectroscopy.response`. Calculator
validation, cache lifetime, method policy, report generation, and observable
conversion remain in place. D-112/D-113 fix every formula and route-specific
accumulation. The focused suite passes 54 tests; the full suite passes 1417 with
10 optional-GPU skips (1427 collected), coverage remains 80%, and strict mypy
covers 78 modules. Chunked exact/approximate extraction remains separate.

P7.4-b7/D-121 moves the unchanged CSR commutator, exact/approximate entry
selection, and fixed-order chunk accumulation to `spectroscopy.response`. Exact
mode retains all response-relevant nonzeros; only explicit approximation uses
the relative threshold. Report state, empty-response return, dispatch, and
observable conversion remain calculator-owned. The focused suite passes 54
tests; the full suite passes 1417 with 10 optional-GPU skips (1427 collected),
coverage remains 80%, and strict mypy covers 78 modules.

P7.4-b8/D-122 records the numerical ownership acceptance audit and adds five
executable guards for facade identity, owners, dependency direction, exception
policy, and remaining constructor defaults. All numerical responsibilities pass.
The focused suite passes 59 tests; the full suite passes 1422 with 10
optional-GPU skips (1432 collected), coverage remains 80%, and strict mypy
covers 78 modules.

P7.4-b9/D-123 starts the approved O-014 migration without removing the old
constructor. `CartesianProjection` owns exact typed axes and a normalized Jones
ket. `standard_absorption` consumes the existing `SystemModel`: scalar coupling
uses its model-owned storage axis with no dummy polarization, while Cartesian
coupling requires that projection. Detection uses the same physical probe ket
and its analyzer bra. Four exact routes are bitwise identical to the previous
explicit same-polarization construction. Arbitrary analyzer observables remain
separate and are not advertised by this unit. The full suite passes 1426 tests
with 10 optional-GPU skips (1436 collected), coverage remains 80%, and strict
mypy covers 79 modules. The temporary constructor must still be removed before
P7.4 final acceptance.

P7.4-b10/D-124 removes the direct-constructor and factory defaults for `axes`
and `pol_int` and rejects non-lowercase axis strings rather than coercing them.
Scalar standard absorption remains model-driven and needs neither input.
Explicit existing calculations retain identical values; arbitrary `pol_det`
remains temporarily until typed complex analyzer response replaces it. The full
suite passes 1427 tests with 10 optional-GPU skips (1437 collected), coverage
remains 80%, and strict mypy covers 79 modules.

P7.4-b11/D-125 separates complex molecular-response production from the
unchanged response-to-mOD conversion. The four numerical routes return their
existing angular-frequency and response arrays; `calculate` converts exactly
once and still applies the optional device function afterward. Exact empty
chunked dtype/value behavior is frozen. This adds no analyzer API and changes no
formula, threshold, route, accumulation order, or returned spectrum. The full
suite passes 1429 tests with 10 optional-GPU skips (1439 collected), coverage
remains 80%, and strict mypy covers 79 modules.

P7.4-b12/D-126 exposes that existing pre-mOD array through
`calculate_complex_response` and immutable `ComplexResponseSpectrum`. The result
stores a read-only cm^-1 grid, projected per-molecule response in C^2 m^2 / J,
and the calculation report. Absorbance and complex response share one strict
validation/dispatch implementation; only absorbance permits the existing
post-mOD device convolution. No formula or existing absorbance output changes.
The full suite passes 1432 tests with 10 optional-GPU skips (1442 collected),
coverage remains 80%, and strict mypy covers 79 modules.


P7.4-b13/D-127 adds `CartesianAnalyzerProjection` and the named
`analyzer_complex_response` constructor. Both Jones kets are normalized and
read-only, projection axes must exactly equal Cartesian model coupling axes,
and scalar models reject the inapplicable analyzer. Typed analyzer calculators
return complex response and explicitly reject scalar mOD conversion. Typed
arrays are not normalized twice. The old direct `pol_det` remains only for the
next removal unit. The full suite passes 1435 tests with 10 optional-GPU skips
(1445 collected), coverage remains 80%, and strict mypy covers 79 modules.


P7.4-b14/D-128 completes O-014. Direct construction and the condition factory
accept one typed standard or analyzer projection; raw axes and Jones arguments
are removed. All standard mOD methods, including radiation/PFID, reject analyzer
measurements, while typed analyzer calculations expose only the approved complex
response. Named model-aware constructors retain scalar and exact-axis checks.
Standard numerical outputs and all kernels remain unchanged. The full suite
passes 1435 tests with 10 optional-GPU skips (1445 collected), coverage remains
80%, and strict mypy covers 79 modules. Final P7.4 acceptance audit is next.


P7.4-c/D-129 accepts the final spectroscopy boundary and completes Phase 7.
Ownership, references, dependency direction, strict failures, typed measurement
construction, standard-absorption parity, and complex-only analyzer response all
pass. The full suite has 1435 CPU passes and 10 optional-GPU skips, branch
coverage is 80%, the calculator is 94% covered, Ruff/mypy pass, all supported
examples and their index pass, build/Twine pass, and the installed wheel exposes
the typed facade. No calculation changes. Analyzer intensity/absorbance,
thermal-state construction, and real CUDA remain explicitly outside acceptance.

### Phase 7 acceptance

- runner modules are individually testable;
- failed cases and resume behavior are deterministic;
- result schema is versioned;
- optimization and spectroscopy no longer have zero/near-zero critical
  coverage;
- no broad catch suppresses physics errors.

## 11. Phase 8 — public API and release

Tasks:

- implement the exact D-073 root exports and remove the old root surface — completed by P8.1-a/D-130;
- rewrite README and Japanese README against the actual API — completed by P8.2-a/D-132;
- close the Markdown/YAML/workflow findings in
  `DOCUMENTATION_WORKFLOW_AUDIT.md`, including executable public snippets,
  truthful badges, one explicit coverage authority, and release gating before any tag;
- execute documentation code snippets;
- update every supported example — completed early under D-044 with three
  typed smoke examples;
- move unsupported examples to an explicit archive or delete them — completed
  early under D-044 for the former v0.2 script set and consolidated by D-115;
- update version to 0.3.0;
- produce migration notes stating that backward compatibility is intentionally
  broken;
- build sdist and wheel;
- test clean installation;
- run all quality, physics, and benchmark gates;
- tag only after the refactor branch is clean.

P8.0-a/D-105 completes the pre-tag tooling safety subset early. The local release
tool accepts the current development-to-final transition but never commits,
tags, pushes, or publishes. The tag workflow rejects non-final versions and
requires full CPU gates plus a self-hosted real-CUDA reference before build and
publication. Jupyter is authenticated and localhost-only by default. The three
supported examples and the parameter template execute in CI smoke, and the
generated example index cannot scan archives. Root API/README migration,
CodeCov disposition, actionlint, final release evidence, and version bump
remain Phase 8 work. No calculation behavior changed.

P8.0-b/D-115 consolidates all tracked v0.2 example and optimization-config
material under `examples/archives/v0_2/{scripts,optimization_configs}`. The
three supported `configs/*.yaml` documents remain unchanged. A contract forbids
loose archived Python files and the superseded archive directory names. Ignored
runtime output, caches, build products, coverage files, and local validation
plots remain outside version control. No calculation behavior changed.

P8.1-a/D-130 implements the exact D-073 root surface with lazy authoritative
object resolution. Old convenience exports are removed without shims, and a
plain root import loads no workflow, persistence, optimization, spectroscopy,
visualization, Pandas, or Matplotlib module. The full CPU suite passes 1438
tests with 10 optional-GPU skips, branch coverage remains 80%, and strict mypy
covers 80 modules. README migration is next; no calculation behavior changed.

P8.1-b/D-131 makes the public runner annotation match its existing generated
and sampled field routes. The explicit `field` keyword remains required and no
execution branch changes. `simulation.runner` joins strict mypy as module 81.

P8.2-a/D-132 rewrites the English/Japanese public READMEs from the exact root,
supported examples, and verified capability matrix. Their marked quickstarts
execute in subprocesses, local links resolve, and stale API/coverage/GPU claims
are contract-rejected. The full suite passes 1446 tests with 10 optional-GPU
skips. Broader `docs/` migration and workflow lint/Codecov disposition remain.

P8.2-b/D-133 rebuilds `docs/README.md` as a current route/status index. It
removes obsolete and unverified recommendations, marks four public guides as
migration-audit work, and adds link/staleness contracts. No calculation or
configuration behavior changes.

P8.2-c/D-134 rebuilds the normal-simulation parameter guide from strict model
and generated-field schemas. Required keys, removed-name rejection, local
links, the corrected template CLI, and the real-CUDA disclosure are executable
contracts. Sweep, propagation, and unit guides remain next.

P8.2-d/D-135 rebuilds the sweep guide around exact insertion order, singleton
scalarization, Cartesian-product and checkpoint-provenance behavior. Dry-run
help now describes its unchanged count-only behavior, and the public simulate
CLI joins strict mypy as module 82.

P8.2-e/D-136 replaces the aspirational propagation-method survey with the
implemented typed time-grid, RK4/split, state-path, output, capability, and
CUDA contracts. It explicitly rejects claims for real-space FFT, fourth-order
Suzuki, adaptive propagation, sparse spectral split, and verified CUDA. The
full suite has 1466 passes and 10 optional-GPU skips (1476 collected). No
calculation code changes.

P8.2-f/D-137 rebuilds the unit guide around required caller value/unit pairs,
unchanged provenance, one conversion to named canonical values, exact converter
spellings, ordinary/angular frequency and wavenumber semantics, context-specific
field/intensity acceptance, opposite local-gain/standard-Krotov-penalty
dimensions, and exact spectroscopy labels. Unresolved Class-D optimizer values
remain unmodified. The full suite has 1481 passes and 10 optional-GPU skips
(1491 collected). No calculation code changes.

P8.3-a/D-138 removes the Docker image's persistent wildcard-bind, tokenless,
root-enabled Jupyter configuration and makes `pyproject.toml` extras the
container dependency authority. The rewritten guide matches the Dockerfile,
Dev Container JSON, safe launcher, supported examples, and real commands.
Static safety, JSON, shell syntax, link, and repository contracts pass; a clean
image build remains an explicit external check because Docker is unavailable
in this environment. The full suite has 1486 passes and 10 optional-GPU skips
(1496 collected). No calculation behavior changes.

P8.3-b/D-139 removes `codecov.yml` and `docs/CODECOV_SETUP.md` because no
workflow uploads to that service and no public badge remains. The existing CI
coverage job remains the sole authority: it enforces 47% branch coverage,
writes the GitHub summary, and retains report/XML artifacts. Contracts prevent
unwired Codecov files or public evidence claims from returning. The full suite
has 1488 passes and 10 optional-GPU skips (1498 collected). No calculation or
CI execution behavior changes.

P8.3-c/D-140 makes repository content and workflow validation mandatory. All
current and archived Markdown local links and code fences plus all YAML/YML
syntax are contract-tested. The required quality job runs checksum-verified,
pinned actionlint v1.7.12; its configuration declares only the intentional
self-hosted `gpu` runner label. Both workflows pass the same version locally.
The full suite has 1492 passes and 10 optional-GPU skips (1502 collected). No
package, configuration schema, calculation, or result behavior changes.

P8.4-a/D-141 publishes the breaking v0.2-to-v0.3 migration guide and links it
from every public documentation entry point and the release guide. It records
exact owner/key migrations, separates legacy batch overlap from standard
Krotov, refuses a guessed conversion for unresolved legacy modulation, and
requires new versioned runs instead of implicit result/checkpoint upgrades.
Five executable contracts protect those claims. The full suite has 1497 passes
and 10 optional-GPU skips (1507 collected). No implementation or calculation
behavior changes.

P8.4-b/D-142 removes two unused dependency manifests and false spectroscopy
subpackage metadata, leaving `pyproject.toml` and installed distribution
metadata as the single authorities. It rewrites the stale test guide around the
actual layout, markers, CI commands, coverage path, and GPU evidence policy.
Three contracts protect the cleanup. The full suite has 1500 passes and 10
optional-GPU skips (1510 collected). Spectroscopy exports and all calculations
are unchanged.

P8.4-c/D-143 completes the local release rehearsal and records it in a dedicated
readiness audit. CPU/coverage/quality, examples, actionlint, dry-run transition,
build/Twine, isolated wheel import/CLI, and payload checks pass. Final v0.3.0
remains blocked on device-native CUDA plus real-GPU evidence, an actual
container build/attach, protected publication setup, and the explicit final
version/changelog transition. No source or calculation behavior changes.

P8.5-a/D-147 adds one hard-failing development-container smoke script and makes
it a required normal-CI and final-tag job. The build context is an explicit
package-input allowlist. The runtime checks the unmounted image as non-root,
then an authenticated Jupyter API through a read-only checkout and isolated
writable notebooks mount; unauthenticated HTTP 200 is rejected. Local static,
workflow, and actionlint checks pass, but Docker is unavailable here. Phase 8
still requires a successful hosted-runner result plus the manual VS Code
attach/Ports check; no calculation behavior changes.


P8.5-b/D-148 replaces every mutable external Action ref in normal CI, manual
CUDA validation, and final-tag publication with a reviewed exact release
commit. A single contract requires the complete five-action allowlist and a
40-character commit for every `uses:` entry. Current Node 24 action releases
require self-hosted runner 2.327.1 or newer; real hosted/self-hosted execution
remains external evidence. No calculation behavior changes.


P8.5-c/D-149 replaces the long-lived PyPI API-token input with fail-closed OIDC
Trusted Publishing. Only the isolated publish job receives `id-token: write`;
it cannot build or check out source and has no password or fallback. The exact
PyPI publisher identity and protected `pypi` environment remain external
release prerequisites. No package or calculation behavior changes.


P8.5-d/D-150 makes successful local release preparation an explicit
non-acceptance handoff. It names the required changelog commit and all external
pre-tag gates instead of suggesting immediate tag creation. Commands, mutations,
and calculation behavior are unchanged.

P8.5-e/D-151 makes normal-CI diagnostics reproducible after the first hosted
run exposed a local/hosted ShellCheck mismatch. The quality job pins and
checksum-verifies ShellCheck 0.11.0 and gives its exact path to actionlint; the
coverage-summary SC2129 violation is removed without changing coverage policy.
Failed normal, physics, and coverage pytest runs publish escaped JUnit
annotations, and the container smoke publishes its failing stage. Diagnostics
never turn a failed calculation/test into success and do not alter package or
calculation behavior. The complete local suite has 1526 passes and 16
optional-GPU skips (1542 collected); a successful hosted rerun remains required.


P8.5-f/D-152 consumes the public diagnostics from hosted run `36812082890`
without changing any production calculation. The frozen legacy generated-field
test now uses a `1e-7 V/m` absolute cross-architecture FFT bound with zero
relative tolerance; the observed x86/arm64 difference was `7.94e-8 V/m`.
Python 3.10 uses the declared `tomli` release-tool fallback. Mypy 1.19.1 runs
nonincrementally under the fixed Python 3.12 quality environment while the
runtime matrix still tests 3.10-3.13. The container smoke avoids an absent
nested destination inside a read-only bind by mounting the checkout and
writable notebooks as siblings. The local suite remains 1526 passes and 16
optional-GPU skips (1542 collected). Hosted run `36813179835` subsequently
accepted Python 3.10 and the corrected Docker smoke; only the quality job's
mypy invocation failed.


P8.5-g/D-153 follows hosted run `36813179835`, which accepted the Python
3.10-3.13 matrix, physics, coverage, build, and corrected container smoke but
failed only at the bare mypy console command without a public diagnostic body.
Normal and release CI now invoke the pinned checker as `python -m mypy
--no-incremental`. The quality job tees output under pipefail and publishes it
through the same escaped Check-annotation reporter, which now has explicit
JUnit and text modes. The reporter cannot mask the original command failure.
No package source, type configuration, or calculation changes. The complete
local suite has 1527 passes and 16 optional-GPU skips (1543 collected).
Hosted run `36814129738` accepted the correction and every required normal-CI
job.


P8.5-h/D-154 records the external acceptance of P8.5-g commit `98e04fb`.
Run `36814129738` passed quality, Python 3.10-3.13, physics/contracts, branch
coverage, build/clean-wheel import, development-container smoke, and the
required aggregate. This closes normal hosted CPU/container acceptance only;
real CUDA, manual Dev Containers UI/Ports, exact publication infrastructure,
the final version transition, and tag-time repetition remain open. No code,
workflow, calculation, or test behavior changes.


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
