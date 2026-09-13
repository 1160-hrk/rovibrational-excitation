# Codex repository instructions

Last verified: 2026-09-13
Active refactor branch: `refactor/v0.3`
Verified structural checkpoint: P5.4-a D-062 Phase 5 CPU acceptance audit
Latest infrastructure checkpoint: `7d4368b`

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
- Krotov initial fields explicitly select generated or sampled input. Generated
  seeds require physical value/unit pairs; sampled two-component fields require
  a direct amplitude unit and exact canonical-grid length. Neither route
  resamples or silently overrides the other.
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

Current local CPU baseline after P5.4-a:

~~~bash
pytest -q
~~~

~~~text
1193 passed, 10 GPU tests skipped (1203 collected)
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
- Current branch coverage: 75%; the initial mandatory CI floor is 47%.
- Optimization modules: 8-90% measured coverage; spectral constraints remain lowest.
- Spectroscopy monolith: 90% measured coverage.
- RK4 Schrödinger implementation: 20% measured line/branch coverage.
- README claims 63% coverage and contains removed APIs; it is not authoritative.

Targets are defined phase-by-phase in
`docs/refactoring/EXECUTION_PLAN.md`. Coverage must never decrease from the
recorded baseline for a phase.

## Current next work

Phase 0, Phase 1, and Phase 2 are complete. D-039 remains the verified typed
propagation boundary. Phase 3 is complete under D-040. Target package owners exist, superseded paths
are removed, all 127 discovered modules import, internal modules avoid root
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
The next work is:

1. Begin Phase 6 with TwoLevel ownership consolidation. Characterize the
   current basis, Hamiltonian, dipole, state mapping, and scalar-coupling
   projection before moving any implementation.
2. Keep Phase 5 open until a real CUDA job can remove and verify the current
   RK4/split `device -> host -> device` round trip. Do not edit that path using
   skipped tests as evidence.
3. Preserve D-061 endpoint reuse. The explored full output-buffer rewrite was
   slower on representative dimensions and introduced sub-ulp differences;
   do not revive it without a separate reference and benchmark.
4. Keep CuPy density propagation unsupported.
5. Do not touch the Class-D `c_abs_min`, `drive_abs_min`, `shape_floor`,
   `learning_rate`, `lambda_a`, or convergence tolerances without the user-defined
   dimensions and independent references.
6. Reduce exact transition debt only with Phase 6 model consolidation; never
   broaden or hide the four recorded reverse imports.
7. Preserve the characterized visualization debts and fix them only in a
   separate behavior commit.
8. Defer persistence schema versioning and checkpoint-manager redesign to its
   separately tested persistence/API phase.
9. Preserve private optimization adapters, especially
   `LocalOptimizerLegacyGridV1`, and obtain independent objective/gradient and
   spectroscopy references before Phase 7 decomposition.
10. Preserve the D-044 support boundary: active examples, benchmarks, and
   scripts remain executable and linted; archives remain historical until
   individually migrated and smoke-tested.
