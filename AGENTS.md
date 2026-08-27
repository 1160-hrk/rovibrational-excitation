# Codex repository instructions

Last verified: 2026-08-27
Active refactor branch: `refactor/v0.3`
Verified structural checkpoint: D-041 unit 4 (generated-field schema)
Latest infrastructure checkpoint: `62e6bfd`

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
5. `docs/refactoring/FALLBACK_AUDIT.md`
6. The source files and tests directly involved in the requested phase

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
- Typed trajectories always include the exact endpoint; if stride does not
  divide the step count, only the final output interval is shorter.
- Typed propagation requires an explicit initial state, algorithm, backend,
  dense/CSR storage, trajectory, stride, scaling, and renormalization policy.
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
- TwoLevel and VibLadder use scalar coupling and are physically independent of
  the supplied polarization direction. LinMol uses Cartesian coupling.
- Density matrices must be finite, square, Hermitian, positive semidefinite,
  and have positive real trace within the documented scale-aware tolerance.
- Liouville propagation currently supports NumPy dense RK4 only.
- Spectroscopy exact routes never prune response-relevant nonzero elements.
  Approximation and automatic routing are separate explicit modes with required
  controls and an observable calculation report.
- Spectroscopy experimental conditions are required, and Doppler broadening is
  derived from the actual uniform frequency-grid spacing.

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

Current local CPU baseline after the fourth D-041 generated-field unit:

~~~bash
pytest -q
~~~

~~~text
785 passed, 10 GPU tests skipped (795 collected)
~~~

The pre-change Phase 0 artifact is `benchmarks/baseline-v0.2.10.json`; the
Numba CSR comparison is `benchmarks/numba-csr-v0.2.10.json`. CUDA remains
unverified.

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
- Current after P1.6 local validation: 0 format failures and 0 Ruff findings.
- Current branch coverage: 70%; the initial mandatory CI floor is 47%.
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
are removed, all 97 discovered modules import, internal modules avoid root
convenience imports, and the top-level import graph has no mutual dependency.
P3.2-b moved model selection and required-input validation to
`models/validation.py`; simulation retains time, field, execution, and M-average
workflow validation and translates model errors at its boundary.
The next work is:

1. Continue Phase 4 with typed quantity and strict unit-validation boundaries.
   Characterize converter round trips and warning/fallback behavior before any
   implementation replacement.
2. Continue accepted D-041 after completing required selection, LinMol
   representation, scalar/Cartesian external-field injection, and the explicit
   generated-field schema: implement frozen model parameter schemas, consume
   them in a typed `SimulationCase`, and then enforce strict unknown- and
   inapplicable-key rejection. Structural field-grid validation never resamples, and numerical
   adequacy is a separate explicit convergence report.
3. Reduce exact transition debt only with the Phase 6 model consolidation. The
   four current entries are `dynamics.utils -> dipole.base`,
   `dynamics.scaling.converter -> dipole.base`, and two
   `models -> dynamics.problem` imports; never broaden or hide them.
4. Preserve the characterized visualization debts during the acceptance audit: `plot_population.state_index` is unused, three standalone plotters save after `show()`, `plot_electric_field` creates an empty legend warning, and optional spectrum/spectrogram errors are print-only. Fix them only in a separate behavior commit.
5. Defer persistence schema versioning and checkpoint-manager redesign until a separately tested persistence/API phase; P3.1-g intentionally preserves the unversioned schema and overwrite behavior.
6. Preserve private optimization adapters, especially `LocalOptimizerLegacyGridV1`, until their separately characterized Phase 7 migration.
7. Obtain independent optimization objective/gradient and spectroscopy references before Phase 7 decomposition.
