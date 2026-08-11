# Refactoring source of truth

Last verified: 2026-08-11
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
| `VALIDATION_INVENTORY.md` | P1.2-B audit of standalone diagnostics, replacements, and unresolved scale-policy constants | Any legacy validation disposition or recovered scientific intent |
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
| Pytest | 509 passed, 10 skipped (519 collected) |
| Measured branch coverage | 59% |
| Mandatory CI coverage floor | 47% |
| Ruff findings | 0 |
| Ruff safely auto-fixable findings | 0 |
| Files failing Ruff format check | 0 |
| Optimization module coverage | 7-16% |
| Spectroscopy coverage | 81% |
| `simulation/runner.py` coverage | 66% |
| RK4 Schrödinger coverage report | 20% |

The pytest, Ruff, and branch-coverage rows were verified locally on 2026-08-11 after the P1.6 implementation. The 47% gate intentionally starts at the accepted Phase 0 baseline; raise it in a dedicated coverage checkpoint after target ownership and omit policy are stable. The old README claim of 63% coverage is stale.

### Largest source hotspots

| File | Physical lines | Main concern |
|---|---:|---|
| `spectroscopy/absorbance_calculator.py` | 913 | Multiple response/spectrum responsibilities, 11% coverage |
| `simulation/runner.py` | 628 | Construction, execution, multiprocessing, output, error handling |
| `core/nondimensional/converter.py` | 562 | Strict scaling, array conversion, and object preparation |
| `core/propagation/algorithms/rk4/schrodinger.py` | 478 | Dense, sparse, CPU, GPU, validation paths in one module |
| `core/electric_field/core.py` | 478 | Field state, pulse construction, polarization, unit conversion |
| `core/propagation/utils.py` | 416 | Backend, units, field mapping, nondimensional preparation |

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

These commits are the starting point, not the final architecture.

## Phase status

| Phase | Name | Status |
|---|---|---|
| 0 | Physics characterization baseline | Complete — P0.1-P0.7 CPU baseline recorded; CUDA remains unverified |
| 1 | Repository and CI normalization | Complete — local and GitHub gates pass; `main` requires `Required CI gates` |
| 2 | Typed propagation contracts | Pending |
| 3 | Target package migration | Pending |
| 4 | Units and nondimensionalization | In progress — strict scaling and API consolidation complete; typed quantity migration pending |
| 5 | Numerical dynamics engine | Early work — P5.1-a RK4 dense/CSR and P5.2 CPU split polarization kernels complete; CUDA parity pending |
| 6 | Model consolidation | Pending |
| 7 | Simulation, optimization, spectroscopy decomposition | Pending |
| 8 | Public API, documentation, and release | Pending |

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
