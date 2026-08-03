# Refactoring source of truth

Last verified: 2026-08-01
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
| Pytest | 360 passed, 9 skipped |
| Measured total coverage | 47% |
| Ruff findings | 1,143 |
| Ruff auto-fixable findings | 925 |
| Files failing Ruff format check | 63 |
| Optimization coverage | 0% |
| Spectroscopy coverage | 11% |
| `simulation/runner.py` coverage | 64% |
| RK4 Schrödinger coverage report | 10% |

The old README claim of 63% coverage is stale.

### Largest source hotspots

| File | Physical lines | Main concern |
|---|---:|---|
| `spectroscopy/absorbance_calculator.py` | 898 | Multiple response/spectrum responsibilities, 11% coverage |
| `core/nondimensional/converter.py` | 799 | Conversion, timestep policy, coupling scaling, object preparation |
| `simulation/runner.py` | 591 | Construction, execution, multiprocessing, output, error handling |
| `core/propagation/algorithms/rk4/schrodinger.py` | 570 | Dense, sparse, CPU, GPU, validation paths in one module |
| `core/electric_field/core.py` | 456 | Field state, pulse construction, polarization, unit conversion |
| `core/nondimensional/analysis.py` | 445 | Low coverage and overlapping policy responsibilities |
| `core/propagation/utils.py` | 426 | Backend, units, field mapping, nondimensional preparation |

Line count alone does not require splitting; mixed responsibility and poor
testability do.

## Known repository hygiene problems

- Generated `.coverage` and historical `tests/results/` files are tracked.
- An example notebook checkpoint is tracked.
- `test_basis_validation.py` and `test_new_api.py` are outside configured
  `testpaths = ["tests"]` and are not part of normal pytest.
- `test_new_api.py` contains code that deletes its own source file when run.
- Two detailed RK4 test files are empty.
- `tests/test_splitop_advanced.py.disabled` is not collected.
- `dipole/rot/jm_old.py` and old validation scripts remain tracked.
- `core/nondimensional/impl.py` is deprecated but still present.
- CI runs Ruff and Black, while Ruff formatting is already configured.
- mypy and physics validation are allowed to fail.
- Coverage upload does not enforce a minimum.
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

These commits are the starting point, not the final architecture.

## Phase status

| Phase | Name | Status |
|---|---|---|
| 0 | Physics characterization baseline | Complete — P0.1-P0.7 CPU baseline recorded; CUDA remains unverified |
| 1 | Repository and CI normalization | Pending |
| 2 | Typed propagation contracts | Pending |
| 3 | Target package migration | Pending |
| 4 | Units and nondimensionalization | Pending |
| 5 | Numerical dynamics engine | Early work — P5.1-a NumPy dense/CSR RK4 kernels complete |
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
