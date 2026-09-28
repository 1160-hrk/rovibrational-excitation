# P7.3 optimization acceptance audit

Verified: 2026-09-28
Scope: P7.3-a through P7.3-e4 on `refactor/v0.3`
Disposition: **P7.3 complete; P7.4, Phase 5 CUDA, and the v0.3.0 release remain open.**

This audit closes the optimization decomposition phase. It does not authorize
another optimizer formula, claim universal convergence, assign units to the
remaining Class-D quantities, add SymTop optimization, or treat skipped CUDA
tests as execution evidence.

## Ownership

| Responsibility | Current owner | Acceptance evidence |
|---|---|---|
| Closed YAML/dictionary schema and plot/output policy | `optimization.config` | strict config and override contracts for all four algorithm names |
| Algorithm option values and axes | `optimization.options` | exact-type, enum, finite-value, unknown-key, and applicability contracts |
| Production model projection | `optimization.model` | LinMol, VibLadder, and TwoLevel construction parity; SymTop rejection |
| GRAPE/legacy generated or sampled field | `optimization.krotov_initial_field` | explicit discriminator, complete units, exact grid length, no resampling |
| Standard Krotov generated or sampled interval control | `optimization.krotov_controls` | explicit discriminator, complete units, exact interval count |
| Canonical RK4 field grid and output-only thinning | `optimization.timegrid` | D-029 time and solver-boundary references |
| Standard Krotov interval grid | `optimization.krotov_timegrid` | endpoints, midpoint controls, exact divisibility, output stride contracts |
| Frozen Local storage/index grid | `LocalOptimizerLegacyGridV1` | D-027 endpoint, shared-boundary, segment, and odd-prefix references |
| Target and Local response evaluation | `optimization.objective` | indexed/vdot separation and both direct Local response references |
| GRAPE discrete gradient | `optimization.grape_rk4` | independent central finite differences with step convergence |
| Standard sequential Krotov iteration | `optimization.krotov_rk4` | independent direct RK4 one-iteration oracle |
| Legacy spectral filter | `optimization.spectral_constraints` | direct Gaussian, DFT, convolution, and analytic-bin references |
| In-memory optimizer result | `optimization.result` | explicit three-layout typed result contracts |
| Algorithm registry and application orchestration | `optimization.ALGO_REGISTRY`; `simulation.optimize_runner` | exact registry identity, closed selection, typed result wiring |

All optimization package modules and the high-level optimization runner are
mandatory strict-mypy targets. The only additional typing change needed by the
runner import graph is an annotation on the existing spectrogram window array;
it changes no value or branch.

## Scientific calculations retained

| Route | Fixed calculation | Independent evidence |
|---|---|---|
| GRAPE | normalized dense NumPy RK4 discrete adjoint of `1-F + lambda_a/2 sum(E^2)` | every field component central-differenced; relative error below `1e-7` |
| Standard Krotov | sequential midpoint interval controls, overlap-scaled unnormalized costates, `dH/dE=-mu`, no extra factor two | all one-iteration arrays agree to `2e-15`; TwoLevel and five-level VibLadder transfers |
| `legacy_batch_overlap` | former normalized-RK4 batch update on the canonical half-step field grid | stored four-level artifact plus independent final propagation and integration test |
| Local | weights and target responses on the exact D-027 grid/index/update order | direct test-only RK4 fields agree to relative array-norm error at most `1.22e-16` |
| Legacy spectral filter | Gaussian pass/stop and max/sum mask followed by `FFT(source)/(1+alpha)` | direct DFT and periodic convolution differ from production by at most `8.89e-16` |

No acceptance edit changes an objective, gradient, update, normalization,
threshold, seed, clipping order, time sample, index, endpoint, propagation
call, or result array.

## Failure and fallback audit

- The registry contains exactly `local`, `krotov`, `legacy_batch_overlap`, and
  `grape`; unknown names raise and there is no default algorithm.
- Root, state, time, plot, output, algorithm, initial-field/control, Local
  initialization, and spectral mappings reject unknown or inapplicable keys.
- Generated and sampled control sources are explicitly discriminated. Arrays
  are never trimmed, padded, interpolated, normalized, or substituted.
- Standard Krotov rejects legacy field keys, custom propagation, spectral
  constraints, and plotting. GRAPE rejects unsupported axes and custom
  propagators without a derivative.
- An omitted optional legacy spectral constraint explicitly selects the
  unfiltered historical update. A supplied constraint has no implicit field
  defaults or method fallback.
- The optimization package and high-level runner contain no bare,
  `Exception`, or `BaseException` catch. Requested top-level plotting errors
  propagate.
- Numerical optimization owners do not import YAML, models, fields, I/O,
  simulation, CLI, or visualization services. The unit-aware spectral mask is
  the deliberate exception only for its `core.units` conversion boundary.

## Intentional retained behavior and open scientific decisions

These items are explicit exclusions, not unnoticed defaults:

- GRAPE and `legacy_batch_overlap` still observe a small
  `convergence_tol` difference without stopping. Changing this changes the
  iteration count and final field and requires a separate user-approved
  behavior decision.
- GRAPE `lambda_a`, `learning_rate`, and convergence tolerance; legacy
  `lambda_a` and convergence tolerance; and Local `c_abs_min`,
  `drive_abs_min`, and `shape_floor` remain Class D. No unit or recommended
  scale is inferred.
- Local legacy optional defaults, storage length, shared endpoints, segment
  slices, clipping, and odd final RK4 prefix remain frozen. Required gain,
  axes, and initialization remain explicit.
- Standard Krotov is dense NumPy interval-control RK4 and has no spectral
  constraint or plotting adapter. Its observed transfer histories are tests,
  not a universal monotonicity guarantee.
- SymTop optimization remains unsupported because it has no independent
  objective/control reference.
- Optimizer results are typed in memory. This audit does not introduce or
  claim an optimization-specific disk schema; `output.dir` currently owns the
  run directory and optional figures.

## Repository evidence

- The three top-level optimization YAML documents parse and pass the complete
  strict schema. Archived v0.2 documents remain excluded.
- The stored 1000-iteration four-level legacy artifact is guarded for algorithm
  identity, trajectory/field shape, final population equality, and exact
  independent-trajectory agreement.
- `test_phase7_optimization_acceptance.py` fixes registry identity, common
  contract ownership, dependency direction, broad-catch absence, and artifact
  consistency.
- The active CI runs the full test suite, physics/contracts, branch coverage,
  Ruff, strict mypy, build/Twine, and clean-wheel import. CUDA remains a
  separate required real-hardware release gate.

## Verification

- Focused optimization, Local-grid, visualization, and independent-reference
  suite: **212 passed**.
- Full CPU suite: **1396 passed, 10 optional-GPU skipped** (1406 collected).
- Branch coverage: **80%**; optimization modules are **72-100%** covered.
- Strict mypy: **72 modules**, including every optimization module and the
  high-level optimization runner.
- Repository Ruff lint and formatting: clean; `git diff --check`: clean.
- All four active example smokes pass and `examples/README.md` is current.
- The sdist and wheel build successfully; Twine accepts both artifacts.

## Completion disposition

P7.3 is complete. P7.4 spectroscopy is the next Phase 7 unit. Any future
change to an optimization formula, stopping condition, Class-D meaning,
supported model/backend, or persistence guarantee requires its own decision
and reference test; it is not authorized by this acceptance.
