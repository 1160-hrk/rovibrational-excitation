# P7.1 simulation-runner acceptance audit

Verified: 2026-09-21
Scope: P7.1-a through P7.1-q on `refactor/v0.3`
Disposition: **P7.1 complete; Phase 7 and v0.3.0 release remain open.**

This audit covers the normal-simulation and resume workflow decomposition.
It does not approve a scientific formula, persistence schema, public root API,
or CUDA capability change.

## Ownership and executable evidence

| Responsibility | Current owner | Acceptance evidence |
|---|---|---|
| Python parameter loading | `simulation.config` | `test_simulation_runner.py`, CLI contracts |
| Strict model/field/execution input | `simulation.validation`, immutable `simulation.case` | `test_simulation_contracts.py`, `test_simulation_case.py` |
| Generated-field sampling | `simulation.field_preparation` | `test_simulation_execution_wiring.py`, field/physics references |
| One-case preparation/propagation | `simulation.execution` | direct execution wiring and simulation physics contracts |
| One-case result payload/write | `simulation.result_persistence` | exact normal/M-average NPZ/JSON characterization |
| Retry and case-failure record | `simulation.safe_execution` | OSError-only backoff, traceback-file characterization |
| Pure Cartesian sweep | `simulation.sweep` | sweep tests and path-order characterization |
| Eager result paths | `simulation.case_paths` | normal, dry-run, and resume path characterization |
| Process batches/checkpoint cadence | `simulation.batch`, `io.checkpoint` | sequential/parallel batch, two-or-final save, hash/dedup contracts |
| Resume entry and filtering | `simulation.resume` | missing/corrupt checkpoint, missing params, all-complete, new-case tests |
| Normal returned-result summary | `simulation.reporting` | scalar/vector/final-row, failure-preview, all-failed CSV tests |
| Resumed file-backed summary | `simulation.reporting` calls `io.storage.update_summary` | old/new results, corrupted/missing file, early-exit tests |

The runner coordinates these owners and retains a top-level
multiprocessing-safe case callable. The new wiring contract checks that it
calls the single batch, path, resume, and reporting owners. No second
simulation formula, time grid, or implicit backend path was introduced.

## Behaviour checks

- Normal and resumed batches keep case order and one fresh process pool per
  batch. The checkpoint is saved after every second or final batch.
- Failure isolation records a returned error and, when saving, the same
  traceback and JSON-safe parameters. Only `OSError` retries.
- Normal summary uses returned populations; resume summary reads persisted
  NPZ files. These sources are deliberately distinct.
- All-complete resume returns without executing a case or rewriting summary.
- Both batch entries reject invalid `checkpoint_interval` before I/O;
  positive integer values retain their previous cadence.
- Direct normal simulation and generated/injected field paths remain under
  the pre-existing physics and unit contracts. Local-optimizer grid/index
  behaviour was not touched.

## Verification

- Full CPU suite: **1258 passed, 10 optional-GPU skipped** (1268 collected).
- Branch coverage: **78%**; mandatory floor: 47%.
- Strict mypy: **51 modules**, no findings.
- Repository Ruff lint: no findings; formatter: 260 active files clean.
- All three active examples execute and retain their recorded final
  populations.
- `python -m build` creates sdist and wheel; `twine check` passes.
  An installed wheel imports outside the workspace, including
  `simulation.resume`. Importing the wheel as a ZIP directly is *not* a
  supported test here because Numba's cache locator requires an extracted
  source path.

## Explicitly not closed by P7.1

- P7.2: result/checkpoint schema version, atomic writes, validated resume
  provenance, and file-backed summary error policy. Current files remain
  unversioned with inherited overwrite behaviour.
- P7.3/P7.4: independent optimization/spectroscopy references and their
  scientific decomposition.
- Phase 5: device-native CuPy internals and real-GPU parity. Ten skipped
  tests are not GPU evidence.
- Phase 8: D-073 minimal typed root exports, public READMEs, migration notes,
  clean-install release audit, and final `0.3.0` tag.
- The legacy `run_all` wrapper and optional process-count policy remain
  public/API review items; they are not claimed as P7.1 physics work.

A development version may mark this structural checkpoint, but the final
`0.3.0` release must wait for the separate Phase 5, 7, and 8 gates.
