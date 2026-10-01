# P7.2 persistence and result-schema acceptance audit

Verified: 2026-09-23
Scope: P7.2-a through P7.2-i on `refactor/v0.3`
Disposition: **P7.2 complete; Phase 7 and v0.3.0 release remain open.**

This audit covers normal-result and checkpoint disk schemas, atomic generation
publication, resumed-summary validation, declared-run resume provenance, and
standalone result-directory plotting. It does not approve a numerical formula,
full source/environment provenance, a final public API, or CUDA capability.

## Ownership and executable evidence

| Responsibility | Current owner | Acceptance evidence |
|---|---|---|
| Single-file JSON/NPZ replacement | `io.atomic` | write/replace failure injection and temporary-file cleanup contracts |
| Result schema, manifest, and strict reader | `io.result_schema` | ordinary and M-average round trips, schema/digest/key/dtype/shape/unit rejection |
| One-case result writer | `simulation.result_persistence` | exact numeric-array characterization and generation-publication contracts |
| Complete-result publication | `result_current.json` plus immutable `.result_generations/` | payload, manifest, and pointer failure tests retain the previous generation |
| Resumed file-backed summary | `io.storage` | strict loader identity, valid population parity, and no-rewrite failure contracts |
| Standalone result plots | `visualization.result_data` | strict-loader identity, field/population projections, and legacy-NPY rejection |
| Checkpoint schema and publication | `io.checkpoint` | exact v1 fields, pair/pointer atomicity, sidecar equality, and malformed-input rejection |
| Declared-run resume binding | `io.checkpoint` and `simulation.resume` | ordered SHA-256 declaration, membership validation, and pre-execution changed-run rejection |

`tests/contracts/test_phase7_persistence_acceptance.py` fixes the application
wiring: normal result writes use the sole result-schema publication authority;
resumed summaries and standalone plots use its sole strict reader; runner,
batch, and resume use the sole checkpoint manager. Production source contains
no second `result.npz` reader and no legacy plot-array reader.

## Preserved calculation and workflow behavior

- The persisted `t_E`, `t_p`, `E`, `pop`, pure-state `psi`, M-average block
  trajectories, magnetic multiplicities, and weights remain exactly equal to
  the values supplied by the established writers.
- Result publication changes paths and failure atomicity only. It does not
  resample fields, reshape states, recompute populations, normalize weights,
  or alter propagation.
- Checkpoint pair publication and schema validation retain the historical MD5
  case identity, completed-over-failed precedence, batch cadence, valid-run
  filtering, process behavior, and calculated results.
- The complete ordered declared-run SHA-256 rejects changed parameters before
  case execution. Runtime-only `outdir`, `save`, and `error` remain excluded.
- Normal summaries still use returned in-memory results. Resumed summaries
  still reconstruct from disk, but now require a valid schema-v1 result.
- Standalone plots retain their series, filenames, all-state population
  behavior, labels, and `show()`-before-`savefig()` order. Their old missing-file
  print-and-return fallback is intentionally gone.

## Failure and fallback audit

- Unknown, unversioned, incomplete, malformed, digest-mismatched, unsafe, or
  differently declared persisted state raises a typed format error. An invalid
  publication pointer never falls back to a root-level legacy payload.
- Only a genuinely absent checkpoint pair returns `None`. An absent result for
  one resumed-summary case remains the established `failed` row; any evidence
  of a malformed result raises before summary replacement.
- Writer-side checkpoint validation occurs before generation creation.
  Result/checkpoint payload or pointer failure leaves the previously selected
  complete generation readable.
- The remaining broad catches are outside the P7.2 persistence authority:
  CLI exception-to-exit conversion, recorded batch-case isolation, and the
  already documented optional `plot_all` visualization debt. None is used to
  suppress schema, publication, provenance, or persistence errors.

## Repository and workflow review

- All 43 tracked Markdown files have valid conventional relative link targets.
- All 19 tracked YAML/YML files parse. The three supported optimization YAML
  documents remain separately covered by strict configuration tests; archived
  v0.2 YAML remains historical.
- `.github/workflows/ci.yml` still runs mandatory Ruff, named strict mypy,
  active examples, Python 3.10-3.13 tests, physics/contracts, 47% branch
  coverage, build, and clean-wheel import gates.
- `.github/workflows/release.yml` remains unsafe for a final tag because it can
  publish a matching development version without depending on the mandatory
  CI or real-GPU gate. `DOCUMENTATION_WORKFLOW_AUDIT.md` keeps this as a Phase 8
  release blocker. No tag or publication is authorized by this acceptance.
- The English/Japanese READMEs and several public guides remain stale in the
  exact ways recorded by `DOCUMENTATION_WORKFLOW_AUDIT.md`; P7.2 acceptance
  does not present them as current executable API documentation.

## Verification

- Focused P7.2 persistence/schema/visualization suite: **64 passed**.
- Full CPU suite: **1314 passed, 10 optional-GPU skipped** (1324 collected).
- Branch coverage: **78%**; mandatory floor: 47%.
- Strict mypy: **58 modules**, no findings.
- Repository Ruff lint: no findings; formatter: **269 active files** clean.
- All three active examples execute with their recorded final populations.
- Tracked YAML parsing and Markdown relative-link checks pass.
- The final sdist/wheel build, Twine inspection, and installed-wheel result and
  checkpoint schema round-trip/rejection smoke pass.

## Explicitly not closed by P7.2

- Complete source, dependency, environment, external-file, Hamiltonian,
  dipole, generated-field, and numerical-input content provenance. The current
  result hashes prove stored-payload consistency; checkpoint SHA-256 proves the
  declared expanded run only.
- Directory fsync/power-loss durability, concurrent-writer coordination,
  generation garbage collection, and an explicit historical-data migration
  tool. There is no automatic recovery, cleanup, or schema upgrade.
- P7.3/P7.4 independent optimization/spectroscopy references and scientific
  decomposition.
- Phase 5 device-native CuPy internals and real-GPU parity. Ten skipped GPU
  tests are not execution evidence.
- Phase 8 D-073 root API, public documentation rewrite, release-workflow
  hardening, migration notes, final clean-install audit, version `0.3.0`, and
  release tag.

P7.3 may begin from this persistence boundary. Any later expansion of the
provenance or durability guarantee requires its own schema version, tests, and
decision; it must not be inferred from P7.2 acceptance.
