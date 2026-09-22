# P7.2 persistence baseline and migration boundary

Verified: 2026-09-21
Scope: P7.2-a, after the `0.3.0.dev1` development checkpoint
Disposition: **Historical pre-P7.2-b baseline; see `PHASE7_RESULT_DISK_SCHEMA_V1.md` for the current result format.**

This inventory is the starting point for P7.2. It records what the current
writers and readers actually do. It is not approval to change a physical
formula, numerical array, time grid, unit conversion, or optimizer index.

## Pre-P7.2-b disk artifacts and owners

| Artifact | Writer | Reader | Current contract |
|---|---|---|---|
| `result.npz` | `simulation.result_persistence` | `io.storage.update_summary`, plotters, users | Direct `np.savez_compressed` overwrite, no disk schema version |
| `parameters.json` | `simulation.result_persistence` | users | Caller case after `json_safe`; written after NPZ |
| `regime_analysis.json` | `simulation.result_persistence` | users | Optional JSON when nondimensional regime data exist |
| `checkpoint.json` | `io.checkpoint.CheckpointManager` | manager and `simulation.resume` | Timestamp, counts, hashes, failures; direct overwrite |
| `failed_cases.json` | `io.checkpoint.CheckpointManager` | users | Separate direct overwrite after checkpoint |
| `summary.csv`, `summary_success.csv` | normal `simulation.reporting`, resumed `io.storage` | users/plotters | Normal uses returned values; resume reconstructs from result files |

The in-memory `dynamics.result.RESULT_SCHEMA_VERSION = 1` is **not** a
version on these disk files. A future disk schema must be independently named
and validated. The normal and M-average NPZ key sets are frozen by
`test_simulation_contracts.py` and `test_linear_molecule_reference.py`;
`test_persistence_contracts.py` additionally freezes exact time, complex
state, population, scalar-field arrays, and optional regime sidecar content.

## Existing data and failure boundaries

- `result.npz` stores time arrays in fs and sampled electric field in V/m.
  Ordinary pure-state results store `t_E`, `psi`, `pop`, `E`, and `t_p`.
  M-average results instead store reduced `pop`, `representation`, block
  `abs_m`/`m_multiplicity`/`m_weight`, and one `psi_abs_m_*` per block.
  The M-average weights and populations must not be recomputed in migration.
- Optional `regime_info` is currently a Python mapping saved into NPZ as an
  object array. Reading that key requires pickle; the separate
  `regime_analysis.json` contains its JSON-safe representation. New disk
  readers should not silently enable pickle on untrusted files.
- The checkpoint hash is MD5 of JSON-safe case data sorted by key, excluding
  `outdir`, `save`, and `error`. Completed cases take precedence over failed
  duplicates. This is a case-deduplication key, **not** full input provenance;
  it neither authenticates the parameter source nor the saved numerical
  arrays. Existing resume reconstructs cases from the saved `params.py` and
  filters by those hashes.
- The NPZ and JSON sidecars, checkpoint and failure file, and summary files
  are written independently without atomic replacement or a shared commit
  marker. Interruption can leave a partially updated set.
- `CheckpointManager.load_checkpoint` catches read/parse errors, prints a
  warning, and returns `None`. It does not validate structure or version.
  `io.storage.update_summary` classifies unreadable NPZ as `corrupted`,
  missing NPZ as `failed`, and any readable NPZ as `success` even when
  `pop` is absent. Its outer exception path prints a warning. These are
  inherited behaviours to replace only in an explicit tested policy unit.
- Normal-run summary is built from in-memory results; resumed-run summary is
  rebuilt from persisted files. P7.1 preserves this difference deliberately.

## Migration order

1. Characterize the current payloads and failure modes before replacing
   any writer (this checkpoint).
2. Define a disk schema identifier independent of package version and the
   typed in-memory result metadata. Choose a reader that accepts a documented
   version or raises an actionable error; never guess by key presence.
3. Add versioned, validated writes without altering existing numeric arrays.
   Keep non-JSON object data out of new NPZ metadata; record physical units,
   model, solver, field, grid, backend, and scaling provenance explicitly.
4. Make each file replacement atomic, then define and test a cross-file
   publication boundary so resume cannot mistake an incomplete set for a
   committed result.
5. Validate checkpoint structure and resume provenance before filtering cases
   or executing calculations. Define an explicit error/migration path for
   unversioned historical files; never treat them as a new version implicitly.
6. Audit normal/resume summary readers against the new schema, including
   missing/corrupt/partial files and all-complete resume, then run full
   physics, CPU, build, and installed-wheel gates.

P7.2-a changed tests and documentation only. P7.2-b then introduced the
versioned result manifest and strict opt-in loader while leaving checkpoints
and the legacy summary reader untouched. This document remains the exact
pre-migration inventory; the current contract is in
`PHASE7_RESULT_DISK_SCHEMA_V1.md`.
