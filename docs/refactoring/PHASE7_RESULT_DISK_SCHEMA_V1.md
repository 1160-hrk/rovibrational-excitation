# P7.2-b result disk schema v1

Verified: 2026-09-22
Status: **Versioned writer and strict opt-in loader implemented.**
Scope: normal wavefunction and fixed-M incoherent-average simulation results.

This is a disk-format change, not a propagation or population change. The
five shared numeric arrays `t_E`, `t_p`, `E`, `pop`, and (for ordinary
pure states) `psi` keep their previous values, shapes, and units. M-average
block arrays and weights likewise remain unchanged.

## Files and authority

A saved result directory has `result.npz`, `parameters.json`, and the new
`result_manifest.json`. Nondimensional runs also have
`regime_analysis.json`. The manifest is the *only* disk-schema authority.
Its integer `schema_version=1` is independent of package version
`0.3.0.dev1` and of the in-memory
`dynamics.result.RESULT_SCHEMA_VERSION`.

The manifest contains exactly:

- `schema_version` and explicit `representation`:
  `wavefunction` or `m_incoherent_average`;
- SHA-256 digests of each named payload file (`result.npz`,
  `parameters.json`, optionally `regime_analysis.json`);
- the exact NPZ array names, dtype strings, and shapes;
- canonical unit labels: `t_E`/`t_p` in fs, `E` in V/m, and state,
  population, M indices/multiplicities/weights as dimensionless;
- a projection of the saved *caller declarations*: model selector,
  algorithm, backend, storage, nondimensional flag, renormalization,
  trajectory flag, and output stride. A missing declaration is recorded
  as JSON null, **not** silently assigned a default.

`parameters.json` retains the caller's value/unit pairs, as before.
`result.npz` retains only NumPy arrays that can be read with
`allow_pickle=False`. The former duplicate object-dtype
`regime_info` NPZ key is removed; its JSON-safe contents remain in
`regime_analysis.json`. No numerical regime scale or calculation changes.

## Strict read contract

`io.result_schema.load_simulation_result(directory)` requires the manifest,
accepts only known schema v1 and representation values, verifies the exact
set and SHA-256 digest of payload files, reads JSON sidecars, and checks every
NPZ key, dtype, shape, and declared unit. It never enables pickle or guesses
the representation from NPZ key presence. Missing manifest, unknown version,
malformed fields, object arrays, and corrupted payloads raise
`ResultFormatError` with an actionable message. An unversioned historical
result requires an explicit future migration path; it is not silently treated
as v1.

This strict loader is not yet used by `io.storage.update_summary` or the
standalone visualization code. Those old readers retain their characterized
behavior until separately tested P7.2 migration units. A versioned manifest
alone does **not** make writes atomic: NPZ and JSON files are still written
directly and the manifest is written last. Atomic replacement, cross-file
publication, checkpoint schema/provenance, and summary error policy remain
open. The SHA-256 digests protect stored file consistency, not full scientific
input provenance; they do not hash Hamiltonian or dipole source arrays.

## Verification

- Versioned ordinary and M-average round trips compare every representative
  numerical array against the writer input with exact equality.
- Tests reject missing/unknown schema, malformed representation, modified
  payload digest, and invalid NPZ even when its manifest digest matches.
- Existing pure-state, M-average, and checkpoint characterization tests
  remain the numerical and workflow guardrails.
- Full suite: 1267 passed, 10 optional-GPU skipped; branch coverage: 78%.
  Repository Ruff and strict mypy for 52 modules pass. Sdist/wheel build,
  Twine validation, and import from an installed wheel outside the workspace
  pass. CUDA execution is not verified.
