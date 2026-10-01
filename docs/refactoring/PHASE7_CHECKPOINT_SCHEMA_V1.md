# P7.2 checkpoint disk schema v1 and resume provenance

Verified: 2026-09-23
Status: **Schema v1, strict validation, and declared-run resume validation implemented.**
Scope: normal batch checkpoints written by `io.checkpoint.CheckpointManager`.

This is a persistence and resume-policy change. It does not change a
Hamiltonian, field sample, time grid, propagation call, population, optimizer
update, process-batch size, or checkpoint cadence.

## Two independent versions

Checkpoint storage has two independent version numbers:

1. `checkpoint_current.json.publication_schema_version = 1` describes how an
   immutable generation is selected.
2. `checkpoint.json.checkpoint_schema_version = 1` describes the selected
   checkpoint payload.

A valid publication has this layout:

~~~text
run_dir/
├── checkpoint_current.json
└── .checkpoint_generations/
    └── <32-lowercase-hex generation ID>/
        ├── checkpoint.json
        └── failed_cases.json
~~~

The pointer is replaced only after both payload files are complete. The reader
resolves the pointer once, then validates the selected pair. An invalid pointer
or selected generation never falls back to a root-level file.

## Exact checkpoint v1 fields

A v1 `checkpoint.json` contains exactly:

~~~json
{
  "checkpoint_schema_version": 1,
  "timestamp": "2026-09-23T00:00:00",
  "start_time": 0.0,
  "total_cases": 2,
  "completed_cases": 1,
  "failed_cases": 0,
  "completed_case_hashes": ["<32-lowercase-hex MD5>"],
  "failed_case_data": [],
  "run_provenance": {
    "digest_algorithm": "sha256",
    "scope": "ordered_declared_cases_excluding_outdir_save_error",
    "case_count": 2,
    "cases_sha256": "<64-lowercase-hex SHA-256>"
  }
}
~~~

`failed_cases.json` must be a JSON list exactly equal to
`checkpoint.json.failed_case_data`. The reader rejects missing or unknown
fields, unknown schema versions, malformed timestamps and counts, non-finite
start times, invalid or duplicate case hashes, inconsistent completed/failed
counts, overlapping completed and failed cases, and a mismatched failure
sidecar.

Malformed JSON, unsafe symlinks, incomplete payload pairs, and corrupt
publication pointers raise `CheckpointFormatError`. They are not printed and
converted to `None`. A genuinely absent checkpoint still returns `None`.

## Declared-run SHA-256

The run digest is computed from the complete ordered case list after sweep
expansion and before execution. For each case, only the runtime fields
`outdir`, `save`, and `error` are removed. The remaining value is
converted through the existing JSON-safe representation and encoded with:

- sorted mapping keys;
- compact JSON separators;
- UTF-8;
- non-finite JSON numbers forbidden;
- case-list order retained.

SHA-256 is applied to those canonical bytes. Consequently, changing a physical
input, unit label, model, field declaration, algorithm, backend, storage,
trajectory policy, sweep membership, or sweep order changes the digest.
Changing only a result path, the save flag, or a recorded error does not.

The existing per-case MD5 remains unchanged. It is still only the historical
completed/failed case identity used for deduplication and filtering. The new
SHA-256 has a different purpose: it binds the checkpoint to the complete
declared run.

## Resume order

A resume performs these checks before executing a case:

1. Resolve one published checkpoint generation.
2. Strictly validate schema v1 and the failure-list sidecar.
3. Load the saved `params.py`.
4. Reconstruct the complete ordered case list and result paths.
5. Recompute the declared-run SHA-256 and require exact equality.
6. Require every completed and failed case identity to belong to that
   reconstructed run.
7. Only then filter completed cases and begin batch execution.

A changed `params.py` therefore raises `CheckpointFormatError` before the
case executor or summary writer runs. Saving also requires an explicitly bound
complete case list; a manager cannot invent provenance from only the currently
completed subset. Before creating a generation directory, the writer converts
both in-memory payloads to their JSON-safe form and passes them through the
same schema-v1 pair validator used by the reader. Invalid writer input therefore
publishes no pointer and creates no generation.

## Historical checkpoints

Unversioned direct-layout checkpoints and unversioned generations from the
earlier development format are not silently interpreted as schema v1 and are
not silently upgraded by saving. They raise an actionable
`CheckpointFormatError`. No automatic migration can prove their missing
complete-run provenance. The safe choices are to retain them as historical
artifacts or start a new run. A future explicit migration tool would need the
original complete parameter source and a separately reviewed policy.

Unknown future schema versions also raise. This prevents a newer payload from
being misread with older semantics.

## Guarantee boundary

Schema v1 guarantees:

- one internally consistent checkpoint/failure-list pair;
- strict known-field and known-version validation;
- exact equality between the failure sidecar and checkpoint payload;
- equality of the complete declared, expanded case list at resume time;
- membership of stored completed/failed case identities in that run;
- no fallback from corruption, unknown versions, or provenance mismatch.

It does not prove:

- that Python package source or dependencies are byte-identical;
- that generated Hamiltonian, dipole, field, or result arrays have identical
  content;
- that external files referenced indirectly by user code are unchanged;
- authenticity against a malicious writer;
- directory-entry durability after sudden power loss;
- safe coordination of concurrent writers;
- garbage collection of old or unpublished generations.

Those are separate source/environment/content provenance and durability
problems. Result-manifest payload hashes continue to protect saved result-file
consistency independently.

## Verification

- Ten direct schema/provenance tests cover exact metadata, unknown and
  unversioned payloads, failure-sidecar tampering, changed declarations,
  out-of-run completed hashes, runtime-field invariance, required binding, and
  legacy refusal, plus rejection before generation creation for invalid writer
  input and incomplete direct pairs.
- One runner integration test changes `params.py` after checkpoint creation
  and verifies that neither case execution nor summary writing begins.
- Atomic payload/pointer failure tests, cadence tests, and valid resume tests
  remain green.
- Full suite: 1309 passed, 10 optional-GPU skipped (1319 collected).
- Measured branch coverage: 78%.
- Strict mypy: 54 modules.
- Sdist/wheel build, Twine validation, and an installed-wheel checkpoint v1
  save/load plus changed-run rejection pass.

## Preserved behavior

The MD5 case formula and its `outdir`/`save`/`error` exclusions,
completed-over-failed deduplication, every-second-or-final batch checkpoint
cadence, result values, resume filtering for a valid unchanged run, and atomic
generation publication remain unchanged.

P7.2-j acceptance evidence and the guarantees deliberately left open are
recorded in `PHASE7_PERSISTENCE_ACCEPTANCE_AUDIT.md`.
