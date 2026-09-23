# Documentation, YAML, and GitHub workflow audit

Verified: 2026-09-22
Scope: all repository Markdown, YAML/YML, and `.github/workflows` files.
Disposition: **Inventory complete; public-doc and release-workflow migration remains open.**

This is a content and wiring audit, not a new physics specification. The
authoritative calculation contracts remain `PHYSICS_CONTRACTS.md` and
`DECISIONS.md`. Do not make an old README example executable by guessing a
unit, sign, model parameter, or time step.

## Mechanical checks

- Inventoried 40 Markdown and 21 YAML/YML files at this audit checkpoint
  (before `docs/RESULT_STORAGE.md` was added), including historical
  optimization YAML under `examples/archives/`.
- Parsed all 21 YAML/YML files without a syntax error. PyYAML's YAML 1.1
  parser reads the GitHub Actions `on` key as boolean `True`; this is a
  parser quirk, **not** validation of GitHub Actions semantics. Neither
  `actionlint` nor `yamllint` is installed here.
- Checked local targets of conventional Markdown links in all 40 Markdown
  files: zero missing relative link targets. This does not execute code
  fences, inspect badges, or prove that linked APIs still exist.
- Loaded the three supported `configs/*.yaml` documents and passed each
  through `validate_optimization_config` without starting an optimization.
  The archived v0.2 YAML documents were syntax-checked only and remain
  unsupported historical evidence.

## User-facing Markdown

| Files | Current finding | Required disposition |
|---|---|---|
| `README.md`, `README_JP.md` | Both still teach removed root/procedural APIs and `use_M`, claim 63% coverage and only linear-molecule support, advertise GPU capability beyond real-CUDA evidence, and link a nonexistent `tests.yml` workflow. Their quick-start parameter sets lack current explicit-unit/typed choices. P7.2-f corrected the output-directory layout and linked `docs/RESULT_STORAGE.md`; P7.2-i synchronized only the visualization tree and strict reader module. | Rewrite together during Phase 8 from supported executable examples and D-073 root API. Replace badges with actually produced evidence and state the SymTop/CUDA support matrix precisely. |
| `docs/README.md` | Quick-start uses the existing `examples/params_template.py`, but that template is outside the three CI-smoked supported examples; several other snippets omit now-required value/unit pairs or recommend unverified CuPy execution. | Smoke-test the template before advertising it, then rebuild the index from supported `examples/README.md` examples and the verified parameter reference. |
| `docs/PARAMETER_REFERENCE.md`, `SWEEP_SPECIFICATION.md`, `TIME_PROPAGATION.md`, `UNIT_SYSTEM.md`, `DOCKER_SETUP.md` | Mixed old/new examples; especially audit unit labels, solver capability, and executable CLI snippets against the frozen current schema. | Migrate one code fence at a time with smoke or contract tests, without changing physical defaults by inference. |
| `docs/CODECOV_SETUP.md`, `docs/VERSION_MANAGEMENT.md` | Describe `tests.yml`/Codecov upload or an automated release process not currently enforced by the workflows; version examples predate `0.3.0.dev1`. | Rewrite with the final CI/release design, then check commands in a clean checkout. |
| `docs/CARTESIAN_SPLIT_OPERATOR.md` | Records the scientific split contract and explicitly states real-GPU parity is unverified. | Preserve equations; recheck source paths and any runnable example after backend acceptance. |

`CHANGELOG.md` contains the development checkpoint, not a final 0.3.0
release. `examples/README.md` correctly separates three CI-smoked current
examples from archives. `configs/README.md` correctly marks only three
top-level YAML documents as supported. `benchmarks/README.md` labels v0.2.10
data as a historical baseline. `tests/*.md` and
`examples/archives/v0_2_optimization_configs/README.md` contain historical
test/migration reports; retain their dates and do not present the old counts
as current. The three `src/**/README.md` and `validation/README.md` need
path/API checks when their owning module is next changed. Refactoring docs
remain the agent-facing source of truth and must be updated per milestone.

## Workflow and YAML wiring

| File | Verified behavior | Risk / next action |
|---|---|---|
| `.github/workflows/ci.yml` | Runs Ruff, mypy, active example smoke, Python 3.10-3.13 tests, physics contracts, 47% branch-coverage floor, and wheel import. `required` rejects failed/skipped jobs. | GPU tests may skip; there is no real-GPU acceptance job. Markdown links/code fences, YAML schema, and workflow semantics are not checked. Coverage XML is uploaded only as an artifact, not to Codecov. |
| `.github/workflows/release.yml` | Triggers on any `v*` tag, checks tag against package version, then builds, creates a GitHub Release, and publishes to PyPI. It does not depend on the CI quality/test/physics/coverage jobs or real-CUDA gate. | **Before any release tag**, reject development/non-final versions and require release-facing gates; test the release workflow without publishing. A matching `v0.3.0.dev1` tag currently reaches the publish path. Do not create one. Review GitHub Release-before-PyPI order and token/permission scope. |
| `codecov.yml` | Configures informational Codecov statuses. | No workflow currently uploads coverage to Codecov; the root README badge/docs must not imply current Codecov evidence until upload is restored and verified. |
| `configs/*.yaml` | All three current optimization configs parse and pass the strict validator. | Keep them smoke-tested after optimization schema changes. No inferred physical values. |
| `examples/archives/**/*.yaml` | Parse as YAML but are explicitly v0.2 archive. | Do not promote without migration and an executable reference run. |

## Completion sequence

1. Continue P7.2 persistence work; do not mix README rewriting or workflow
   policy with numerical or storage commits.
2. Before creating any release tag, harden `release.yml` against development
   tags and require the final accepted CI/GPU gates. This safety gate has
   priority even if broader docs are deferred.
3. In Phase 8, finalize D-073 root exports, then rewrite the English/Japanese
   READMEs and `docs/README.md` using executed examples; correct support,
   installation, result-schema, and migration sections.
4. Migrate the remaining public guides and code fences, then add automated
   local-link and supported-snippet checks to CI. Resolve Codecov upload versus
   badge/docs deliberately.
5. Validate YAML, workflow semantics (with `actionlint` in CI), package
   build/clean install, all active examples, and a release dry run before
   the final `0.3.0` tag. Archived guides/configs remain explicitly historical.
