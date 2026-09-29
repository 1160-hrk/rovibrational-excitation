# Documentation, YAML, and GitHub workflow audit

Verified: 2026-09-29
Scope: all repository Markdown, YAML/YML, and .github/workflows files.
Disposition: **Inventory complete; public READMEs and release safety corrected, broader docs/workflow migration remains open.**

This is a content and wiring audit, not a new physics specification. The
authoritative calculation contracts remain PHYSICS_CONTRACTS.md and
DECISIONS.md. Do not make an old README example executable by guessing a unit,
sign, model parameter, or time step.

## Mechanical checks

- Inventoried 40 Markdown and 21 YAML/YML files at the original audit
  checkpoint, including historical optimization YAML under examples/archives.
- Parsed all YAML/YML files without a syntax error. PyYAML's YAML 1.1 parser
  reads the GitHub Actions on key as boolean True; this is a parser quirk, not
  validation of GitHub Actions semantics. Neither actionlint nor yamllint is
  installed here.
- Checked local targets of conventional Markdown links: no missing relative
  targets at the recorded acceptance checkpoints. This does not execute every
  code fence, inspect badges, or prove that linked APIs still exist.
- Loaded the three supported configs/*.yaml documents and passed each through
  validate_optimization_config without starting an optimization. Archived v0.2
  YAML documents remain unsupported historical evidence.
- D-105 adds executable contracts for release wiring, safe Jupyter defaults,
  release dry-run behavior, generated example-index consistency, and inclusion
  of params_template.py in the no-save smoke suite.

P7.2-j rechecked 43 tracked Markdown files and 19 tracked YAML/YML files.
Subsequent decision/reference documents legitimately increase those counts;
the count difference never promotes generated or archived artifacts.

## User-facing Markdown

| Files | Current finding | Required disposition |
|---|---|---|
| README.md, README_JP.md | D-132 rewrites both from the exact D-073 root and supported examples. Marked quickstarts execute, local links resolve, stale APIs/evidence are rejected, and SymTop/CUDA/analyzer limits are explicit. | Keep both languages contract-synchronized; update measured counts only from complete local gates. |
| docs/README.md | D-133 replaces the stale examples/recommendations with a current route and verification-status index; local links and forbidden old advice are contract-tested. | Keep migration-audit labels until each remaining public guide is independently corrected. |
| docs/PARAMETER_REFERENCE.md | D-134 rebuilds the guide from strict model/generated validators and the executable template; required keys, removed-name rejection, links, CLI, and CUDA disclosure are contract-tested. | Keep synchronized with schema changes; do not duplicate sweep or optimizer schemas. |
| docs/SWEEP_SPECIFICATION.md | D-135 records exact singleton scalarization, insertion-ordered Cartesian products, fixed list keys, paths, and resume provenance; guide/help/link contracts pass. | Keep order and checkpoint identity synchronized with `simulation.sweep`. |
| docs/TIME_PROPAGATION.md | D-136 replaces the aspirational method survey with the implemented typed time-grid, RK4/split, state-path, output, capability, and CUDA contracts. CPU capability/failure rows and key claims are tested. | Keep synchronized with `TimeGrid`, `PropagationOptions`, capability validation, and backend acceptance. |
| docs/UNIT_SYSTEM.md | D-137 rebuilds the guide around retained caller provenance, one canonical conversion, exact supported spellings, the 2π convention, field/intensity boundaries, optimizer dimensions, and spectroscopy labels. Converter parity and key boundary distinctions are tested. | Keep synchronized with `core.units` and never assign units to unresolved Class-D optimizer quantities by inference. |
| docs/DOCKER_SETUP.md | D-105 corrects Jupyter authentication, active example, quality, coverage, build, and publication commands. The rest of the container guide still needs final Phase 8 verification. | Recheck the complete Dev Container/Makefile flow in a clean checkout. |
| docs/VERSION_MANAGEMENT.md | Rewritten for the final-only, CPU-plus-real-GPU, no-automatic-push release contract. | Recheck once on the final clean release commit before creating a tag. |
| docs/CODECOV_SETUP.md | Still describes upload behavior that the current workflow does not perform. | Decide whether to restore Codecov upload or remove the badge/documentation claims. |
| docs/CARTESIAN_SPLIT_OPERATOR.md | Records the scientific split contract and explicitly states real-GPU parity is unverified. | Preserve equations; recheck source paths after backend acceptance. |

CHANGELOG.md contains a development checkpoint, not a final 0.3.0 release.
examples/README.md is generated only from the three top-level supported
examples; archives are never scanned. params_template.py keeps its numerical
values but labels them as examples rather than universal recommendations and
uses the library unit boundary instead of approximate hand-conversion advice.
configs/README.md correctly marks only three top-level YAML documents as
supported. D-115 consolidates all tracked historical v0.2 material under
`examples/archives/v0_2/{scripts,optimization_configs}`; ignored result, cache,
build, coverage, and validation-image artifacts are not tracked archives.
benchmarks/README.md labels v0.2.10 data as a historical baseline.
Historical test/archive reports retain their dates and are not current
capability evidence. Refactoring docs remain the agent-facing source of truth.

## Workflow and YAML wiring

| File | Verified behavior | Risk / next action |
|---|---|---|
| .github/workflows/ci.yml | Runs Ruff, mypy, four smoke executions, Python 3.10-3.13 tests, physics contracts, 47% branch coverage, and wheel import. required rejects failed/skipped jobs. | GPU tests may skip; normal CI is not real-GPU evidence. Markdown links/code fences and workflow semantics are not actionlint-gated yet. |
| .github/workflows/release.yml | Rejects non-final tags; requires full CPU gates and a self-hosted real-CUDA job; builds and clean-installs distributions; publishes PyPI before creating the GitHub Release. | The workflow is intentionally blocked until a [self-hosted, linux, x64, gpu] runner and PyPI environment/token exist. It has structural contract tests but has not been executed against those external systems here. |
| codecov.yml | Configures informational Codecov statuses. | No workflow uploads coverage to Codecov; badges/docs must not imply current Codecov evidence. |
| configs/*.yaml | All three current optimization configs parse and pass strict validation. | Keep them smoke-tested after optimization schema changes. No inferred physical values. |
| examples/archives/v0_2/optimization_configs/*.yaml | Parse as YAML but are explicitly unsupported v0.2 archives. | Do not promote without migration and an executable reference run. |

## Completion sequence

1. P7.3 optimization is accepted under D-111. Continue P7.4 spectroscopy
   decomposition from independent references.
2. D-105 completes the pre-tag tooling safety gate. Do not weaken the real-GPU
   job or substitute skipped CPU tests for its evidence.
3. D-130 finalizes the root and D-132 rewrites both public READMEs with
   executable examples. Rebuild docs/README.md next from audited current guides.
4. Migrate remaining public guides and snippets, and add automated local-link,
   supported-snippet, YAML, and actionlint checks.
5. On the final clean commit, repeat build/clean install, all active examples,
   release dry-run, real-GPU workflow, and external publication configuration
   checks before the final 0.3.0 tag.
