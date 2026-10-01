# Documentation, YAML, and GitHub workflow audit

Verified: 2026-10-01
Scope: all repository Markdown, YAML/YML, and .github/workflows files.
Disposition: **Public guides and mechanical repository-content/workflow gates are current; final external release evidence remains open.**

This is a content and wiring audit, not a new physics specification. The
authoritative calculation contracts remain PHYSICS_CONTRACTS.md and
DECISIONS.md. Do not make an old README example executable by guessing a unit,
sign, model parameter, or time step.

## Mechanical checks

- The original audit inventoried 40 Markdown and 21 YAML/YML files,
  including historical optimization YAML under `examples/archives/`.
  P8.3-c rechecks the current 48 Markdown and 19 YAML/YML files.
- A repository contract parses every current and archived YAML/YML file with
  PyYAML's `BaseLoader`, so the GitHub Actions `on` key is not coerced by YAML
  1.1 boolean rules. This is syntax coverage; actionlint separately validates
  workflow semantics.
- Repository contracts inspect rendered Markdown outside fenced code blocks,
  require every conventional local link target to exist, and reject unclosed
  backtick or tilde fences. They do not prove that an externally linked API or
  an unmarked prose snippet remains current.
- Official actionlint v1.7.12 passes all three workflows locally. Normal CI
  downloads pinned Linux amd64 actionlint and ShellCheck 0.11.0 archives,
  verifies both published SHA-256 values before extraction, and passes the
  explicit ShellCheck path to actionlint in the required `quality` job.
  `.github/actionlint.yaml` declares only the intentional `gpu` label used by
  the self-hosted CUDA runners.
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
| README.md, README_JP.md | D-132 rewrites both from the exact D-073 root and supported examples. D-155 documents the CTK-complete GPU extra while retaining the unverified real-CUDA boundary. Marked quickstarts execute, local links resolve, stale APIs/evidence are rejected, and SymTop/CUDA/analyzer limits are explicit. | Keep both languages contract-synchronized; update measured counts only from complete local gates. |
| docs/MIGRATION_V0_3.md | D-141 maps v0.2 imports, normal inputs, optimizer layouts, spectroscopy, and disk data to explicit v0.3 boundaries. It forbids guessed legacy modulation conversion, Krotov-layout conflation, and implicit result/checkpoint upgrades. | Keep mappings synchronized with runtime migration errors; add no inferred physical meaning. |
| tests/README.md | D-142 replaces stale counts, removed runners, Python 3.9/Actions v2 examples, ad-hoc installs, and false marker advice with the actual pyproject/CI commands and evidence hierarchy. D-155 records that the GPU extra supplies CUDA 12 user-space components but not the host driver or acceptance evidence. | Keep commands synchronized with CI and avoid duplicating volatile per-module coverage tables. |
| docs/README.md | D-133 replaces the stale examples/recommendations with a current route and verification-status index; local links and forbidden old advice are contract-tested. | Keep migration-audit labels until each remaining public guide is independently corrected. |
| docs/PARAMETER_REFERENCE.md | D-134 rebuilds the guide from strict model/generated validators and the executable template; required keys, removed-name rejection, links, CLI, and CUDA disclosure are contract-tested. | Keep synchronized with schema changes; do not duplicate sweep or optimizer schemas. |
| docs/SWEEP_SPECIFICATION.md | D-135 records exact singleton scalarization, insertion-ordered Cartesian products, fixed list keys, paths, and resume provenance; guide/help/link contracts pass. | Keep order and checkpoint identity synchronized with `simulation.sweep`. |
| docs/TIME_PROPAGATION.md | D-136 replaces the aspirational method survey with the implemented typed time-grid, RK4/split, state-path, output, capability, and CUDA contracts. CPU capability/failure rows and key claims are tested. | Keep synchronized with `TimeGrid`, `PropagationOptions`, capability validation, and backend acceptance. |
| docs/UNIT_SYSTEM.md | D-137 rebuilds the guide around retained caller provenance, one canonical conversion, exact supported spellings, the 2π convention, field/intensity boundaries, optimizer dimensions, and spectroscopy labels. Converter parity and key boundary distinctions are tested. | Keep synchronized with `core.units` and never assign units to unresolved Class-D optimizer quantities by inference. |
| docs/DOCKER_SETUP.md | D-138 removes unsafe Jupyter overrides; D-147 documents the shared build/non-root/authenticated-HTTP smoke used by normal and release CI. Static, JSON, shell, link, safety, and workflow contracts pass. | Require a successful hosted-runner container-smoke result and manually verify VS Code attach/Ports; this environment has no Docker CLI/daemon. |
| docs/VERSION_MANAGEMENT.md | Rewritten for the final-only, CPU-plus-real-GPU, no-automatic-push release contract. D-155 fixes driver-only WSL/Linux provisioning through `cupy-cuda12x[ctk]` and forbids installing a second WSL NVIDIA driver. | Recheck once on the final clean release commit before creating a tag. |
| removed docs/CODECOV_SETUP.md | D-139 removes the setup guide because no workflow uploads to Codecov and no public badge remains. | Do not recreate an external-service claim without a required, tested upload job. |
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
| .github/workflows/ci.yml | Runs checksum-verified actionlint v1.7.12 with pinned ShellCheck 0.11.0, Ruff, pinned nonincremental mypy through the Python 3.12 module, four smoke executions, Python 3.10-3.13 tests, physics contracts, Markdown/YAML contracts, 47% branch coverage, wheel import, and the D-147/D-152 container smoke. D-151/D-153 publish failed JUnit cases, captured command output, and the failing container stage as escaped Check annotations; `required` still rejects failed/skipped jobs. D-148 pins every external Action to an approved commit. Hosted run `36814129738` accepted D-153 and every required normal-CI job. | GPU tests may skip; normal CI is not real-GPU evidence. The automated container smoke is hosted-accepted, but manual Dev Containers UI/Ports verification remains. |
| .github/workflows/cuda-validation.yml | Manual pre-tag real-CUDA validation installs the D-155 CTK-complete GPU extra plus the complete test-collection extras under D-156, runs the trusted parity case, every GPU-marked test, and the schema-v1 evidence recorder, and retains success or diagnostic JSON for 90 days. D-148 pins every external Action. | It requires an online [self-hosted, linux, x64, gpu] runner at version 2.327.1 or newer. A basic CuPy probe, CPU skip, queued job, or status=error artifact is not acceptance evidence. |
| .github/workflows/release.yml | Rejects non-final tags; requires CPU, self-hosted real-CUDA, and D-147 container jobs; retains CUDA evidence, builds/clean-installs distributions, publishes PyPI, then attaches distributions and CUDA JSON to the GitHub Release. D-148 pins every external Action. D-149 gives only the isolated publish job OIDC permission and removes API-token input/fallback. | The workflow is blocked until GPU, Docker, protected environment, and exact PyPI Trusted Publisher infrastructure pass. Structural contracts do not substitute for those external executions. |
| removed codecov.yml | D-139 removes the unused service configuration. CI continues to enforce branch coverage and upload report/XML artifacts to GitHub Actions. | The repository-owned CI coverage job is the sole current authority. |
| removed requirements.txt, requirements-dev.txt | D-142 removes dependency manifests that diverged from the build metadata. `pyproject.toml` now solely owns runtime, optional, and development dependencies; D-155 makes its GPU extra CTK-complete for driver-only hosts. | Add dependencies only to the appropriate pyproject group and its tested installation route. |
| configs/*.yaml | All three current optimization configs parse and pass strict validation. | Keep them smoke-tested after optimization schema changes. No inferred physical values. |
| examples/archives/v0_2/optimization_configs/*.yaml | Parse as YAML but are explicitly unsupported v0.2 archives. | Do not promote without migration and an executable reference run. |

## Completion sequence

1. P7.3 optimization is accepted under D-111. Continue P7.4 spectroscopy
   decomposition from independent references.
2. D-105 completes the pre-tag tooling safety gate. Do not weaken the real-GPU
   job or substitute skipped CPU tests for its evidence.
3. D-130 finalizes the root and D-132 rewrites both public READMEs with
   executable examples. Rebuild docs/README.md next from audited current guides.
4. Public guide migration, Codecov disposition, repository-wide Markdown/YAML
   checks, actionlint gating, and the breaking-change migration note are complete
   under D-132 through D-141.
5. D-143 records the passing local CPU/coverage/quality/example/actionlint,
   release dry-run, build/Twine, and isolated-wheel checks. D-146 adds strict
   manual and tag-time CUDA evidence collection without claiming a local GPU
   result. Before the final `0.3.0` tag, obtain an accepted real-hardware
   artifact, require the D-147 hosted container smoke and manually verify VS
   Code attach/Ports, configure the exact D-149 PyPI Trusted Publisher and
   protected environment, then repeat the final-version gates from a clean
   commit. D-150 makes the local apply command print this as a non-acceptance
   handoff rather than suggesting immediate tag creation. D-151 pins the
   ShellCheck implementation used by actionlint and exposes failed tests and
   container stages through public Check annotations. D-152 consumes those
   diagnostics for Python 3.10, cross-architecture FFT-test, mypy, and clean
   checkout mount portability. Hosted run `36813179835` accepted all of those
   paths except the bare mypy command. D-153 invokes mypy through the active
   Python module and exposes captured output. D-154 records hosted run
   `36814129738`, where every required normal-CI gate passed. Retain real-CUDA,
   manual Dev Containers UI/Ports, publication, and tag-time gates before the
   final transition.
