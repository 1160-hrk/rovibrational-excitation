# Phase 8 release-readiness audit

Last verified: 2026-10-01
Local code checkpoint: P5.5-h/D-159 candidate
Hosted accepted checkpoint: P5.5-g/D-158 (`b9de848`; run `36876859856`)
Target release: `0.3.0`
Current package version: `0.3.0.dev1`
Disposition: **not ready to tag**

This audit separates locally verified release inputs from evidence that requires
external infrastructure. Passing CPU checks or skipped GPU tests must not be
reported as complete v0.3 release acceptance.

## Locally verified gates

| Gate | Evidence | Status |
|---|---|---|
| Clean source checkpoint | D-158 ignores local virtualenvs while retaining strict source provenance | Pass |
| Real-CUDA numerical acceptance | D-159: 15 GPU tests and five schema-v1 cases pass on clean `b9de848`; committed raw report | Pass |
| Complete CPU suite | 1532 passed, 15 optional-GPU skipped; 1547 collected | Pass |
| Branch coverage | `coverage ... --branch`; total 81%, required floor 47% | Pass |
| Active-scope Ruff | `ruff check --no-fix src tests examples benchmarks scripts` | Pass |
| Active-scope format | 307 files formatted | Pass |
| Strict mypy scope | pinned 1.19.1, nonincremental, 84 named modules | Pass |
| Supported examples/template | three supported examples and `params_template.py --no-save` | Pass |
| Example index | `examples/tools/build_index.py --check` | Pass |
| Workflow semantics | checksum-verified actionlint v1.7.12 plus ShellCheck 0.11.0; every external Action is an approved exact commit | Pass |
| GPU dependency metadata | wheel and pyproject require `cupy-cuda12x[ctk]`; both GPU workflows install the `gpu` extra | Pass locally |
| Hosted normal CI | run `36814129738`: quality, Python 3.10-3.13, physics, coverage, build, container, and required aggregate | Pass |
| Publication authentication | isolated job-scoped OIDC; no username/password/secret/fallback | Pass locally; PyPI exchange external |
| Container gate wiring | minimal build context, shell syntax, required normal/release jobs; hosted run `36814129738` succeeded | Pass hosted; manual UI check external |
| Release transition | `python scripts/release.py 0.3.0 --dry-run` reports `0.3.0.dev1 -> 0.3.0` and writes nothing | Pass |
| Distribution build | sdist and pure-Python wheel for `0.3.0.dev1` | Pass |
| Distribution metadata | Twine accepts the new sdist and wheel | Pass |
| Isolated wheel import | temporary venv outside checkout imports the wheel from site-packages and reports `0.3.0.dev1` | Pass |
| Console entry points | installed `rve-simulate --help` and `rve-optimize --help` | Pass |
| Wheel payload | 155 entries, including both CuPy owners; no tests, archived examples, or removed dependency manifests | Pass |

The isolated wheel check used `--no-deps` with system site packages so it proves
wheel installation, import origin, metadata version, and entry-point creation.
It does not replace the clean-network dependency-resolution job in GitHub
Actions. Build isolation and Twine completed in this environment.

Ignored `dist/` contained older local artifacts, so local inspection selected
the new `0.3.0.dev1` files explicitly. GitHub Actions starts from a clean
checkout and publishes only artifacts built in that run.

## Blocking release gates

### 1. Final-candidate real-CUDA workflow repetition

Phase 5 is complete under D-159. The clean `b9de848` target-host run passed all
15 GPU-marked tests and produced an accepted five-case schema-v1 report on an
RTX 5070 Ti. It verifies device-native CuPy `complex128` results for RK4
final/trajectory and all three split modes, fixed parity and norm bounds,
transfer volumes, synchronized timings, hardware/software identity, and exact
source provenance. The report is committed as
`benchmarks/real-cuda-v0.3-b9de848.json` and revalidated by a contract test.

This does not eliminate the release-specific gate. D-159's documentation and
evidence commit is newer than the tested source, and the final version/changelog
commit will be newer again. Before tagging, dispatch `Real CUDA validation` on
the exact final candidate using an online `[self-hosted, linux, x64, gpu]`
runner version 2.327.1 or newer. Review its `status: pass` artifact and require
the tag workflow to repeat the same recorder. A local artifact from a different
commit, queued job, skip, source inspection, or `status: error` is not release
acceptance.

The accepted 32-state timings are diagnostic: GPU public-call medians were
approximately 16-68 ms versus 0.31-0.52 ms on CPU and include validation and
algorithm setup. No GPU speed advantage, workload crossover, or production
performance threshold is claimed.

### 2. Development-container manual UI verification

D-147 supplies one hard-failing smoke script to both required normal-CI and
final-tag jobs. It builds the minimal-context image, verifies its non-root
package/Jupyter environment without a checkout mount, then verifies an
authenticated Jupyter API over a dynamically published localhost port with a
read-only checkout and isolated writable notebooks mount. Unauthenticated HTTP
200 is rejected.

Shell, content, workflow, and actionlint-plus-ShellCheck contracts pass, and
Docker absence fails rather than skips. Hosted run `36812082890` localized
Docker exit 125 to the Jupyter-launch stage. D-152 removes the absent nested
mount target by placing the read-only project and writable notebooks mounts at
sibling paths. The corrected container-smoke job passed on hosted run
`36813179835`, including image build/import and authenticated HTTP checks.
VS Code Dev Containers UI attach and its Ports view remain an explicit manual
check; neither is inferred from that automated smoke.

### 3. Publication infrastructure

The protected `pypi` environment, exact PyPI Trusted Publisher identity,
availability of the target version on PyPI, and online self-hosted GPU runner at
version 2.327.1 or newer must be confirmed externally. D-148 fixes every
external Action to a reviewed immutable commit, and D-149 removes long-lived
publication secrets in favor of job-scoped OIDC. Neither proves execution on
that infrastructure.
No local command in this audit publishes, tags, pushes, or creates a release.

### 4. Final version transition

Do not run `python scripts/release.py 0.3.0 --apply` until the CUDA, manual
Dev Containers UI, and external publication prerequisites are ready. That
command changes only `pyproject.toml`, runs local gates, and never
commits/tags/pushes/publishes. Its success message explicitly
states that this is not release acceptance and repeats the external blockers.
After it passes:

1. update the changelog from an Unreleased development record to the reviewed
   final `0.3.0` release date;
2. review the version diff and rerun the clean local gates;
3. commit the version/changelog change explicitly;
4. push the commit and confirm normal required CI;
5. create and push the annotated `v0.3.0` tag explicitly;
6. require the tag workflow to pass CPU, real CUDA, container, build,
   clean-wheel, and PyPI gates before GitHub Release creation.

## Release decision

Local CPU, documentation, packaging, and dry-run preparation are complete at
this checkpoint. Hosted run `36813179835` accepted the D-152 Python matrix,
physics, coverage, build, and container corrections; only the quality job's
bare mypy command failed with exit 2 and no public body. D-153 uses the pinned
module entrypoint and publishes captured command output while retaining hard
failure. Hosted run `36814129738` accepted D-153 and every required normal-CI
job, including the automated container smoke. The project is close to a version
transition. Phase 5 now has accepted real-hardware numerical, transfer, norm,
and timing evidence, but the exact final candidate still requires a fresh
manual CUDA workflow, manual Dev Containers UI/Ports verification, and the
publication prerequisites. The version remains `0.3.0.dev1` and no tag or
publication is authorized by this audit.
