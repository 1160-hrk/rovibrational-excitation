# Phase 8 release-readiness audit

Last verified: 2026-10-01
Local code checkpoint: P5.5-g/D-158 candidate
Hosted accepted checkpoint: P8.5-h/D-154 (`8a543b6`; run `36814615696`)
Target release: `0.3.0`
Current package version: `0.3.0.dev1`
Disposition: **not ready to tag**

This audit separates locally verified release inputs from evidence that requires
external infrastructure. Passing CPU checks or skipped GPU tests must not be
reported as complete v0.3 release acceptance.

## Locally verified gates

| Gate | Evidence | Status |
|---|---|---|
| Clean source checkpoint | D-158 candidate ignores local virtualenvs while retaining strict source provenance | Pass |
| Complete CPU suite | 1531 passed, 15 optional-GPU skipped; 1546 collected | Pass |
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

### 1. Device-native CUDA implementation and real-GPU evidence

Phase 5 is still open. D-144 and D-145 give RK4 and all three split modes
separate device-native CuPy owners. Source contracts forbid `.get()` and
`cp.asnumpy` before the explicit host boundary. CPU-backed doubles verify the
calculation graphs but cannot verify CUDA execution or performance.

Fifteen GPU tests are collected but skipped locally. D-155 makes the `gpu`
extra install the complete CUDA 12 user-space components; a driver-only WSL2
probe on the intended RTX 5070 Ti host now executes a basic CuPy kernel. The
focused library parity case also passes there. D-156 repairs the manual job after
the all-GPU command exposed a missing `plot` extra during global collection;
manual and tag-time jobs now install `dev,io,plot,gpu`. D-157 addresses
the resulting real-device LinMol failures: 12 GPU cases passed, three reached
unsupported `cupy.vectorize`, and one was a misclassified CPU error test. The
approved replacement preserves every existing dipole factor and phase in
device array operations; CPU-reference characterization passes. The D-157
hardware rerun passes all 15 GPU-marked tests. Its schema-v1 recorder rejected
only because repository-root `.venv-cuda/` was unignored; D-158 adds the
`.venv*/` ignore contract without weakening tracked or other untracked source
detection. A clean-commit evidence rerun remains required. These are not yet
library acceptance. D-146 adds
a manual pre-tag workflow and a release job for the real
`[self-hosted, linux, x64, gpu]` runner. Both run the trusted TwoLevel case,
every `gpu`-marked test, and a hard-failing schema-v1 recorder covering RK4
final/trajectory plus all three split modes. The report records NumPy/CuPy
parity, norm, shape, dtype, backend identity, explicit transfer volumes,
synchronized timing, hardware/software versions, and source identity.

Before release, run the manual workflow at the candidate commit, review an
accepted `status: pass` artifact, and rerun all local gates at that exact
commit. The tag workflow repeats the recorder, retains the artifact for 90
days, and attaches it to the GitHub Release. Source inspection, CPU skips,
queued jobs, and diagnostic `status: error` reports are not acceptance.

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
transition, but it is not a release candidate while the supported CUDA paths
lack real-hardware numerical, transfer, and performance evidence and the manual
Dev Containers UI and publication prerequisites remain open. The version
remains `0.3.0.dev1` and no tag or publication is authorized by this audit.
