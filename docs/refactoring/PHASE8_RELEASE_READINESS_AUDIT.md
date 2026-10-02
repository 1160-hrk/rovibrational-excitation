# Phase 8 release-readiness audit

Last verified: 2026-10-02
Local code checkpoint: P8.5-m/D-164 release typing remediation
Hosted normal-CI checkpoint: PR #15 main merge `c497318b`; run `36968773740`
Hosted real-CUDA checkpoint: main merge `c497318b`; run `36969653438`
Target release: `0.3.0`
Current package version: `0.3.0` (failed tag retired; unpublished)
Disposition: **typing remediation pending merge and exact-candidate revalidation**

This audit separates locally verified release inputs from evidence that requires
external infrastructure. Passing CPU checks or skipped GPU tests must not be
reported as complete v0.3 release acceptance.

## Locally verified gates

| Gate | Evidence | Status |
|---|---|---|
| Clean source checkpoint | D-158 ignores local virtualenvs while retaining strict source provenance | Pass |
| Real-CUDA numerical acceptance | D-159 implementation report plus D-160 main manual-workflow report; both raw JSON artifacts committed | Pass |
| Complete CPU suite | 1537 passed, 15 optional-GPU skipped; 1552 collected | Pass |
| Branch coverage | `coverage ... --branch`; total 81%, required floor 47% | Pass |
| Active-scope Ruff | `ruff check --no-fix src tests examples benchmarks scripts` | Pass |
| Active-scope format | 317 files formatted | Pass |
| Strict mypy scope | pinned 1.19.1, nonincremental, 84 named modules | Pass |
| Supported examples/template | three supported examples and `params_template.py --no-save` | Pass |
| Example index | `examples/tools/build_index.py --check` | Pass |
| Workflow semantics | checksum-verified actionlint v1.7.12 plus ShellCheck 0.11.0; every external Action is an approved exact commit | Pass |
| GPU dependency metadata | wheel and pyproject require `cupy-cuda12x[ctk]`; both GPU workflows install the `gpu` extra | Pass locally |
| Hosted normal CI | PR #15 main merge run `36968773740`: quality, Python 3.10-3.13, physics, coverage, build, container, and required aggregate | Pass for superseded `c497318b`; D-164 replacement pending |
| Publication authentication | D-162 tag-`v*` protected `pypi` environment and exact Trusted Publisher; isolated job-scoped OIDC with no username/password/secret/fallback | Pre-tag configuration pass; PyPI exchange remains tag-time |
| Target identity availability | PyPI latest is `0.2.10`; PyPI and GitHub Release have no `0.3.0`; the failed local/remote `v0.3.0` tag was retired with explicit user approval | Pass pre-tag |
| Failed tag diagnosis | Release run `36970840715` attempts 1-2 passed version, CUDA, and container gates; Python-3.12 mypy execution against its Python-3.10 target failed on NumPy 2.5.3 stub syntax before build/publication | Remediated locally by D-164; hosted acceptance pending |
| Container gate | D-162 main run `36966229314` passed the automated smoke; D-161 verified WSL attach, non-root environment, editable import, port 8888, and Jupyter in the browser | Pass hosted and manual |
| Release transition | `python scripts/release.py 0.3.0 --apply` changed only the authoritative version before running every local release gate | Pass |
| Distribution build | exact `0.3.0` sdist and pure-Python wheel | Pass |
| Distribution metadata | Twine accepts the new sdist and wheel | Pass |
| Isolated wheel import | temporary venv outside checkout imports the exact wheel from site-packages and reports `0.3.0` | Pass |
| Console entry points | installed `rve-simulate --help` and `rve-optimize --help` | Pass |
| Wheel payload | 155 entries, including both CuPy owners; no tests, archived examples, or removed dependency manifests | Pass |

The isolated wheel check used `--no-deps` with system site packages so it proves
wheel installation, import origin, metadata version, and entry-point creation.
It does not replace the clean-network dependency-resolution job in GitHub
Actions. Build isolation and Twine completed in this environment.

Ignored `dist/` contained older development artifacts. D-163 therefore
requires the exact `rovibrational_excitation-0.3.0.tar.gz` and
`rovibrational_excitation-0.3.0-py3-none-any.whl` names; stale substring matches
cannot satisfy or broaden the local Twine gate. GitHub Actions starts from a
clean checkout and publishes only artifacts built in that run.

## Release gates and remaining blockers

### 1. Final-version real-CUDA workflow repetition

Phase 5 is complete under D-159. D-160 additionally accepts the actual manual
GitHub workflow on main merge commit
`4f7efaed992b05691e204f78536dd9f2c54abfd3`. Normal CI run `36887536595`
and real-CUDA run `36887643743` both completed successfully. The ephemeral
`ashilab-gpu` runner carried the required `self-hosted`, `linux`, `x64`, and
`gpu` labels; device recognition, the trusted NumPy/CuPy reference, all 15 GPU
tests, five schema-v1 evidence cases, artifact upload, and post-job cleanup all
passed. The accepted raw report is committed as
`benchmarks/real-cuda-v0.3-4f7efaed.json`.

D-163 creates the untagged `0.3.0` version/changelog candidate. After it is
merged without further source changes, dispatch `Real CUDA validation` on that
exact main commit before tagging. Review its
`status: pass` artifact and provision another ephemeral runner for the tag
workflow, which repeats the same recorder. A report from a different commit,
queued job, skip, source inspection, or `status: error` is not final-release
acceptance.

The accepted 32-state timings remain diagnostic: GPU public-call medians were
approximately 17-76 ms versus 0.33-0.61 ms on CPU and include validation and
algorithm setup. No GPU speed advantage, workload crossover, or production
performance threshold is claimed.

### 2. Development-container manual UI verification — complete

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
PR #12 merged D-160 to main commit `841cb56`, whose normal-CI run
`36893481772` again passed the complete container smoke. D-161 then exercised
the independent UI path on Windows 11 with WSL2 Ubuntu 24.04. An explicitly
disabled WSL drive automount/interop configuration first produced the expected
`Failed to translate` error; restoring those standard facilities allowed
`Dev Containers: Reopen in Container` to complete. The opened container
reported `devuser`, `/workspace`, `/usr/local/bin/python`, Python 3.12.15,
package `0.3.0.dev1`, and the editable `/workspace/src` import. Port 8888
forwarding opened authenticated Jupyter in the browser. The manual attach and
Ports gate is therefore complete; the tag workflow must still repeat its
automated hard-failing container job.

### 3. Publication infrastructure — pre-tag configuration complete

D-162 accepts the external configuration after PR #13 main normal-CI run
`36964166468` passed every required gate. The existing GitHub environment is
named exactly `pypi` and its public API exposes one custom deployment rule:
tag pattern `v*`. The user confirmed registration of the existing PyPI project
with the exact Trusted Publisher identity: owner `1160-hrk`, repository
`rovibrational-excitation`, workflow filename `release.yml`, and environment
`pypi`. No long-lived API-token secret or fallback is configured. For this
single-maintainer repository, this audit does not claim an independent required
reviewer; deliberate annotated-tag creation plus the tag-only environment and
complete release workflow are the human and automated gates.

D-148 still fixes every external Action to a reviewed immutable commit, and
D-149 restricts `id-token: write` to the isolated publish job. Configuration
cannot prove the first OIDC exchange or publication: both remain hard-failing
operations in the final-tag workflow. The target `0.3.0` must also remain absent
from PyPI until that run, and the ephemeral self-hosted GPU runner must be
online at version 2.327.1 or newer. No local command in this audit publishes,
tags, pushes, or creates a release.

### 4. Final version transition and failed-tag recovery

D-163 produced the final `0.3.0` metadata and artifacts. PR #15 then merged it
to `c497318b0a72ca63f7b2e67704638c3ceb815dcd`; normal-CI run `36968773740`
and manual real-CUDA run `36969653438` passed on that exact source. The latter
used the required RTX 5070 Ti runner, passed all 15 GPU-marked tests and five
schema-v1 cases, and recorded a clean source commit.

The first annotated `v0.3.0` tag started release run `36970840715`. Attempts 1
and 2 passed tag/version verification, real CUDA, and container smoke, then
failed before the CPU suite in the combined quality/typing step. Build, PyPI
OIDC publication, and GitHub Release were skipped. A clean reproducer confirmed
Ruff and format success followed by mypy exit 2: NumPy 2.5.3 installed under
Python 3.12 contains Python-3.12-only stub syntax, while mypy intentionally
targets Python 3.10.

D-164 keeps the target and strictness intact and separates mandatory mypy into
Python 3.10 jobs in normal and release CI. Python 3.12 continues to own Ruff,
examples, the complete release CPU suite, and build; normal CI still tests the
full Python 3.10-3.13 runtime matrix. No runtime dependency is pinned to hide
the mismatch, and no type import is suppressed.

PyPI `0.3.0` and the GitHub Release were absent. With explicit user approval,
the failed remote and local `v0.3.0` tag was retired. The corrected release
sequence is now:

1. commit D-164 and merge it to main;
2. require normal CI, including the new Python 3.10 typing job, to pass;
3. dispatch manual real-CUDA validation on that exact new main commit;
4. review its schema-v1 `status: pass` clean-source artifact;
5. create a new annotated `v0.3.0` tag on the same accepted commit;
6. require the tag workflow to pass Python 3.10 typing, Python 3.12 CPU,
   real CUDA, container, build, clean-wheel, OIDC PyPI, and GitHub Release.

Evidence from `c497318b` documents the failure and CUDA correctness but cannot
authorize a tag on the replacement source commit.

## Release decision

The package remains unpublished at version `0.3.0`. D-164 fixes a release-only
tooling-environment mismatch without touching package runtime or calculations.
The failed tag has been retired, but no replacement tag is authorized until
the D-164 main commit passes normal CI and fresh source-bound manual CUDA.

The self-hosted GPU runner may remain registered for those gates. PyPI Trusted
Publishing is still configured exactly as required, but no OIDC exchange has
occurred and no GitHub Release exists.
