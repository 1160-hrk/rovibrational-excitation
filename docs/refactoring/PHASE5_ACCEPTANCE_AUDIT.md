# Phase 5 numerical-engine acceptance audit

Last verified: 2026-10-01
Checkpoint: P8.5-i / D-160 main-workflow real-CUDA evidence; Phase 5 complete

## Scope

This audit compares the current implementation and executable tests with every
Phase 5 acceptance row. CPU-backed doubles verify dispatch and calculation graphs;
D-159 and D-160 separately supply accepted numerical, backend, transfer, norm,
and synchronized timing evidence from real CUDA hardware. D-160 additionally
proves that the repository-owned manual workflow executes that gate on `main`.

D-155 verifies the environment boundary on the intended WSL2 host: plain
`cupy-cuda12x` enumerated the real device but lacked CUDA user-space runtime,
NVRTC, and headers; `cupy-cuda12x[ctk]` supplied CUDA 12.9 components and ran
a basic CuPy kernel on the RTX 5070 Ti. This proves reproducible device
bootstrap only. It does not replace the library parity tests or evidence
recorder.

## P5.1 RK4

| Requirement | Status | Evidence |
|---|---|---|
| NumPy dense and CSR kernels are separate | Verified | D-018; `test_sparse_rk4_reference.py` |
| CSR selection is explicit and does not prune values | Verified | D-018; sparse reference contracts |
| Liouville validation and kernel are separate | Verified | D-060; `test_liouville_kernel_separation_contracts.py` |
| Liouville endpoint reuse preserves the old loop | Verified exactly | D-061; final and trajectory `np.array_equal` comparisons |
| Field left/mid/right indices and `-mu E` agree | Verified | solver physics and density contracts |
| NumPy performance is measured | Verified | Numba CSR and Liouville endpoint-reuse artifacts |
| CuPy RK4 graph matches CPU and remains device-native | Verified on real GPU | D-144; D-159; accepted schema-v1 artifact |
| Real CUDA parity and diagnostic timing | Verified | D-159/D-160; both accepted `real-cuda-v0.3-*.json` reports |

D-144 removes the former fused RawKernel. That kernel used `H0 + mu E`, added
its `dt*k3` stage a second time during the final update, ignored trajectory,
stride, and renormalization, and returned through `.get()`. The replacement
uses the same dense RK4 graph as CPU and returns its CuPy allocation directly.
No GPU speed claim is made without measurement on actual hardware.

## P5.2 split operator

| Requirement | Status | Evidence |
|---|---|---|
| Diagonal `H0` is required | Verified | solver invariant and contract tests |
| Both Cartesian field components use midpoint samples | Verified | rotating-xy circular reference and RK4 convergence |
| Fixed direction uses one static eigensystem | Verified | static NumPy kernel and permanent-dipole reference |
| Changing xy direction uses M-diagonal rotations | Verified | rotation covariance and circular-field tests |
| Cartesian and helicity-projected are distinct explicit models | Verified | D-019 and public interaction-mode contracts |
| Hermiticity and covariance failures raise without repair | Verified | polarization physics tests |
| CSR inputs execute through documented dense spectral vectors | Verified exactly | `test_phase5_acceptance_contracts.py` |
| Cartesian converges to RK4 at two step sizes | Verified | coarse/fine ratio 3.987 |
| Public, spectral-setup, and inner-loop timing are separate | Verified | `split-polarization-v0.3.json` and its report contracts |
| NumPy/CuPy numerical parity and shapes | Verified on real GPU | D-159; all three split evidence cases pass |
| CuPy split graph and result remain device-native | Verified on real GPU | D-145; D-159; device-result checks |

The direct NumPy/Numba dependency is honest. Numba is a required project
dependency, so missing Numba raises the ordinary dependency error rather than
silently changing execution mode.

## P5.3 backend transfer policy

D-026 and D-039 decide the public policy:

1. `PropagationResult.state` belongs to the selected backend.
2. An existing device state passes through finalization by identity.
3. Host conversion occurs only through explicit `PropagationResult.to_numpy()`.
4. Persistence and host-only analysis call that boundary explicitly.

The typed boundary and CuPy RK4 now satisfy this source-level policy. RK4
operators, fields, stages, trajectories, and final states stay in CuPy arrays.
With `renorm=True`, a scalar validity check synchronizes so the existing error
contract can be preserved; no state or trajectory array is copied to the host.

D-145 gives split the same source-level transfer policy. Static Cartesian,
rotating Cartesian, and helicity-projected preparation, eigensystems, states,
and outputs stay in CuPy arrays. Scalar synchronization remains only where the
existing zero-amplitude branch or validation/renormalization error semantics
requires a Python decision; no state or trajectory array crosses to host.

## Phase 5 acceptance disposition

| Acceptance condition | Status |
|---|---|
| Every advertised CPU capability executes | Pass |
| CPU physics baselines | Pass |
| No silent algorithm/backend/storage fallback | Pass |
| NumPy performance and trajectory memory are documented | Pass |
| Existing device state is retained by the result boundary | Pass |
| CuPy RK4 source graph and device-native return | Pass |
| CuPy split source graph and device-native return | Pass |
| Real CUDA RK4 and split parity/backend/norm | Pass |
| Real CUDA transfer and synchronized diagnostic timing | Pass |
| No array round trip before the explicit host boundary | Pass |

Phase 5 is **complete** under D-159. The accepted report binds the result to the
clean `b9de848cf3fa8322e5f685f60191857e528531d4` source commit and records all
five required cases as CuPy `complex128` device results. This phase completion
does not authorize a tag: the final release candidate must repeat the manual
self-hosted CUDA workflow at its own exact commit.

## Accepted real-CUDA evidence and release repetition

D-159 records the clean D-158 rerun on an RTX 5070 Ti. All 15 GPU-marked tests
passed. The schema-v1 recorder reported `status: pass`, `worktree_dirty: false`,
CuPy 14.2.0, CUDA runtime 12.9, driver 13.1, and compute capability 12.0. RK4
final/trajectory and static Cartesian, rotating Cartesian, and
helicity-projected split all returned CuPy `complex128` device arrays. The
largest maximum-absolute difference was `1.4376699353313202e-13`; the largest
norm error was `9.592326932761353e-14`, both below the fixed `2e-10` bounds.

The synchronized 32-state public-call medians were about 16-68 ms on GPU and
0.31-0.52 ms on CPU. They include validation and algorithm setup. They are
retained as honest diagnostics and demonstrate no speed advantage at this
size; no size crossover or equal-accuracy production benchmark is inferred.

The D-159 implementation report is committed as
`benchmarks/real-cuda-v0.3-b9de848.json`. D-160 commits the report emitted by
the actual manual `Real CUDA validation` workflow on the merged `main` commit
`4f7efaed992b05691e204f78536dd9f2c54abfd3` as
`benchmarks/real-cuda-v0.3-4f7efaed.json`. GitHub run `36887643743` passed on
the ephemeral `[self-hosted, linux, x64, gpu]` runner after normal-CI run
`36887536595` passed on the same commit. Both reports' schemas and source
bindings are executable contracts.

Before tagging, rerun the manual workflow on the exact final-version candidate
commit. The tag workflow independently repeats the same test and recorder and
attaches its accepted JSON to the GitHub Release. A queued job, skipped GPU
test, `status: error` diagnostic, source inspection, or an artifact from a
different source commit is not release acceptance evidence.

Further work may not change formula, precision, polarization, tolerance, or
renormalization semantics without a separate approved decision.
