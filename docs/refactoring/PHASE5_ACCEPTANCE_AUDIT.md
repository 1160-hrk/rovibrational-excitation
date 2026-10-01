# Phase 5 numerical-engine acceptance audit

Last verified: 2026-10-01
Checkpoint: P5.5-g / D-158 clean-source CUDA evidence environment; evidence rerun pending

## Scope

This audit compares the current implementation and executable tests with every
Phase 5 acceptance row. CPU-backed doubles can verify dispatch and calculation
graphs, but they are not CUDA numerical or performance evidence.

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
| CuPy RK4 graph matches CPU and remains device-native | Implemented; real GPU pending | D-144; `test_cuda_rk4_contracts.py` |
| Real CUDA parity and performance | Harness complete; not verified | D-146; 15 GPU cases skip here |

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
| NumPy/CuPy numerical parity and shapes | Harness complete; not verified here | D-146; CUDA tests skip without hardware |
| CuPy split graph and result remain device-native | Implemented; real GPU pending | D-145; `test_cuda_split_contracts.py` |

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
| CuPy RK4 source graph and device-native return | Implemented, real GPU pending |
| CuPy split source graph and device-native return | Implemented, real GPU pending |
| Real CUDA RK4 and split parity/performance | Pending real GPU |
| No array round trip before the explicit host boundary | Source-verified; real GPU pending |

Phase 5 remains **in progress**. Source-level device residency and the strict
evidence recorder are implemented, but the final v0.3.0 tag still requires an
accepted report produced on real hardware.

## Remaining CUDA closure work

D-146 fixes the required five cases, schema, numerical bounds, synchronized
timing scopes, hardware/software/source identity, failure behavior, and
artifact retention. D-155 makes the pyproject `gpu` extra provision the
complete CUDA 12 user-space environment. D-156 additionally makes the manual
workflow install `dev,io,plot,gpu`, matching the tag-time job so global pytest
collection has every optional dependency. D-157 then replaces the real-device
LinMol failure at unsupported `cupy.vectorize` with the approved,
formula-preserving device-array implementation; all six axis/potential
combinations match the existing CPU reference at `2e-15`. D-158 records that
all 15 GPU-marked tests then pass on the target RTX 5070 Ti. Its first evidence
run correctly rejected the unignored in-tree `.venv-cuda/` as dirty source;
`.venv*/` is now ignored without changing the strict provenance check. Run the
manual `Real CUDA validation` workflow before tagging. Accept Phase 5 only if
its schema-v1 report has `status: pass`, then
rerun the complete CPU/release gates at that exact commit.

The tag workflow independently repeats the same test and recorder and attaches
the accepted JSON to the GitHub Release. A queued job, skipped GPU test,
`status: error` diagnostic, or source inspection is not acceptance evidence.

Further work may not change formula, precision, polarization, tolerance, or
renormalization semantics without a separate approved decision.
