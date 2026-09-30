# Phase 5 numerical-engine acceptance audit

Last verified: 2026-09-30
Checkpoint: P5.5-b / D-145 device-native CuPy RK4 and split; real CUDA pending

## Scope

This audit compares the current implementation and executable tests with every
Phase 5 acceptance row. CPU-backed doubles can verify dispatch and calculation
graphs, but they are not CUDA numerical or performance evidence.

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
| Real CUDA parity and performance | Not verified | three new GPU cases plus existing GPU suite skip here |

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
| NumPy/CuPy numerical parity and shapes | Collected, not verified here | CUDA tests skip without hardware |
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

Phase 5 remains **in progress**. Source-level device residency is implemented,
but the final v0.3.0 tag still requires successful real-GPU evidence.

## Remaining CUDA closure work

1. run RK4 final/trajectory and static Cartesian, rotating Cartesian, and
   helicity-projected split cases on real CUDA;
2. verify CPU/GPU parity, norm, shape, dtype, backend identity, and actual
   transfer behavior;
3. benchmark setup, propagation, and end-to-end execution before making a GPU
   performance claim;
4. archive the hardware/software identity and benchmark evidence;
5. rerun the complete CPU/release gates at the accepted CUDA checkpoint.

Further work may not change formula, precision, polarization, tolerance, or
renormalization semantics without a separate approved decision.
