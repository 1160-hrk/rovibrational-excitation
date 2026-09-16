# Phase 5 numerical-engine acceptance audit

Last verified: 2026-09-16
Checkpoint: P5.4-a / D-062 acceptance evidence; D-071 CUDA release scope

## Scope

This audit compares the current implementation and executable tests with every
Phase 5 acceptance row. It does not authorize a physics change or treat a
skipped CUDA test as evidence.

## P5.1 RK4

| Requirement | Status | Evidence |
|---|---|---|
| NumPy dense and CSR kernels are separate | Verified | D-018; `test_sparse_rk4_reference.py` |
| CSR selection is explicit and does not prune values | Verified | D-018; sparse reference contracts |
| Liouville validation and kernel are separate | Verified | D-060; `test_liouville_kernel_separation_contracts.py` |
| Liouville endpoint reuse preserves the old loop | Verified exactly | D-061; final and trajectory `np.array_equal` comparisons |
| Field left/mid/right indices and `-mu E` agree | Verified | solver physics and density contracts |
| NumPy performance is measured | Verified | Numba CSR and Liouville endpoint-reuse artifacts |
| CuPy RK4 remains device-native internally | Not satisfied | `_rk4_gpu` calls `.get()` before the typed result boundary |
| Real CUDA parity | Not verified | ten GPU tests are collected but skipped here |

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
| CuPy result remains device-native internally | Not satisfied | both split CuPy helpers call `cp.asnumpy` |

The direct NumPy/Numba dependency is now honest. Numba is a required project
dependency, so the dead import-time dummy-decorator fallback was removed.
Missing Numba now raises the ordinary dependency error instead of silently
changing execution mode.

## P5.3 backend transfer policy

D-026 and D-039 already decide the public policy:

1. `PropagationResult.state` belongs to the selected backend.
2. An existing device state passes through finalization by identity.
3. Host conversion occurs only through explicit `PropagationResult.to_numpy()`.
4. Persistence and host-only analysis call that boundary explicitly.

The typed boundary satisfies these rules and has executable device-like
contracts. The current CuPy numerical adapters do not: RK4 performs
`device -> host` with `.get()`, split performs it with `cp.asnumpy`, and
`finalize_propagation_result` must then perform `host -> device` to honor the
public result type. This repeated transfer is observable source debt, not a
different public contract.

## Phase 5 acceptance disposition

| Acceptance condition | Status |
|---|---|
| Every advertised CPU capability executes | Pass |
| CPU physics baselines | Pass |
| No silent algorithm/backend/storage fallback | Pass |
| NumPy performance and trajectory memory are documented | Pass |
| Existing device state is retained by the result boundary | Pass |
| Real CUDA RK4 and split parity | Pending real GPU |
| No repeated transfer on CuPy execution | Fail until CUDA migration |

Phase 5 must remain **in progress**. CPU work is accepted; CUDA closure is not.
Under D-071, CUDA is an explicit v0.3 supported target rather than a deferred
extension. The external hardware blocker does not prevent independently tested
Phase 6 CPU model consolidation, but Phase 5 must not be marked complete and
the final v0.3.0 tag must not be created before the required real-GPU run.

## Required CUDA closure unit

Implementation may proceed without local CUDA, but closure requires a real GPU:

1. separate Schrödinger RK4 and split CuPy kernels from mixed CPU modules;
2. keep prepared operators, fields, trajectories, and final states on device;
3. return CuPy arrays directly from low-level GPU functions;
4. prohibit `.get()` and `cp.asnumpy` before an explicit host boundary;
5. test static Cartesian, rotating Cartesian, and helicity-projected split;
6. test trajectory and final-only shape, norm, numerical CPU/GPU parity, and
   backend-native result identity;
7. benchmark transfer count, setup, propagation, and end-to-end execution;
8. rerun the complete CPU suite to ensure the CUDA separation changes no NumPy
   calculation.

No new formula, tolerance, precision, polarization convention, or
renormalization rule is permitted in that unit.
