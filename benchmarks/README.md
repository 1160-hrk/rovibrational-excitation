# Propagation benchmark baseline

This directory contains non-blocking performance records. They are diagnostic
artifacts, not ordinary correctness gates: noisy wall-clock values must not make
the test suite fail.

## v0.2.10 CPU baseline

From the repository root, run:

~~~bash
python benchmarks/run_baseline.py
~~~

The command writes `benchmarks/baseline-v0.2.10.json`. Its default protocol is:

- deterministic TwoLevel, 16-level VibLadder, and 18-state M-resolved LinMol
  Schrödinger propagation;
- NumPy dense and SciPy sparse RK4 paths for every model;
- one dense TwoLevel Liouville case for trace and Hermiticity error;
- a 4001-point electric-field grid, corresponding to 2000 propagation steps;
- one untimed warmup per workload, followed by seven timed repetitions;
- `time.perf_counter_ns` timing and the median as the headline value;
- propagation only in the timed region; basis, Hamiltonian, dipole, field, and
  initial-state construction are excluded.

Use explicit options for a shorter diagnostic run without overwriting the
committed baseline:

~~~bash
python benchmarks/run_baseline.py \
  --field-points 101 \
  --repeats 3 \
  --output /tmp/rve-benchmark.json
~~~

## Recorded v0.2.10 result

The committed artifact was measured from clean source commit `3b081e1` on
CPython 3.12.12, Linux aarch64, NumPy 2.3.5, SciPy 1.17.0, and Numba 0.63.1.

| Workload | Dimension | Median (ms) | Norm/trace error | Trajectory (KiB) |
|---|---:|---:|---:|---:|
| TwoLevel dense Schrödinger | 2 | 0.380 | `1.33e-15` norm | 62.5 |
| TwoLevel sparse Schrödinger | 2 | 39.040 | `1.33e-15` norm | 62.5 |
| VibLadder dense Schrödinger | 16 | 0.849 | `2.22e-15` norm | 500.2 |
| VibLadder sparse Schrödinger | 16 | 41.178 | `2.33e-15` norm | 500.2 |
| LinMol dense Schrödinger | 18 | 0.968 | `5.55e-16` norm | 562.8 |
| LinMol sparse Schrödinger | 18 | 43.588 | `5.55e-16` norm | 562.8 |
| TwoLevel dense Liouville | 2 | 2.079 | `2.94e-18` trace | 125.1 |

Dense/sparse final-state L2 differences are between `5.55e-17` and
`1.57e-16`. Sparse is 45–103 times slower for these deliberately small
systems; this is a migration baseline, not a claim that sparse storage is
advantageous below a crossover dimension.

## Interpreting memory and GPU fields

`peak_trajectory_memory_estimate_bytes` is the allocated size of the returned
`complex128` trajectory. It intentionally excludes operators, temporary RK4
vectors, JIT/runtime allocations, Python overhead, and process RSS. The matching
`trajectory_array_bytes` field checks the estimate against the actual returned
array.

The script never claims a CuPy result from package availability alone. The
artifact records GPU timing as not run unless a dedicated measurement is made
on a real CUDA device. The committed v0.2.10 artifact is the NumPy CPU baseline;
CUDA remains separately unverified.

## Numba CSR RK4 result

The post-change artifact was recorded from clean source commit `6e154ec`:

~~~bash
python benchmarks/run_baseline.py \
  --artifact numba-csr-v0.2.10 \
  --output benchmarks/numba-csr-v0.2.10.json
~~~

| Sparse workload | Previous SciPy (ms) | Numba CSR (ms) | Speedup |
|---|---:|---:|---:|
| TwoLevel | 39.040 | 0.388 | 100.6x |
| VibLadder, dimension 16 | 41.178 | 0.818 | 50.4x |
| LinMol, dimension 18 | 43.588 | 0.974 | 44.8x |

The largest dense/sparse final-state L2 difference is `1.11e-16`; the largest
final norm error is `2.34e-15`. The Liouville trace reference is unchanged at
`2.94e-18`.

The old workload labelled dense accepted dense input but internally scanned it
into CSR. The new dense measurement executes the actual dense kernel. For the
structurally sparse 16- and 18-dimensional models, explicit Numba CSR is 1.80
and 1.97 times faster than the honest dense path respectively. A final-only
tridiagonal diagnostic with 200 RK4 steps measured 5.67x speedup at dimension
64 and 24.77x at dimension 256, with dense/sparse final L2 differences below
`5.56e-17`.

## Regression policy

Wall time is environment-dependent, so the artifact contains no absolute test
threshold. Compare medians on the same machine and dependency stack. A slowdown
larger than 10% requires investigation under the refactoring policy, but is not
automatically a correctness failure. Dense/sparse final-state differences,
norm/trace error, dependency versions, source commit, and worktree state are
stored so comparisons are auditable.
