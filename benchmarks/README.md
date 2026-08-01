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

## Regression policy

Wall time is environment-dependent, so the artifact contains no absolute test
threshold. Compare medians on the same machine and dependency stack. A slowdown
larger than 10% requires investigation under the refactoring policy, but is not
automatically a correctness failure. Dense/sparse final-state differences,
norm/trace error, dependency versions, source commit, and worktree state are
stored so comparisons are auditable.
