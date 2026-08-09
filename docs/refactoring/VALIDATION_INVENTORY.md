# Legacy validation inventory

Last audited: 2026-08-05
Scope: P1.2-B read-only audit

## Purpose

This document records the proposed disposition of the standalone
`validation/` scripts and `tests/run_tests.py`. It is migration evidence,
not a source of accepted physical constants. Authoritative behavior lives in
`PHYSICS_CONTRACTS.md` and collected pytest tests.

The audit found no trusted golden array or domain reference value that exists
only in these scripts. They are not scientific gates: most use removed APIs,
run at import time, print or plot without assertions, catch failures without a
failing exit status, or contain invalid syntax.

No deletion was performed by this audit. Deletion requires review of the
explicit list below.

## Proposed disposition

| Path | Audit finding | Collected replacement or retained evidence | Proposal |
|---|---|---|---|
| `tests/run_tests.py` | subprocess wrapper around pytest; changes directory and hides the normal command | pytest configuration and commands in `tests/README.md` | delete |
| `validation/check_memory_eigendecomposition.py` | RSS/tracemalloc demonstration with no threshold; constructs the superseded circular interaction by averaging a matrix with its adjoint | `benchmarks/run_split_operator.py` times public split setup including eigendecomposition; split dense-memory limitation is documented | delete |
| `validation/core/check_core_basis.py` | prints basis size, bounds, and one index formula | basis unit tests and `tests/physics/test_linear_molecule_reference.py` | delete |
| `validation/core/check_core_states.py` | only instantiates a state vector | `tests/test_states.py` and state contract tests | delete |
| `validation/core/check_core_hamiltonian.py` | removed Hamiltonian import and a plot without assertions | basis/Hamiltonian tests and model physics references | delete |
| `validation/core/check_core_electric_field.py` | large interactive field/spectrum plot without acceptance criteria | `tests/test_electric_field.py`; future visual examples belong under examples or visualization docs | delete |
| `validation/core/check_core_propagator.py` | removed propagation API and plot-only LinMol run | propagator tests, solver invariants, and model references | delete |
| `validation/core/check_core__rk4_schrodinger.py` | removed low-level API and Gaussian pulse plot without analytic comparison | two-level matrix-exponential reference and RK4 convergence tests | delete |
| `validation/core/check_core__rk4_schrodinger_old.py` | duplicate obsolete diagnostic | same as preceding row | delete |
| `validation/core/check_core__splitop_schrodinger.py` | removed low-level API and plot-only Rabi example | split invariants, Cartesian/helicity physics tests, and split benchmark | delete |
| `validation/core/check_nondimensional_consistency.py` | old LinMol API; prints dimensional/nondimensional differences and saves a figure | LinMol and VibLadder dimensional/nondimensional physics references | delete |
| `validation/core/check_nondimensional_consistency_twolevel.py` | old API; repeats parity plot and estimates pulse response from peak-field Rabi frequency without a valid analytic pulse reference | `tests/physics/test_two_level_reference.py` | delete |
| `validation/core/debug_nondimensional_details.py` | old APIs; manually recomputes maximum energy span, off-diagonal dipole maximum, field maximum, and physical time | current scaling utilities plus dimensional/nondimensional reference tests; undecided scale-policy constants are listed below | delete |
| `validation/core/investigate_propagator_time_issue.py` | old APIs; investigates the historical factor-of-two time-grid defect | `tests/physics/test_solver_invariants.py` fixes field/state spacing and returned time | delete |
| `validation/core/analyze_energy_scale_problem.py` | exploratory printout; proposes arbitrary 1000 fs and 1 meV alternatives without a trusted scientific basis | the observed choices and unresolved constants are recorded below for Phase 4 | delete |
| `validation/core/fix_nondimensional_energy_scale.py` | invalid indentation; proposed `min_energy_diff_ratio=1e-6` cannot change the selected maximum gap when the ratio is below one | current maximum-gap behavior is visible in `core/nondimensional/utils.py`; policy remains a Phase 4 review item | delete |
| `validation/core/test_nondimensional_timestep_twolevel.py` | not collected; changes solver `dt` and step count without consistently resampling field stages, then ranks runtime with an arbitrary score | deterministic RK4 order and physical-time tests; future timestep policy needs an approved error target | delete |
| `validation/dipole/check_cache.py` | cache timing and interactive matrix plots without assertions | cache identity tests in `tests/test_dipole_linmol.py` | delete |
| `validation/dipole/check_dipole_builder.py` | harmonic/Morse plots; useful assertions are commented out; relies on former global Morse setup | builder, selection-rule, Morse-locality, and dense/sparse tests | delete |
| `validation/dipole/check_sparse_dipole.py` | machine-dependent RSS diagnostic; uses the superseded averaged circular interaction and partial sparse eigenspectra | dense/sparse operator parity and explicit split benchmark/docs | delete |
| `validation/dipole/test_unit_management.py` | print-based obsolete constructors; attempts Morse with zero anharmonicity | unit round trips and TwoLevel/VibLadder dipole tests; D-003 requires rejection | delete |
| `validation/simulation/check_simulation_runner.py` | invokes one example and prints runtime without checking output; path depends on launch directory | runner unit, contract, and integration tests | delete |

The tracked `validation/README.md` should be replaced with a short redirect to
pytest, the benchmark suite, and this inventory. The ignored PNG files present
in some working copies are generated outputs, not tracked reference assets.
They should not be deleted without a separate explicit request.

## Why the suspicious formulas are not migrated as tests

The proposed `min_energy_diff_ratio=1e-6` implementation first retains gaps
larger than `max_gap * ratio` and then chooses the maximum retained gap. For
any ratio below one, `max_gap` is itself retained, so the result is always
`max_gap`. The threshold is inert.

The timestep script is not a valid convergence test. It multiplies the RK4
step and reduces the number of steps, but continues to pass the original field
sample arrays to a kernel whose stages expect adjacent left/mid/right samples.
The compared calculations therefore do not consistently describe the same
time-dependent Hamiltonian.

The Rabi estimates use the maximum of a Gaussian carrier pulse as though it
were a constant resonant drive. They can be useful order-of-magnitude
diagnostics, but they are not exact reference solutions and have no asserted
tolerance.

## Nondimensionalization findings resolved by D-020

The 2026-08-06 P4.1 implementation resolves the policy questions exposed by
the stale validation scripts:

1. The 1000 fs time-scale cap was removed. The reference energy is derived from
   the full active generator or supplied explicitly.
2. Zero Hamiltonian span no longer creates a 1 fs energy. A driven gapless
   problem uses its interaction scale; a completely zero generator raises.
3. Zero dipole and zero field no longer create 1 Debye or 1e8 V/m values.
   ZeroField and inactive scale metadata express field-free intent.
4. The absolute 1e-20 J gap threshold and duplicate
   from_physical_system implementation were removed from execution; the legacy
   factory now raises migration guidance.
5. Weak/intermediate/strong boundaries 0.1 and 1.0 were removed from regime
   reporting. Numerical coefficient and physical coupling ratio are separate.
6. Empirical auto-timestep selection was removed from propagation and
   convenience helpers. The ElectricField grid is authoritative; future
   adaptation requires error tolerances and a distinct algorithm.
7. Energy extraction now uses eigvalsh on a finite Hermitian matrix. Active
   coupling dipoles are likewise validated as finite and Hermitian.

Replacement validation is collected in
tests/contracts/test_strict_nondimensional_contracts.py and the dimensional
versus nondimensional integration tests. The former checks non-diagonal H0,
operator-norm scaling, gapless drive, explicit ZeroField, invalid driven-zero
coupling, all-zero rejection, and absolute wavefunction phase restoration.

## Future validation policy

- Correctness and physics checks are collected pytest tests and fail by
  assertion.
- Reproducible timing and memory measurements live under `benchmarks/` or
  `tests/performance/` with their measured scope stated explicitly.
- Interactive plots belong in tested examples or visualization documentation,
  not in an uncollected validation tree.
- A new manual diagnostic must state its question, inputs, output artifact,
  and pass/fail interpretation; otherwise it is temporary local work and is
  not committed.
