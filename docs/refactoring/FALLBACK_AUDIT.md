# Explicit fallback audit

Date: 2026-08-09
Scope: src/rovibrational_excitation
Policy: D-021 in DECISIONS.md

## Policy

A requested physical model, numerical algorithm, backend, scale, or time grid
must either execute exactly as requested or raise before propagation. Missing
optional acceleration may use a slower implementation only when the numerical
backend and equations are unchanged and the choice is observable. Batch
failure isolation is permitted only when the failed case and traceback are
recorded.

## Resolved in the current change

| Area | Previous behavior | Resolution | Contract test |
|---|---|---|---|
| Nondimensional scales | Zero quantities created 1 fs, 1 Debye, 1e8 V/m, or a 1000 fs cap | Explicit ZeroField/inactive scales or precise error | test_strict_nondimensional_contracts.py |
| Time grid | auto_timestep could replace caller samples; target_accuracy could be accepted after removal | Both options are rejected at converter, solver, mixed-state, M-average, and simulation boundaries | test_strict_nondimensional_contracts.py; test_density_solver_contracts.py |
| Propagator kwargs | Misspelled or unsupported kwargs were silently ignored | Schrodinger, Liouville, and MixedState reject unknown options | contract solver tests |
| Dipole backend | backend='cupy' could return NumPy when CuPy was absent | RuntimeError; unknown backend names also raise | test_solver_contracts.py |
| Energy centering | Centering could change the returned absolute wavefunction phase | Exact global phase is restored | test_strict_nondimensional_contracts.py |

## P1: fix before API stabilization

1. core/nondimensional/analysis.py, strategies.py, and impl.py still export
   obsolete timestep recommendations, empirical coupling boundaries, and
   competing lambda-scaling strategies. The old scale methods now raise, but
   the namespace and implementation remain misleading. Remove these exports
   after retaining only tested equation verification and neutral reporting.
2. core/units/validators.py catches broad exceptions and converts validation
   failures to warnings. It also falls back from SI accessors to raw mu_axis
   attributes, which can bypass unit conversion. Split diagnostic warnings
   from strict propagation validation; strict mode must raise.
3. optimization/local.py silently disables eigenvalue lookahead on any
   exception and silently ignores target-weight indexing errors. These alter
   the optimization objective or update rule. Replace them with validated
   capability checks and explicit configuration errors after reference tests
   required by O-006 exist.
4. simulation/optimize_runner.py supplies mu0=1e-30 C m when absent. Dipole
   strength is physical input and should be required. Several target and
   plotting defaults also need a typed optimization configuration.
5. simulation/serialization.py interprets missing real or imaginary mapping
   fields as zero. Reject unknown keys and require an unambiguous complex
   number schema so misspellings cannot change polarization.

## User decisions required before changing physics-facing defaults

1. simulation/runner.py defaults pulse duration to half of the simulation
   window when neither duration nor pulse_duration is present. Decide whether
   duration becomes required.
2. linear and vibrational model builders default delta_omega, B, and alpha to
   zero and potential_type to harmonic. Some zeros are physically meaningful,
   but omission may also be accidental. Decide which parameters are mandatory
   per model dataclass.
3. initial_states defaults to the first basis state. Decide whether production
   simulation configuration must always state the initial condition.
4. backend, algorithm, sparse/dense, and renorm currently have documented
   computational defaults. These do not change the model Hamiltonian, but the
   typed PropagationOptions design should decide whether configs must state
   them explicitly.

## P2: cleanup and observability

- plots/plot_all.py and storage/checkpoint helpers catch broad exceptions.
  Plotting failures may remain non-fatal only if returned in result metadata;
  persistence failures must be surfaced.
- simulation/runner.py intentionally catches case failures for batch runs and
  writes tracebacks. Keep this behavior, but replace print-only reporting with
  a structured failure result.
- split-operator uses a pure NumPy implementation when Numba is unavailable.
  This is a performance fallback, not a physics/backend substitution. Expose
  acceleration availability in diagnostics and benchmarks.
- get_dipole_component_SI in propagation/utils.py is unused compatibility code
  with a raw-attribute fallback. Delete it with the Phase 1 legacy cleanup.

## Completion condition

The audit is complete when no public option is silently ignored; every
physics-bearing default is either accepted in a typed contract or required;
strict validation never degrades to warnings; and optional performance
fallbacks are observable without changing array backend or equations.
