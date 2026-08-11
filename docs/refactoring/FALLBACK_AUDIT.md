# Explicit fallback audit

Date: 2026-08-10
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
| Physical model inputs | duration and zero-valued constants could be omitted and silently defaulted | Model-specific constants, direct dipole mu0, vibrational potential type, units, and duration are required; explicit 0.0 remains valid | simulation and basis contract tests |
| Nondimensional API | 25 exports exposed competing lambda strategies and removed heuristics | One strict conversion path plus neutral reporting; legacy modules and wrappers removed | strict nondimensional contract tests |
| Spectroscopy policy | `optimized` silently chose paths, `sparse_threshold` was ignored, fixed response/Doppler cutoffs changed work, and requested device broadening was not applied | Explicit exact, approximate, and auto modes; required controls and execution report; grid-derived Doppler; requested device function applied | test_spectroscopy_reference.py |
| Spectroscopy polarization | Complex detection reused ket coefficients, `xyz` ignored its third component, and malformed vectors could normalize silently | Jones-bra detection conjugates coefficients; every ordered axis contributes; dimensions, finiteness, uniqueness, and nonzero norm are required | test_spectroscopy_reference.py |
| Spectroscopy pathway | `use_v_mask=True` silently kept `abs(delta_v) < 2` and missing V labels fell back to no mask | Required `pump_probe` (`V_i == V_j`) or `unfiltered`; discarded norm is reported; missing V labels raise; radiation/PFID remains unfiltered | test_spectroscopy_reference.py |

## P1: fix before API stabilization

1. core/units/validators.py catches broad exceptions and converts validation
   failures to warnings. It also falls back from SI accessors to raw mu_axis
   attributes, which can bypass unit conversion. Split diagnostic warnings
   from strict propagation validation; strict mode must raise.
2. optimization/local.py silently disables eigenvalue lookahead on any
   exception and silently ignores target-weight indexing errors. These alter
   the optimization objective or update rule. Replace them with validated
   capability checks and explicit configuration errors after reference tests
   required by O-006 exist.
3. simulation/optimize_runner.py now requires basis constants, dipole value and
   unit, potential type, and Krotov pulse duration. Target and plotting options
   still need a typed optimization configuration after O-006 reference tests.
4. simulation/serialization.py interprets missing real or imaginary mapping
   fields as zero. Reject unknown keys and require an unambiguous complex
   number schema so misspellings cannot change polarization.

## User decisions required before changing physics-facing defaults

1. initial_states defaults to the first basis state. Decide whether production
   simulation configuration must always state the initial condition.
2. backend, algorithm, sparse/dense, and renorm currently have documented
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
