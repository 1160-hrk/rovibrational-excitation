# Explicit fallback audit

Date: 2026-08-28
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
| Public propagator kwargs | Misspelled or unsupported kwargs were silently ignored | Public methods accept no `**kwargs`; typed options, coupling, and split mode are validated before work; private array adapters remain only for optimizer migration | public propagation contract tests |
| Dipole backend | backend='cupy' could return NumPy when CuPy was absent | RuntimeError; unknown backend names also raise | test_solver_contracts.py |
| Energy centering | Centering could change the returned absolute wavefunction phase | Exact global phase is restored | test_strict_nondimensional_contracts.py |
| Physical model inputs | duration and zero-valued constants could be omitted and silently defaulted | Model-specific constants, direct dipole mu0, vibrational potential type, units, and duration are required; explicit 0.0 remains valid | simulation and basis contract tests |
| Nondimensional API | 25 exports exposed competing lambda strategies and removed heuristics | One strict conversion path plus neutral reporting; legacy modules and wrappers removed | strict nondimensional contract tests |
| Spectroscopy policy | `optimized` silently chose paths, `sparse_threshold` was ignored, fixed response/Doppler cutoffs changed work, and requested device broadening was not applied | Explicit exact, approximate, and auto modes; required controls and execution report; grid-derived Doppler; requested device function applied | test_spectroscopy_reference.py |
| Spectroscopy polarization | Complex detection reused ket coefficients, `xyz` ignored its third component, and malformed vectors could normalize silently | Jones-bra detection conjugates coefficients; every ordered axis contributes; dimensions, finiteness, uniqueness, and nonzero norm are required | test_spectroscopy_reference.py |
| Spectroscopy pathway | `use_v_mask=True` silently kept `abs(delta_v) < 2` and missing V labels fell back to no mask | Required `pump_probe` (`V_i == V_j`) or `unfiltered`; discarded norm is reported; missing V labels raise; radiation/PFID remains unfiltered | test_spectroscopy_reference.py |
| Typed propagation defaults | Initial state and computational mode could be selected by defaults or inferred from input | D-026 requires explicit choices; P2.4-a makes algorithm, execution, trajectory, stride, scaling, and renormalization one required object | Phase 2 contract tests |
| Optimization time options | `dt_fs` hid the half-spaced field grid; `sample_stride` could thin states used by GRAPE/Krotov updates | D-029 requires `field_dt_fs`, exact divisibility, full internal trajectories, and output-only `output_stride`; local remains frozen under D-027 | optimization time contract and physics tests |
| Optimization plotting | Package plotting imported an examples-only FFT helper, NumPy `tlist or time` raised before plotting, and reconstructed trajectory times could disagree with retained endpoints | Package-local FFT helper, explicit `None` selection, and returned trajectory times | package import smoke check and full optimization contracts |
| External sampled field | Generated-field parameters could otherwise conflict with caller-owned samples, or a Cartesian waveform could be guessed into helicity data | `run_simulation_case` rejects every generation key; typed fields reject shape, finiteness, complex time-domain values, and kind mismatches; no trim/pad/interpolation/resampling/normalization occurs; helicity decomposition must be explicit | sampled-field and simulation contract tests |
| Numerical time-step adequacy | Removed `auto_timestep`/`target_accuracy` could otherwise be reintroduced as an implicit grid change | `assess_simulation_convergence` requires two caller-selected grids, a named observable, and explicit tolerance; it reports maximum absolute difference and never refines, retries, resamples, writes, or changes either calculation | `test_simulation_convergence.py` |
| Molecular symmetry preset | A molecule name could otherwise imply guessed constants, a generic model, or unverified nuclear-spin weights | D-052 resolves only an explicit alias to source-versioned symmetry rules; constants remain empty/required, unknown aliases raise, unsupported vibronic symmetry raises, and CH3F weights raise until signed-K symmetry adaptation | `test_molecular_symmetry_presets.py` |
| SymTop execution | The broken legacy route could be selected by optimization, while unverified CuPy or split execution could appear available | D-053 routes normal simulation through the independent production builder for NumPy dense/CSR RK4; CuPy, split operator, coherent all-isomer input, unknown presets, and optimization raise before numerical work | `test_symmetric_top_model_contracts.py`; `test_symmetric_top_reference.py` |
| Optimization documents | Unknown options and arbitrary runner kwargs could be ignored; invalid axes fell back to `xy`; YAML output/plot policy was not authoritative; plotting exceptions were printed | D-056 closes every document/algorithm key set, requires axes, restricts GRAPE to `xy`, removes runner kwargs, defines explicit output/plot precedence, and raises top-level requested plotting failures | `test_optimization_config_contracts.py`; optimization time/reference contracts |
| Optimization option values | Numeric/string values were coerced, duplicate axes were accepted, Local mode typos selected another branch, lookahead/weight/cost errors were suppressed, and spectral updates applied a repair floor | D-057 requires exact types/enums/distinct axes, surfaces requested Local failures, validates finite nonnegative spectral alpha, and divides directly by `1+alpha` | `test_optimization_option_contracts.py`; Local propagation contracts |

## P1: fix before API stabilization

1. Resolved by D-043. `core/units/validators.py` now requires canonical
   accessors and structural consistency and raises the original failure. The
   context ranges, fixed 1000 fs estimate, broad exception downgrade, and raw
   `mu_axis` fallback are deleted. Numerical adequacy remains an explicit
   convergence report.
2. Resolved by D-057. Local resolves eigenvalues only when lookahead is
   requested and raises on failure or invalid shape/value. Target-weight and
   display-cost exceptions are no longer suppressed.
3. io/serialization.py interprets missing real or imaginary mapping
   fields as zero. Reject unknown keys and require an unambiguous complex
   number schema so misspellings cannot change polarization.

## Resolved Phase 2 decisions

D-026 requires typed production configuration to state the initial condition,
backend, algorithm, dense/CSR storage, and renormalization policy explicitly.
Legacy defaults remain only as characterized migration adapters and must not be
copied into the final public typed API.

## P2: cleanup and observability

- visualization/plot_all.py still catches errors inside optional spectrum and
  spectrogram branches. The optimization runner no longer catches the top-level
  plot call. Optional branch failures may remain non-fatal only if returned in
  result metadata; persistence failures must be surfaced.
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
