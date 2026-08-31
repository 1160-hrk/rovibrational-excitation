# API and entry-point inventory

Last verified: 2026-08-31
Scope: Phase 0 task P0.1
Original inventory baseline: `613ce93`
Latest API checkpoint: D-048 canonical low-level field construction

This document freezes the entry points that exist before the v0.3 package
migration. It is an inventory, not a promise of backward compatibility.
Decision D-001 permits breaking these paths; the disposition below prevents us
from deleting or moving them accidentally before their callers and scientific
behavior are understood.

## 1. Disposition vocabulary

| Disposition | Meaning |
|---|---|
| **target public** | Part of the intended v0.3 supported API, possibly at a new path |
| **temporary public** | Currently importable and used, but replaced or moved before v0.3 |
| **internal** | Required by package orchestration; not a supported user API |
| **delete** | Remove after its replacement and characterization tests exist |

The proposed root namespace is recorded under O-008 in `DECISIONS.md`. It is a
working proposal, not yet an accepted API decision.

## 2. Package root

### 2.1 Names declared in `rovibrational_excitation.__all__`

| Current root name | Observed callers | Target path or replacement | Disposition |
|---|---|---|---|
| `LinMolBasis` | README and direct subpackage examples use the concept; no source file imports it from root | `models.linear_molecule.LinearMoleculeModel` and typed parameters | temporary public |
| `Hamiltonian` | Root re-export remains; internal callers now use the target module | `core.operators.Hamiltonian` | temporary root re-export pending O-008; target submodule complete |
| `StateVector` | Tests and examples use `core.basis.StateVector` | `core.states.StateVector` | temporary public |
| `DensityMatrix` | Tests use `core.basis.DensityMatrix` | `core.states.DensityMatrix` | temporary public |
| `ElectricField` | Simulation, optimization, tests, and examples | root re-export backed by `fields.ElectricField` | target public; D-047/D-048 require explicit input units and canonical fs/V/m storage |
| `LinMolDipoleMatrix` | Tests and examples use its model subpackage | constructed by `models.linear_molecule`; advanced class remains under that model | temporary public |
| `AbsorbanceCalculator` | spectroscopy examples and tests | `spectroscopy.AbsorbanceCalculator` or decomposed facade | temporary public at root; target public in subpackage |
| `ExperimentalConditions` | spectroscopy examples and tests | `spectroscopy.ExperimentalConditions` | D-046 requires exact value/unit pairs and frozen canonical fields; temporary public at root; target public in subpackage |
| `create_calculator_from_params` | spectroscopy examples and tests | typed spectroscopy constructor under `spectroscopy` | D-046 requires and forwards every condition unit; temporary public at root; target public in subpackage |

Every current root `__all__` name therefore has an explicit disposition. Only
`ElectricField` is proposed to remain a root re-export.

### 2.2 Other accessible root attributes

`__version__` and `__author__` are accessible but absent from `__all__`.
`core`, `dipole`, `fields`, `simulation`, `spectroscopy`, and
`visualization` are bound by eager root imports. Each now has an explicit
package initializer. `simulation` narrowly exports `run_simulation_case`.

| Current name | Target | Disposition |
|---|---|---|
| `__version__` | root metadata | target public; add to `__all__` |
| `__author__` | package metadata only | internal; do not promise as API |
| `core` | explicit `core/__init__.py` with narrow exports | target public subpackage |
| `fields` | explicit field construction, envelopes, and modulation package | target public subpackage; `ElectricField` remains a temporary root re-export pending O-008 |
| `dipole` | functionality moves under model ownership | temporary public; delete package after migration |
| `visualization` | explicit target package with module-level plotting helpers | target public subpackage; root import does not load optional Matplotlib |
| `simulation` | typed `simulation` workflows | target public subpackage |
| `spectroscopy` | decomposed `spectroscopy` package | target public subpackage |

The package root docstring is already stale: it demonstrates
`LinearResponseCalculator`, `SpectroscopyParameters`,
`calculate_absorption_spectrum`, `prepare_variables`, and
`absorbance_spectrum_for_loop`, none of which is exported by the current root.
It must be replaced in Phase 8, after the target facade exists.

No file under `src/` currently uses `from rovibrational_excitation import ...`
or imports the root alias. Newly edited internal modules must keep using their
owning subpackage paths.

## 3. Subpackage exports

This is the complete set of explicit subpackage `__all__` declarations at the
baseline. Direct imports from modules not listed here remain possible, but are
not treated as intentional API.

### 3.1 Core state and operator layer

| Current package | Exact exported names | Target | Disposition |
|---|---|---|---|
| `core` | no re-exported names yet | narrow generic state/operator/time/unit surface pending O-008 | target public package created in P3.1-a |
| `core.operators` | `Hamiltonian` is directly importable; no package `__all__` yet | generic unit-aware operator owner | target public module; root re-export remains temporary |
| `core.basis` | `BasisBase`, `LinMolBasis`, `TwoLevelBasis`, `VibLadderBasis`, `SymTopBasis`, `StateVector`, `DensityMatrix` | generic states to `core`; model bases to their `models.*` owners | temporary public; `Hamiltonian` removed in P3.1-a |
| `core.units` | `PhysicalConstants`, `UnitConverter`, `Frequency`, `TimeQuantity`, `DipoleMoment`, `ElectricFieldAmplitude`, `GroupDelayDispersion`, `ThirdOrderDispersion`, `converter`, `UnitValidator`, `validator` | immutable constants and frozen explicit conversion boundaries under `core.units`; typed config handles parameter conversion | target public quantity and pure-conversion surface; generic processor deleted by D-045 |
| `core.time` | `TimeGrid`, `FIELD_INTERVALS_PER_PROPAGATION_STEP` | immutable time invariant under `core.time` | target public module; root re-export remains subject to O-008 |
| `core.execution` | `ArrayBackend`, `MatrixStorage`, `ExecutionPolicy` | one explicit backend/storage choice | target public module; normal runner/model wiring complete in P2.3-b |
| `core.states` | `PureState`, `IncoherentEnsemble`, `DensityState` | explicit immutable initial-state kinds | target public module; all propagator facades migrated in P2.2 |
| `core.validation` | `NUMERICAL_VALIDATION_EPSILON_FACTOR`, `density_matrix_tolerance`, `validate_density_matrix_properties`, `validate_density_matrix_problem`, `validate_wavefunction_problem` | generic numerical input validation owner reached in P3.1-e | internal; no root re-export |

`core/__init__.py` now makes the target package explicit but intentionally
re-exports nothing until O-008 fixes the supported convenience surface.

### 3.2 Fields

| Current package | Exact exported names | Target | Disposition |
|---|---|---|---|
| `fields` | `ScalarField`, `CartesianField`, `SampledField`, `ElectricField`, `ZeroField`, `gaussian`, `lorentzian`, `voigt`, `gaussian_fwhm`, `lorentzian_fwhm`, `voigt_fwhm`, `apply_sinusoidal_mod`, `apply_dispersion`, `get_mod_spectrum_from_bin_setting` | target owner reached in P3.1-b | target public subpackage; only `ElectricField`, `gaussian`, and `gaussian_fwhm` proposed at root |

`ScalarField` and `CartesianField` are the normal-simulation typed values.
They defensively copy real V/m samples and own one canonical `TimeGrid`.
`ElectricField.from_time_grid` remains the generated-pulse constructor and its
legacy array constructor remains available for kernels, optimization code, and
tests that have not yet migrated.
`add_arbitrary_Efield` requires a direct field-amplitude unit and stores V/m;
intensity labels are inapplicable to signed arbitrary samples. Direct
construction requires `time_units`, converts to fs, and has no constructor
field-unit selector because stored fields are always V/m. Generated-pulse unit
defaults remain P4.3-g migration work.

The modulation helpers remain public under `fields` only if Phase 4/5 tests
establish their units and sampling contracts. Until then their stability is
temporary.

### 3.3 Propagation and scaling

| Current package | Exact exported names | Target | Disposition |
|---|---|---|---|
| `dynamics` | `PropagatorBase`, `PropagationDirection`, `SchrodingerPropagator`, `LiouvillePropagator`, `MixedStatePropagator`, `PropagatorFactory`, `PropagationOptions`, `RenormalizationPolicy`, `ScalingMode`, `Axis`, `CouplingMode`, `CouplingSpec`, `PropagationProblem`, `PropagationState`, `SystemModel`, `PropagationResult` | target typed problem/options/result owner reached in P3.1-c | typed public package; factory class deletes after replacement |
| `dynamics.algorithms` | `rk4_lvne`, `rk4_lvne_traj`, `rk4_schrodinger`, `splitop_schrodinger` | `dynamics.solvers` private kernels | internal |
| `dynamics.algorithms.rk4` | `rk4_lvne`, `rk4_lvne_traj`, `rk4_schrodinger` | `dynamics.solvers.rk4` | internal |
| `dynamics.algorithms.split_operator` | `splitop_schrodinger` | `dynamics.solvers.split_operator` | internal |
| `dynamics.scaling` | `NondimensionalizationScales`, `ScaleValue`, `nondimensionalize_system`, `nondimensionalize_with_SI_base_units`, `nondimensionalize_from_objects`, `determine_SI_based_scales`, `create_dimensionless_time_array`, `analyze_regime`, `dimensionalize_wavefunction`, `get_physical_time` | target owner reached in P3.1-d with one explicit scaling representation | target public subpackage; old `core.nondimensional` path removed |

The former 25-name surface was reduced under D-022 after dimensional-equivalence
and strict-generator tests identified the production path. Compatibility
wrappers, competing lambda strategies, heuristic verification, auto-timestep,
and demo factories are deleted rather than deprecated.

### 3.4 Models and dipoles

| Current package | Exact exported names | Target | Disposition |
|---|---|---|---|
| `dipole` | `LinMolDipoleMatrix`, `TwoLevelDipoleMatrix`, `VibLadderDipoleMatrix`, `SymTopDipoleMatrix`, `create_dipole_matrix` | respective `models.*` packages | temporary public; generic factory becomes internal |
| `dipole.linmol` | `LinMolDipoleMatrix` | `models.linear_molecule` | temporary public |
| `dipole.twolevel` | `TwoLevelDipoleMatrix` | `models.two_level` | temporary public |
| `dipole.viblad` | `VibLadderDipoleMatrix` | `models.vib_ladder` | temporary public |
| `dipole.symtop` | `SymTopDipoleMatrix` | `models.symmetric_top` | experimental temporary public pending O-005 |
| `dipole.rot` | `tdm_jm_x`, `tdm_jm_y`, `tdm_jm_z`, `tdm_j` | private linear/symmetric-top kernels | internal |
| `dipole.vib` | `tdm_vib_harm`, `tdm_vib_morse`, `omega01_domega_to_N`, `validate_morse_v_max` | private/shared vibration kernels under model ownership | internal |
| `models` | `CouplingSpec`, `LinMolRepresentation`, `ModelComponents`, `build_model`; model validation remains explicit under `models.validation` | flat target-owner facade reached in P3.1-f; model validation ownership reached in P3.2-b; model-specific package split pending Phase 6 | internal transition facade; `build_model` requires `ExecutionPolicy`, `basis_type`, and `initial_states`; LinMol also requires `representation`, and `m_resolved` requires `axes`. Normal-simulation `use_M` is removed under D-041 |

`SymTopBasis` and `SymTopDipoleMatrix` are importable, but the primary
simulation `build_model` registry supports only `linmol`, `twolevel`, and
`vibladder`. SymTop must not be advertised as stable until O-005 is resolved.

### 3.5 Persistence

| Current package | Exact exported names | Target | Disposition |
|---|---|---|---|
| `io` | `CheckpointManager`, `deserialize_polarization`, `json_safe`, `make_results_root`, `update_summary` | target persistence owner reached in P3.1-g | internal transition facade; schema versioning and manager/persistence separation remain deferred |

The former `simulation.{checkpoint,serialization,storage}` modules were removed
without compatibility shims. P3.1-g changes ownership only: checkpoint and
summary filenames, JSON/CSV/NPZ representations, deduplication, corruption
status, and overwrite behavior remain unchanged and unversioned.

### 3.6 Visualization

| Current package | Exact exported names | Target | Disposition |
|---|---|---|---|
| `visualization` | no package-level function exports; functions remain explicit under `visualization.plot_all`, `visualization.plot_electric_field`, `visualization.plot_electric_field_vector`, `visualization.plot_population`, and `visualization.spectrogram` | target visualization owner reached in P3.1-h | target public subpackage; root exposes only the package and keeps Matplotlib lazy |

The former `plots` namespace was removed without a shim. The five implementation
files are exact renames. Package-level function aliases are deliberately omitted
because names such as `plot_all` collide with Python submodule attributes;
callers import functions from their owning modules.

### 3.7 Optimization and spectroscopy

| Current package | Exact exported names | Target | Disposition |
|---|---|---|---|
| `optimization` | `run_local_optimization`, `run_krotov_optimization`, `run_grape_optimization`, `ALGO_REGISTRY` | typed functions under `optimization`; private registry | run functions target public in subpackage; registry internal |
| `spectroscopy` | `AbsorbanceCalculator`, `ExperimentalConditions`, `SpectroscopyCalculationReport`, `create_calculator_from_params` | decomposed spectroscopy modules with a tested facade | target public in subpackage; numerical/polarization/pathway policy accepted by D-023 through D-025, strict units by D-046, scientific references pending O-007 |

`cli/__init__.py` remains empty. `simulation.__all__` contains only
`run_simulation_case`; specialized validation, convergence, sweep, and runner
helpers remain module-level internals. `visualization` has an explicit empty
initializer so root import does not load Matplotlib. `io` has a narrow facade
but is not re-exported from the package root.

## 4. Console scripts and configuration routes

| Script | Current route | Input and construction path | Target | Disposition |
|---|---|---|---|---|
| `rve-simulate` | `cli.simulate:main` | Python file executed by `simulation.config.load_params_file` -> unchanged value/unit mapping -> iterable sweep expansion -> per-case validation -> `models.build_model` -> `SchrodingerPropagator` | versioned typed simulation config and one shared model/field builder | target public command; replace input contract |
| `rve-optimize` | `cli.optimize:main` | YAML `safe_load` -> dotted overrides -> private `_build_basis` and `_build_dipole` -> `optimization.ALGO_REGISTRY` -> algorithm function | versioned typed optimization config reusing the same model/field builders | target public command; replace orchestration internals |

Both command names should remain. Backward compatibility for current config
files is not required, but result and config schemas must be explicitly
versioned so historical calculations remain interpretable.

### 4.1 Current simulation parameter path

1. `rve-simulate PARAMFILE` calls `run_all_with_checkpoint`.
2. `load_params_file` executes arbitrary Python and collects non-dunder values,
   excluding imported module objects. Other helper names remain visible and
   must be valid schema keys rather than disappearing implicitly.
3. `expand_cases` treats most iterable values as sweep dimensions; only
   `polarization` and `initial_states` are fixed-value exceptions.
4. `validate_simulation_case` runs only after expansion. Its closed key set
   rejects unknown names and model/field/algorithm-inapplicable parameters.
   Generated cases require named envelope and modulation discriminators. It
   also requires algorithm, backend, storage, trajectory, stride, scaling, and
   renormalization choices, constructs one `PropagationOptions`, and performs
   capability preflight. Generated-field values are converted exactly once by
   frozen `GeneratedFieldParameters`; the caller mapping remains unchanged.
5. The immutable `SimulationCase` owns a frozen model schema and validated
   `ExecutionPolicy`; model construction returns `ModelComponents` plus
   scalar/Cartesian coupling metadata without re-reading the raw mapping.
6. `runner._run_one` consumes the canonical typed time grid, generates the legacy
   pulse unchanged, and freezes its exact values as `ScalarField` or
   `CartesianField` before propagation.
7. Python callers may instead use
   `simulation.runner.run_simulation_case(params, field=...)`; generated-field
   keys are then rejected, and the injected field owns the exact `TimeGrid`.
8. The same options object reaches model construction and propagation before
   writing an unversioned NPZ/JSON result. The JSON preserves submitted value/unit
   pairs; scalar `E` is stored one-dimensional
   and Cartesian `E` is stored with shape `(n_samples, 2)`.

Accuracy assessment is a separate public application service at
`simulation.convergence.assess_simulation_convergence`. It compares two
otherwise identical generated or externally injected cases and returns an
immutable `ConvergenceReport`; it is not part of config loading and cannot
select, repair, or replace a time grid. `ConvergenceConfigurationError` reports
invalid comparisons before propagation.

This entire route is temporary. Python-file execution, implicit sweep inference,
and unversioned output are not part of the target contract.

### 4.2 Current optimization path and divergence

`run_from_config` accepts YAML, a `Path`, or a dictionary. It owns a second set
of basis and dipole builders instead of using `models.build_model`.
The parameter names also differ (`omega_cm` versus `omega_rad_phz`, for
example). It silently changes an unknown dipole unit to `C*m` and an unknown
potential type to `harmonic`. These fallbacks violate the explicit-validation
policy and must become errors when the typed config is introduced. They are
documented here only; P0.1 does not change calculation behavior.

D-029 migrates the characterized GRAPE/Krotov time behavior to canonical
`TimeGrid`. Configuration now states the historical half-spaced field interval
directly as `field_dt_fs`; `output_stride` cannot alter optimizer-internal
trajectories. D-028 provides the explicit dimensional NumPy RK4 backward route
used by the Krotov costate. The local optimizer instead retains the versioned
`LocalOptimizerLegacyGridV1` contract accepted in D-027: its existing
`np.arange` storage array, shared-boundary ownership, segment slices, and
floor-based RK4 consumption are not reconstructed through canonical `TimeGrid`.
Only the final RK4 view is restricted to the odd prefix that the legacy kernel
already consumed. Normal and optimization workflows may still share typed
propagation and result boundaries without sharing time-array construction.

The runner catches every plotting exception and returns a nominally successful
optimization. Phase 7 must distinguish an optimization result from optional
visualization failure.

## 5. Factories and registries

| Current entry | Dispatch key | Current callers | Target | Disposition |
|---|---|---|---|---|
| `models.build_model` | `basis_type`, plus LinMol `representation`: `m_resolved` or `m_incoherent_average` | simulation runner and tests | one typed model registry shared by both workflows | internal transition facade |
| `models.build_{linmol,twolevel,vibladder}` and `build_initial_state` | selected by `build_model` | simulation model facade | model-owned constructors and one explicit state specification | internal |
| `dipole.create_dipole_matrix` | runtime basis class including SymTop | optimization runner, examples, tests | model-owned construction called by shared model builder | temporary public, then internal/delete |
| `dipole.<model>.builder.build_mu` | model-specific parameters | dipole cache classes | private model dipole kernels | internal |
| `dynamics.PropagatorFactory.create_propagator` | required typed state path and `PropagationOptions` | tests and possible direct users | `propagate(problem, options)` with explicit solver selection | typed transition facade since P2.3-c; delete after `PropagationProblem` owns construction |
| `optimization.ALGO_REGISTRY` | `local`, `krotov`, `grape` | package and example optimization runners | private typed optimization dispatch | internal |
| `spectroscopy.create_calculator_from_params` | spectroscopy parameter mapping | examples and tests | typed spectroscopy facade | target public in subpackage |
| removed `core.units.parameter_processor` | parameter-name suffix and mutable conversion tables | no remaining callers after D-045 | typed schema conversion at boundary | deleted by D-045 |
| removed `ParameterProcessor.create_hamiltonian_from_params` and `create_efield_from_params` | parameter dictionary | no callers found by P3.2-a acceptance audit | constructors remain owned by operator/field and the config boundary | deleted in P3.2-a; removal also eliminates `core -> fields` reverse dependency |
| removed `ElectricField.create_from_SI` and `create_with_units` | explicit units | no callers found | direct construction with required `time_units`, or `from_time_grid` | deleted by D-048 |

`PropagatorFactory` no longer inspects polarization or sparsity. It requires a
typed state path and one `PropagationOptions`, then validates them against the
shared capability registry before construction. Public solver `propagate()` methods require that same typed options object and one `PropagationProblem`; coupling is owned by `SystemModel`. They return one unconditional `PropagationResult`; `return_times` and loose physical arguments are removed. Backend state remains native until explicit `to_numpy()`. Only the private optimizer migration adapters retain `**kwargs`.

## 6. Examples and documentation callers

D-044 reduces the supported set to three top-level typed examples:

| Example | Boundary exercised |
|---|---|
| `example_typed_twolevel.py` | generated scalar field and dense NumPy RK4 |
| `example_typed_spectral_modulation.py` | unit-aware delay, phase modulation, GDD, and TOD |
| `example_external_scalar_field.py` | exact `TimeGrid` and external `ScalarField` injection |

`examples/launcher.py` lists only those files. All three import the public
`simulation.run_simulation_case` facade, write no output, check population
normalization, and execute through `scripts/smoke_examples.py` in CI.

Former v0.2 examples, parameter modules, notebook, and dedicated helpers are
under `examples/archives/v0_2_scripts/`. Additional optimizer scripts with
undefined experiment-specific constants and the external C++ RK4 example are
also archived without inferred fixes. Archived files are historical migration
evidence, are excluded from Ruff and execution, and are not public API callers.

The stale root package docstring and root README APIs remain Phase 8
documentation debt; archiving old examples does not recreate any compatibility
shim.

## 7. P0.1 acceptance record

- All nine current root `__all__` exports have a disposition.
- Both console scripts have a disposition and a traced configuration route.
- Every explicit subpackage `__all__` was enumerated.
- All factories and registries found by source search were classified.
- Active example package imports were checked without running simulations.
- Internal `src/` files do not use root convenience imports.
- No source or numerical behavior changed during this inventory.

Validation at this checkpoint: both CLI `--help` routes loaded successfully,
all documented relative links resolved, every current `__all__` name was found
in this inventory, and the full suite passed with 360 tests and 9 skips.

P0.1 is complete. P0.2 may add the physics test layout without beginning the
target directory migration.
