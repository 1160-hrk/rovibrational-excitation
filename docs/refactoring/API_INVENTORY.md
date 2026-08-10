# API and entry-point inventory

Last verified: 2026-08-10
Scope: Phase 0 task P0.1
Original inventory baseline: `613ce93`
Latest API checkpoint: `7d14fda`

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
| `Hamiltonian` | Tests and examples use `core.basis.Hamiltonian` | `core.operators.Hamiltonian` | temporary public |
| `StateVector` | Tests and examples use `core.basis.StateVector` | `core.states.StateVector` | temporary public |
| `DensityMatrix` | Tests use `core.basis.DensityMatrix` | `core.states.DensityMatrix` | temporary public |
| `ElectricField` | Simulation, optimization, tests, and examples | root re-export backed by `fields.ElectricField` | target public |
| `LinMolDipoleMatrix` | Tests and examples use its model subpackage | constructed by `models.linear_molecule`; advanced class remains under that model | temporary public |
| `AbsorbanceCalculator` | spectroscopy examples and tests | `spectroscopy.AbsorbanceCalculator` or decomposed facade | temporary public at root; target public in subpackage |
| `ExperimentalConditions` | spectroscopy examples and tests | `spectroscopy.ExperimentalConditions` | temporary public at root; target public in subpackage |
| `create_calculator_from_params` | spectroscopy examples and tests | typed spectroscopy constructor under `spectroscopy` | temporary public at root; target public in subpackage |

Every current root `__all__` name therefore has an explicit disposition. Only
`ElectricField` is proposed to remain a root re-export.

### 2.2 Other accessible root attributes

`__version__` and `__author__` are accessible but absent from `__all__`.
`core`, `dipole`, `plots`, `simulation`, and `spectroscopy` are bound by
eager root imports even though only `dipole` and `spectroscopy` have package
`__init__.py` files. The others currently rely on namespace-package behavior.

| Current name | Target | Disposition |
|---|---|---|
| `__version__` | root metadata | target public; add to `__all__` |
| `__author__` | package metadata only | internal; do not promise as API |
| `core` | explicit `core/__init__.py` with narrow exports | target public subpackage |
| `dipole` | functionality moves under model ownership | temporary public; delete package after migration |
| `plots` | `visualization` | temporary public; rename |
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
| `core.basis` | `BasisBase`, `Hamiltonian`, `LinMolBasis`, `TwoLevelBasis`, `VibLadderBasis`, `SymTopBasis`, `StateVector`, `DensityMatrix` | generic types to `core`; model bases to their `models.*` owners | temporary public |
| `core.units` | `PhysicalConstants`, `UnitConverter`, `converter`, `UnitValidator`, `validator`, `ParameterProcessor`, `parameter_processor` | immutable constants and explicit conversion services under `core.units`; typed config handles parameter conversion | classes temporary public; singleton objects internal and delete |

`core` itself has no `__init__.py`, so it has no explicit public contract today.

### 3.2 Fields

| Current package | Exact exported names | Target | Disposition |
|---|---|---|---|
| `core.electric_field` | `ElectricField`, `gaussian`, `lorentzian`, `voigt`, `gaussian_fwhm`, `lorentzian_fwhm`, `voigt_fwhm`, `apply_sinusoidal_mod`, `apply_dispersion`, `get_mod_spectrum_from_bin_setting` | `fields` | target public subpackage; only `ElectricField`, `gaussian`, and `gaussian_fwhm` proposed at root |

The modulation helpers remain public under `fields` only if Phase 4/5 tests
establish their units and sampling contracts. Until then their stability is
temporary.

### 3.3 Propagation and scaling

| Current package | Exact exported names | Target | Disposition |
|---|---|---|---|
| `core.propagation` | `PropagatorBase`, `SchrodingerPropagator`, `LiouvillePropagator`, `MixedStatePropagator`, `PropagatorFactory` | `dynamics` typed problem/options/result plus `propagate` | temporary public; factory class deletes after replacement |
| `core.propagation.algorithms` | `rk4_lvne`, `rk4_lvne_traj`, `rk4_schrodinger`, `splitop_schrodinger` | `dynamics.solvers` private kernels | internal |
| `core.propagation.algorithms.rk4` | `rk4_lvne`, `rk4_lvne_traj`, `rk4_schrodinger` | `dynamics.solvers.rk4` | internal |
| `core.propagation.algorithms.split_operator` | `splitop_schrodinger` | `dynamics.solvers.split_operator` | internal |
| `core.nondimensional` | `NondimensionalizationScales`, `ScaleValue`, `nondimensionalize_system`, `nondimensionalize_with_SI_base_units`, `nondimensionalize_from_objects`, `determine_SI_based_scales`, `create_dimensionless_time_array`, `analyze_regime`, `dimensionalize_wavefunction`, `get_physical_time` | `dynamics.scaling` with one explicit scaling representation | strict temporary public surface; move without adding competing strategies |

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
| `simulation.models` | `CouplingSpec`, `ModelComponents`, `build_model` | typed model protocol and config dispatch under `models` | internal transition facade; delete after callers migrate |

`SymTopBasis` and `SymTopDipoleMatrix` are importable, but the primary
simulation `build_model` registry supports only `linmol`, `twolevel`, and
`vibladder`. SymTop must not be advertised as stable until O-005 is resolved.

### 3.5 Optimization and spectroscopy

| Current package | Exact exported names | Target | Disposition |
|---|---|---|---|
| `optimization` | `run_local_optimization`, `run_krotov_optimization`, `run_grape_optimization`, `ALGO_REGISTRY` | typed functions under `optimization`; private registry | run functions target public in subpackage; registry internal |
| `spectroscopy` | `AbsorbanceCalculator`, `ExperimentalConditions`, `SpectroscopyCalculationReport`, `create_calculator_from_params` | decomposed spectroscopy modules with a tested facade | target public in subpackage; numerical policy accepted by D-023, scientific references pending O-007 |

`cli`, `simulation`, and `plots` have no explicit `__all__`. `cli/__init__.py`
exists but is empty; `simulation` and `plots` are namespace packages.

## 4. Console scripts and configuration routes

| Script | Current route | Input and construction path | Target | Disposition |
|---|---|---|---|---|
| `rve-simulate` | `cli.simulate:main` | Python file executed by `simulation.config.load_params_file` -> implicit unit conversion -> iterable sweep expansion -> per-case validation -> `simulation.models.build_model` -> `SchrodingerPropagator` | versioned typed simulation config and one shared model/field builder | target public command; replace input contract |
| `rve-optimize` | `cli.optimize:main` | YAML `safe_load` -> dotted overrides -> private `_build_basis` and `_build_dipole` -> `optimization.ALGO_REGISTRY` -> algorithm function | versioned typed optimization config reusing the same model/field builders | target public command; replace orchestration internals |

Both command names should remain. Backward compatibility for current config
files is not required, but result and config schemas must be explicitly
versioned so historical calculations remain interpretable.

### 4.1 Current simulation parameter path

1. `rve-simulate PARAMFILE` calls `run_all_with_checkpoint`.
2. `load_params_file` executes arbitrary Python and collects every non-dunder
   module attribute.
3. The mutable global `parameter_processor` heuristically converts recognized
   unit-suffixed values.
4. `expand_cases` treats most iterable values as sweep dimensions; only
   `polarization`, `initial_states`, and `envelope_func` are fixed-value
   exceptions.
5. `validate_simulation_case` runs only after expansion.
6. `build_model` dispatches through a local dictionary and returns
   `ModelComponents` plus scalar/Cartesian coupling metadata.
7. `runner._run_one` constructs the field and propagator, then writes an
   unversioned NPZ/JSON result.

This entire route is temporary. Python-file execution, heuristic conversion,
implicit sweep inference, and unversioned output are not part of the target
contract.

### 4.2 Current optimization path and divergence

`run_from_config` accepts YAML, a `Path`, or a dictionary. It owns a second set
of basis and dipole builders instead of using `simulation.models.build_model`.
The parameter names also differ (`omega_cm` versus `omega_rad_phz`, for
example). It silently changes an unknown dipole unit to `C*m` and an unknown
potential type to `harmonic`. These fallbacks violate the explicit-validation
policy and must become errors when the typed config is introduced. They are
documented here only; P0.1 does not change calculation behavior.

The runner catches every plotting exception and returns a nominally successful
optimization. Phase 7 must distinguish an optimization result from optional
visualization failure.

## 5. Factories and registries

| Current entry | Dispatch key | Current callers | Target | Disposition |
|---|---|---|---|---|
| `simulation.models.build_model` | `basis_type`: `linmol`, `twolevel`, `vibladder` | simulation runner and tests | one typed model registry shared by both workflows | internal transition facade |
| `simulation.models.build_{linmol,twolevel,vibladder}` and `build_initial_state` | selected by `build_model` | simulation model facade | model-owned constructors and one explicit state specification | internal |
| `dipole.create_dipole_matrix` | runtime basis class including SymTop | optimization runner, examples, tests | model-owned construction called by shared model builder | temporary public, then internal/delete |
| `dipole.<model>.builder.build_mu` | model-specific parameters | dipole cache classes | private model dipole kernels | internal |
| `core.propagation.PropagatorFactory.create_propagator` | state type, backend, algorithm, polarization/sparsity heuristic | tests and possible direct users | `propagate(problem, options)` with explicit solver selection | temporary public, then delete |
| `optimization.ALGO_REGISTRY` | `local`, `krotov`, `grape` | package and example optimization runners | private typed optimization dispatch | internal |
| `spectroscopy.create_calculator_from_params` | spectroscopy parameter mapping | examples and tests | typed spectroscopy facade | target public in subpackage |
| `core.units.parameter_processor` | parameter-name suffix and mutable conversion tables | simulation config and tests | typed schema conversion at boundary | internal singleton, then delete |
| `ParameterProcessor.create_hamiltonian_from_params` and `create_efield_from_params` | parameter dictionary | no callers found | typed constructors owned by operator/field or config boundary | delete after confirming no external workflow |
| `ElectricField.create_from_SI` and `create_with_units` | explicit units | no callers found | one explicit field constructor contract | temporary public method; consolidate in Phase 4 |

The current `PropagatorFactory` automatically prefers split-operator for a
pure state with constant polarization. The v0.3 contract must not retain that
heuristic silently: the selected solver is explicit and validated against the
problem capabilities.

## 6. Examples and documentation callers

The active inventory contains 31 Python files under `examples/` after excluding
`results/`, `figures/`, `archives/`, notebook checkpoints, and `__pycache__`.
They divide into:

| Group | Files | Disposition |
|---|---:|---|
| direct simulation/spectroscopy examples named `example_*.py` | 21 | reduce to small tested canonical examples; migrate useful scientific cases |
| parameter/config examples named `params_*.py` | 4 | replace with versioned declarative config examples |
| launcher and reusable optimization system | 2 | migrate only if still needed after typed CLI |
| runner/tool/utility helpers | 4 | package reusable code or delete duplicate example infrastructure |

A static import execution check loaded 121 package import statements without
executing example bodies. One active file is definitely broken:

- `examples/example_default_units.py` imports nonexistent
  `get_default_units`, `set_default_units`, `auto_convert_parameters`,
  `apply_default_units`, and `print_unit_help` from `core.units`.

It should be deleted or rewritten around the final typed unit/config API in
Phase 4/8; recreating these legacy globals is forbidden.

Other stale references are documentation-level rather than active example
imports:

- root package docstring: five nonexistent spectroscopy APIs listed in
  section 2.2;
- root `README.md`: `rve.generate_H0_LinMol`,
  `rovibrational_excitation.core.states.StateVector`, and
  `rve.schrodinger_propagation`.

Examples import deep implementation paths such as
`core.propagation.schrodinger`, `core.units.constants`, and
`dipole.<model>`. Their current import success does not make those paths target
public APIs. Phase 8 must run canonical examples as smoke tests against only
the final supported surface.

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
