# Target architecture for v0.3

Status: Accepted working target; Phase 3 migration complete
Last updated: 2026-09-15

## 1. Design goals

The architecture must make the following questions answerable from types and
module ownership:

- Which model defines this basis, Hamiltonian, and dipole?
- Which unit is an array in when it enters a kernel?
- Which backend owns every array in one calculation?
- Is the coupling scalar or Cartesian?
- Is the state pure, an incoherent ensemble, or a density matrix?
- Which algorithm and storage modes are supported?
- What time does each returned state represent?
- Which layer may read configuration or write files?

The design should favor explicit data flow over inheritance and broad
`**kwargs` APIs. Hot numerical loops may remain backend-specific.

## 2. Current architecture problems

### 2.1 Model ownership is split

A linear molecule is currently represented across:

- `core/basis/linmol.py`;
- the generic `core/operators.py` plus model-specific basis modules;
- `dipole/linmol/builder.py`;
- `dipole/linmol/cache.py`;
- `models/linmol.py`;
- factories in both `dipole` and `models`.

VibLadder and LinMol still follow split variations of this pattern. SymTop has
a production model owner alongside legacy direct paths, while TwoLevel is
consolidated under `models/two_level`. The remaining splits make model
capability, required parameters, and coupling semantics hard to discover.

### 2.2 Propagation preparation is overloaded

`dynamics/utils.py` currently handles combinations of:

- backend lookup;
- axes validation;
- field component extraction;
- units;
- dense/sparse conversion;
- scalar coupling;
- nondimensionalization;
- automatic timestep preparation.

Preparation, policy, and numerical execution must become separate layers.

### 2.3 Workflow modules own too much

`simulation/runner.py` combines model construction, field construction,
propagation setup, execution, progress reporting, multiprocessing, validation,
checkpoint handling, and storage orchestration.

`spectroscopy/absorbance_calculator.py` combines many response and spectrum
operations in one 898-line module.

### 2.4 Capabilities are represented by strings and conventions

Examples:

- backend strings appear on dipoles and propagators independently;
- scalar models still use storage axes;
- algorithm support is split between factories and runtime checks;
- return types change according to boolean kwargs;
- a CuPy request can still encounter helper-level NumPy fallback.

These become typed contracts and explicit capability checks.

## 3. Target package tree

~~~text
src/rovibrational_excitation/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── arrays.py
│   ├── operators.py
│   ├── states.py
│   ├── time.py
│   └── units/
│       ├── __init__.py
│       ├── constants.py
│       ├── converters.py
│       └── types.py
├── fields/
│   ├── __init__.py
│   ├── field.py
│   ├── envelopes.py
│   └── modulation.py
├── models/
│   ├── __init__.py
│   ├── base.py
│   ├── registry.py
│   ├── validation.py
│   ├── _parameter_validation.py
│   ├── symmetry/
│   │   ├── groups.py
│   │   ├── policy.py
│   │   └── presets.py
│   ├── two_level/
│   │   ├── model.py
│   │   ├── basis.py
│   │   ├── dipole.py
│   │   └── parameters.py
│   ├── vib_ladder/
│   │   ├── model.py
│   │   ├── basis.py
│   │   ├── dipole.py
│   │   └── morse.py
│   ├── linear_molecule/
│   │   ├── model.py
│   │   ├── basis.py
│   │   ├── dipole.py
│   │   └── rotation.py
│   └── symmetric_top/
│       ├── model.py
│       ├── basis.py
│       └── dipole.py
├── dynamics/
│   ├── __init__.py
│   ├── problem.py
│   ├── options.py
│   ├── result.py
│   ├── capabilities.py
│   ├── scaling/
│   │   ├── scales.py
│   │   ├── transform.py
│   │   └── policy.py
│   └── solvers/
│       ├── base.py
│       ├── rk4/
│       │   ├── schrodinger_numpy.py
│       │   ├── schrodinger_cupy.py
│       │   ├── sparse.py
│       │   └── liouville_numpy.py
│       └── split_operator/
│           ├── numpy.py
│           └── cupy.py
├── simulation/
│   ├── config.py
│   ├── case.py
│   ├── execute.py
│   ├── sweep.py
│   └── manager.py
├── optimization/
│   ├── model.py
│   ├── objective.py
│   ├── result.py
│   ├── local.py
│   ├── grape.py
│   ├── krotov.py
│   ├── krotov_initial_field.py
│   └── spectral_constraints.py
├── spectroscopy/
│   ├── response.py
│   ├── thermal.py
│   ├── broadening.py
│   ├── transform.py
│   └── observables.py
├── io/
│   ├── schema.py
│   ├── serialization.py
│   ├── storage.py
│   └── checkpoint.py
├── visualization/
│   └── ...
└── cli/
    ├── simulate.py
    └── optimize.py
~~~

Names may be refined during Phase 3, but ownership and dependency direction are
binding unless the decision log changes.

## 4. Dependency rules

Allowed dependency direction:

~~~text
core <- fields
core <- models
core <- dynamics
fields <- models
models + fields + dynamics + io <- simulation
models + fields + dynamics <- optimization
core + models <- spectroscopy
simulation + optimization <- cli
results + fields <- visualization
~~~

Rules:

1. `core` imports only the standard library, NumPy/SciPy as required, and its
   own submodules.
2. `models` may depend on core units, states, and operators, but not on
   simulation runners or CLI.
3. `dynamics/solvers` may depend on core array contracts and capabilities, but
   not on concrete model classes.
4. `simulation` is orchestration. It may construct models and solvers but must
   not contain physical formulas.
5. `io` serializes typed inputs/results and owns schema versions. Numerical
   kernels perform no I/O.
6. `cli` only parses arguments, calls application services, and converts
   exceptions to exit codes.
7. `visualization` consumes results and does not participate in calculation.
8. No lower layer imports package-root convenience exports; use direct module
   imports to avoid cycles.
9. Optional dependencies are imported lazily at the capability boundary.
10. `optimization` reuses frozen model parameters and model-owned operator
    builders. Workflow-specific state tuples, controls, and update rules remain
    in optimization and do not leak into model construction.

Architecture tests should inspect imports and reject reverse dependencies after
Phase 3.

## 5. Core typed contracts

The following names are provisional but the represented information is
required.

### 5.1 TimeGrid

~~~python
@dataclass(frozen=True)
class TimeGrid:
    field_times_fs: NDArray[np.float64]
    field_dt_fs: float
    propagation_dt_fs: float
    propagation_steps: int
~~~

Invariants:

- one-dimensional, finite, strictly increasing field times;
- at least three samples;
- odd number of samples;
- `propagation_dt_fs == 2 * field_dt_fs`;
- `len(field_times_fs) == 2 * propagation_steps + 1`;
- first and last values are the configured endpoints.

The constructor should derive redundant fields and reject inconsistent input
rather than accepting all values independently. It owns a defensive,
read-only copy of `field_times_fs` so a frozen instance cannot be mutated
through an external NumPy reference.

`TimeGrid` is the solver-facing canonical grid. D-027 defines one explicit
pre-solver exception: local optimization retains
`LocalOptimizerLegacyGridV1` for control-field storage and segment indexing.
That layout is never rebuilt or endpoint-repaired by `TimeGrid`; it exposes the
odd prefix historically consumed by RK4 when constructing the solver input.
The returned optimization field and cost still use the complete legacy storage
array. GRAPE and Krotov have no such exception: D-029 requires canonical
`TimeGrid`, explicit `field_dt_fs`, complete internal trajectories, and
output-only `output_stride`.

D-050 gives Krotov one frozen initial-field ownership boundary. It selects
generated or sampled input explicitly, converts public quantities to canonical
fs/cycles-per-fs/V/m/fs^2/fs^3 values, and validates sampled fields against the
same `TimeGrid`. `krotov.py` consumes only the canonical writable field copy;
its update loop does not parse units or choose a source.

D-058 gives Local control gain a frozen scalar boundary. Public callers provide
a required value/unit pair, `core.units` converts it to `(V/m)^2 fs`, and
the Local workflow passes only the positive canonical scalar into its unchanged
update expressions. The legacy grid adapter and numerical loop never inspect a
unit string.

D-059 gives Local control a required initialization sum type. The
`seed_field` branch owns a required positive direct-amplitude value/unit pair
and positive segment count, then supplies canonical V/m to the unchanged legacy
seed update. The `none` branch owns no field data and runs a mode-specific
initial-response preflight before propagation. Configuration chooses the branch
explicitly; the numerical loop never invents or silently replaces it.

### 5.2 CouplingSpec

~~~python
@dataclass(frozen=True)
class CouplingSpec:
    mode: Literal["scalar", "cartesian"]
    scalar_axis: Literal["x", "y", "z"] | None
    cartesian_axes: tuple[Axis, Axis] | None
~~~

Invariants:

- scalar mode has exactly one storage/operator axis;
- Cartesian mode has exactly two field/operator axes;
- user-facing polarization is not required for scalar physics.

### 5.3 SystemModel

~~~python
@dataclass(frozen=True)
class SystemModel:
    name: str
    basis: Basis
    hamiltonian: Hamiltonian
    dipole: DipoleOperator
    coupling: CouplingSpec
    metadata: Mapping[str, JSONValue]
~~~

The model owns basis/Hamiltonian/dipole consistency. Construction validates
matrix dimensions and the state-index mapping once.

### 5.4 PropagationProblem

~~~python
@dataclass(frozen=True)
class PropagationProblem:
    model: SystemModel
    field: ElectricField
    time_grid: TimeGrid
    initial_state: PureState | IncoherentEnsemble | DensityState
~~~

The state type is explicit. Do not use an arbitrary list or square-array
heuristic in the final public API.

### 5.5 ExecutionPolicy and PropagationOptions

~~~python
@dataclass(frozen=True, slots=True)
class ExecutionPolicy:
    backend: ArrayBackend
    storage: MatrixStorage

@dataclass(frozen=True, slots=True)
class PropagationOptions:
    algorithm: PropagationAlgorithm
    execution: ExecutionPolicy
    return_trajectory: bool
    sample_stride: int
    scaling: ScalingMode
    renormalization: RenormalizationPolicy
~~~

There is one execution policy per calculation. Dipole construction and solver
selection receive the same policy. Backend, matrix storage, and algorithm are
required explicit choices; they are never inferred from polarization, matrix
type, or optional dependency availability.

Since P2.4-c, public solver calls accept one `PropagationProblem`, one required
`PropagationOptions`, and only temporary result controls; they accept no loose
Hamiltonian, field, dipole, state, coupling mode, or axes. `SystemModel.coupling`
owns the exclusive scalar or Cartesian selection. Split calls additionally
require the same explicit `split_interaction` used to construct the solver. The
private array adapters still receive the characterized legacy projections, so
this ownership change does not alter coupling or numerical semantics.

Do not encode the field grid with a separate solver `dt` option. The time grid
is the source of truth.

### 5.6 PropagationResult

~~~python
@dataclass(frozen=True)
class PropagationResult:
    times_fs: NDArray[np.float64]
    state: Array
    state_kind: Literal["wavefunction", "density_matrix"]
    trajectory: bool
    backend: str
    metadata: Mapping[str, JSONValue]
~~~

The return type no longer changes between raw array and tuple based on
`return_time_*` flags. A final-state result still contains a one-element time
array. A trajectory always includes the configured endpoint, appending it after
the regular stride samples when necessary without altering integration.

`state` remains native to the selected array backend. The explicit
`PropagationResult.to_numpy()` method creates a host result; storage invokes
that conversion only at the I/O boundary.

Metadata should include:

- package and result-schema versions;
- model name and basis dimension;
- algorithm/backend/storage;
- dimensionalization scales when used;
- time-grid summary;
- normalization policy;
- deterministic configuration hash.

P2.5 implements this contract in `dynamics.result`. Metadata values are recursively validated as finite JSON data and frozen; unsupported values fail explicitly. The SHA-256 hash scope is named `declared_model_metadata_and_propagation_contract`: it covers declared model metadata, coupling, options, time grid, and actual same-call nondimensional scales. It intentionally excludes numerical Hamiltonian, dipole, field, and state arrays so computing provenance never triggers a hidden device-to-host transfer. Full content-addressed input provenance belongs at a future explicit persistence boundary.

The Phase 2 adapter requests a complete private trajectory with `sample_stride=1`, preserving every integration step and field index. Stride one reuses that backend state array. Larger output strides thin it and append the already computed endpoint; they temporarily retain the full internal trajectory. Phase 5 should replace only this storage policy with an endpoint-complete low-level writer after characterization tests, without changing integration.

Normal simulation and fixed-M averaging call `to_numpy()` explicitly where their current population and persistence code requires NumPy. Optimizers retain their private array bridge until their separately characterized time/index contracts are migrated.

## 6. Backend and capability design

Backend selection is separated from solver capability.

~~~python
@dataclass(frozen=True)
class SolverCapabilities:
    state_kinds: frozenset[StateKind]
    algorithms: frozenset[Algorithm]
    backends: frozenset[BackendName]
    storage_modes: frozenset[StorageMode]
    requires_diagonal_h0: bool
~~~

A registry lookup validates the complete combination before constructing large
matrices.

Required adapters:

~~~python
class ArrayBackend(Protocol):
    name: str
    def asarray(self, value, *, dtype): ...
    def to_host(self, value) -> np.ndarray: ...
    def is_array(self, value) -> bool: ...
~~~

Rules:

- no helper silently returns NumPy for an unavailable CuPy request;
- no repeated CPU/GPU conversion inside a propagation loop;
- result host/device policy is explicit;
- sparse support is a capability, not inferred from a matrix after selection;
- solver capability tests cover every advertised combination;
- hot CPU, sparse, and GPU loops may remain separate implementations.

P5.1-b establishes this split for density propagation without moving the
validated low-level API. `dynamics.algorithms.rk4.lvne` owns validation and
numeric dense-array preparation; `liouville_numpy` owns only the prevalidated
NumPy/Numba loop. The kernel has no dependency on units, model objects,
configuration, runners, or I/O. Its inherited temporary-allocation behavior is
not an architectural requirement. P5.1-c reuses only the bitwise-identical
right/next-left endpoint Hamiltonian and records the modest measured benefit.
Remaining stage intermediates may be replaced only in a separately benchmarked
unit that assesses any numerical drift; the slower sub-ulp-different
output-buffer experiment is not part of the architecture.

D-062 verifies the NumPy side of this target. The typed result boundary already
preserves an existing device array, but current CuPy RK4 and split helpers
return through host memory. The target is still one direct
`device kernel -> backend-native PropagationResult` path; the current
`device -> host -> device` adapter is recorded debt, not an accepted layer.

## 7. Units and scaling ownership

`core/units` owns physical unit definitions and pure conversions.

`dynamics/scaling` owns nondimensional transformation of a complete propagation
problem.

Model constructors accept explicit values plus unit information, produce
validated operators, and do not perform solver policy.

Numerical kernels receive only canonical or nondimensional arrays. They never
call the converter.

The target internal dimensional units are those listed in
`PHYSICS_CONTRACTS.md`.

`core.units.Frequency` is the first quantity value object. It stores the finite
validated input value and unit for provenance and exposes canonical
`angular_rad_per_fs`; `cycles_per_fs` is an explicit representation for FFT
boundaries. New public configuration uses neutral names plus required unit
fields. Unit-bearing suffixes such as `_rad_phz` are migration debt, not target
schema names.

## 8. Model ownership

Each model package owns:

- a frozen parameter schema;
- basis construction and state-index mapping;
- Hamiltonian construction;
- dipole construction and selection rules;
- coupling capability;
- model-specific validation;
- small reference fixtures/tests.

Example:

~~~python
@dataclass(frozen=True)
class VibLadderParameters:
    v_max: int
    vibrational_frequency: Frequency
    anharmonic_shift: Frequency
    potential: Literal["harmonic", "morse"]
    dipole_scale: DipoleMoment
~~~

The first frozen schemas now exist as `LinMolParameters`,
`VibLadderParameters`, and `TwoLevelParameters`. Public mappings are validated
before basis or dipole allocation. Each neutral model frequency requires a
paired `*_units`; model builders consume only canonical
`Frequency.angular_rad_per_fs`. Existing low-level basis constructors retain
their angular-frequency arguments until Phase 6.

P6.1-a freezes the complete current TwoLevel projection before ownership
moves. P6.1-b separately resolves O-013 through the central constant and unit
converter. P6.1-c moves the implementation, and P6.1-d completes the target
owner with `parameters.py`. The unused mapping and stateless-dipole wrappers
are deleted, and the legacy generic dipole factory no longer depends on or
constructs TwoLevel. Every P6.1-a behavioral contract and corrected conversion
value remains unchanged. Shared strict scalar/unit conversion helpers live in
private `_parameter_validation.py`, separate from every model schema.

P6.2-a freezes the complete current VibLadder projection before ownership
moves. P6.2-b places its basis, dipole class, transitional stateless builder,
and production builders in `models/vib_ladder/` and removes all three former
owners. The shared `dipole/vib` harmonic and Morse element functions do not
move while LinMol and SymTop still import them. P6.2-c completes the owner by
moving its frozen schema and deleting the unused mapping and stateless-dipole
wrappers. The generic dipole factory drops VibLadder during the move because
retaining it would create a new `dipole -> models` reverse dependency. All
D-067 values remain unchanged.

P6.3-a freezes the complete current LinMol ownership and construction
projection before any move. Its signed basis order, parameter conversion,
Hamiltonian, M-resolved Cartesian dipoles, coherent basis-index state input,
dense/CSR behavior, and current builder paths are executable contracts under
D-070. The independent physics suite continues to own selection-rule,
M-average, Morse, and propagation references.

P6.3-b completes the ownership-only move under D-074. `LinMolBasis`, the
stateful/stateless dipole implementation, and all model builders now share
`models/linear_molecule/`; former owner paths are absent. The generic dipole
factory drops LinMol to preserve the dependency direction. The shared schema
and transitional wrapper cleanup remain P6.3-c, and
`simulation/m_average.py` remains a workflow rather than model owner.

P6.3-c completes the owner under D-075. The frozen schema joins the model
package unchanged, the unused mapping wrapper is deleted, and the stateless
dipole implementation becomes the private `_build_mu` kernel used by the
stateful class. Only the schema, basis, stateful dipole, and typed builders are
exported. All D-070 values remain unchanged.

P6.4-a/D-076 audits the two SymTop implementations before cleanup. The legacy
core/dipole skeleton has no production caller, uses different ordering,
filtering, anharmonic semantics, and transverse phases, and its dense and CSR
dipole routes fail directly. It is therefore a deletion target; none of its
formulae migrate into the independently referenced D-053 production owner.

Derived values such as Morse `N` are properties or construction-local values,
not global configuration.

A model registry maps an explicit model name to a builder. It does not inspect
unrelated parameter names to guess the model.

`models/symmetry` owns reusable geometric descriptors, rotational-state
classification policies, and source-versioned molecule-name presets. It is
deliberately below model builders and above no numerical kernel. A preset may
select symmetry constraints but may not construct a Hamiltonian or provide
physical constants. Model builders must pass an explicit state to the policy
and apply an explicitly selected spin-isomer filter; kernels receive the
already constructed basis and weights, never molecule names.

## 9. Current-to-target mapping

| Current path | Target owner | Migration note |
|---|---|---|
| `core/basis/base.py` | `core/states.py` and `models/base.py` | Separate generic state protocol from model basis |
| `core/basis/states.py` | `core/states.py` | Remove model-specific assumptions |
| `core/basis/hamiltonian.py` | `core/operators.py` | Complete in P3.1-a; implementation moved unchanged and old path removed |
| `core/basis/linmol.py` | `models/linear_molecule/basis.py` | Complete in P6.3-b; old path removed and D-070 values preserved |
| `core/basis/viblad.py` | `models/vib_ladder/basis.py` | Complete in P6.2-b; old path removed and D-067 values preserved |
| `dipole/viblad/*` | `models/vib_ladder/dipole.py` | Complete in P6.2-c; old package removed and unused stateless wrapper deleted |
| `models/vibladder.py` | `models/vib_ladder/model.py` | Complete in P6.2-c; typed builders remain and unused mapping wrapper is deleted |
| `models/parameters.py::VibLadderParameters` | `models/vib_ladder/parameters.py` | Complete in P6.2-c; validation and unit conversion unchanged |
| `core/basis/twolevel.py` | `models/two_level/basis.py` | Complete in P6.1-c; old path removed and D-063/D-064 values preserved |
| `dipole/twolevel/*` | `models/two_level/dipole.py` | Implementation moved in P6.1-c; redundant stateless builder removed in P6.1-d |
| `models/twolevel.py` | `models/two_level/model.py` | Complete in P6.1-c; registry and optimization imports moved |
| `models/parameters.py::TwoLevelParameters` | `models/two_level/parameters.py` | Complete in P6.1-d; validation/conversion behavior unchanged |
| `core/basis/symtop.py` | `models/symmetric_top/{basis,rotational,dipole,model}.py` | D-053 production owner complete for normal NumPy dense/CSR RK4; legacy direct path remains experimental until Phase 6 removal |
| no former shared symmetry owner | `models/symmetry/{groups,policy,presets}.py` | D-052 foundation complete; D-053 connects CH3F sector filtering to the production symmetric-top builder; linear-builder integration remains later work |
| `core/electric_field/*` | `fields/*` | Complete in P3.1-b; bodies unchanged, `core.py` renamed `field.py`, old path removed |
| `dipole/base.py` | `core/operators.py` or `models/base.py` | Split generic operator/cache from model builder |
| `dipole/linmol/*` | `models/linear_molecule/{dipole,dipole_builder}.py` | Complete in P6.3-b; old package removed and implementations unchanged |
| `models/linmol.py` | `models/linear_molecule/model.py` | Complete in P6.3-b; registry and optimization imports moved |
| `models/parameters.py::LinMolParameters` | `models/linear_molecule/parameters.py` | Complete in P6.3-c; validation and unit conversion unchanged |
| `dipole/vib/*` | `models/vib_ladder/morse.py` or shared vibration module | Decide sharing from actual users |
| `simulation/models/*` | `models/*/model.py` plus `simulation/m_average.py` | P3.1-f moved construction to flat `models` and kept propagation workflow in `simulation`; model-specific split pending Phase 6 |
| model-selection subset of `simulation/validation.py` | `models/validation.py` | Complete in P3.2-b; predicates and messages preserved, model errors translated at the simulation boundary |
| `core/propagation/*` | `dynamics/*` | Complete in P3.1-c; numerical kernels unchanged, old path removed |
| `dynamics/algorithms/rk4/lvne.py` mixed boundary/kernel | `dynamics/algorithms/rk4/{lvne,liouville_numpy}.py` | P5.1-b separates validated preparation from the unchanged dense NumPy/Numba loop; final solver-package move remains later |
| `core/nondimensional/*` | `dynamics/scaling/*` | Complete in P3.1-d; formulas and thresholds unchanged, old path removed |
| `dynamics/algorithms/validation.py` | `core/validation.py` | Complete in P3.1-e; file is an exact rename and old path removed |
| `simulation/timegrid.py` | `core/time.py` | Complete in P3.2-a; obsolete writable-array wrapper removed and callers use `TimeGrid.from_bounds` directly |
| `simulation/storage.py` | `io/storage.py` | Complete in P3.1-g as a 100% exact rename; schema versioning deferred |
| `simulation/serialization.py` | `io/serialization.py` | Complete in P3.1-g as a 100% exact rename; typed schema redesign deferred |
| `simulation/checkpoint.py` | `io/checkpoint.py` | Complete in P3.1-g as a 100% exact rename; manager/persistence split deferred |
| `plots/*` | `visualization/*` | Complete in P3.1-h as five 100% exact renames; old namespace removed and optional Matplotlib remains lazy |
| `spectroscopy/absorbance_calculator.py` | `spectroscopy/{response,thermal,broadening,transform,observables}.py` | Characterize first |
| `optimization/*.py` | typed optimization modules | Characterize objectives and gradients first |

Migration uses `git mv` first, import repair second, and internal redesign only
after movement tests pass.

## 10. Public API target

Under D-073, root `rovibrational_excitation.__init__` exports exactly:

- `__version__`;
- `ElectricField`;
- `TimeGrid` and `ExecutionPolicy`;
- `PropagationProblem`, `PropagationOptions`, and `PropagationResult`;
- `run_simulation_case`.

Low-level kernels, cache implementations, converter internals, and runner
helpers must require submodule imports. Model, optimization, spectroscopy,
advanced field, and low-level core names remain public only through their
explicit subpackages. Root loading does not import optional heavy subpackages.

## 11. Configuration and result schemas

Simulation configuration becomes a typed schema with:

- explicit schema version;
- strict unknown-key rejection;
- model-discriminated parameter section;
- field-discriminated parameter section;
- explicit initial-state kind;
- one execution policy;
- serialization-safe values.

D-041 requires two mutually exclusive field construction routes at the typed
case boundary:

- a fully explicit generated-field specification; or
- an externally sampled scalar or Cartesian field supplied by the Python API
  with an exact canonical `TimeGrid` match.

Scalar fields serve TwoLevel, VibLadder, and LinMol M averaging. Cartesian
fields serve M-resolved LinMol. The concrete boundary is
`fields/sampled.py::{ScalarField, CartesianField}`; each makes a defensive
read-only copy of real V/m samples and owns the canonical `TimeGrid`. Generated
pulses are projected into these types only after the existing generator has
produced its exact arrays. Python injection enters through
`simulation.runner.run_simulation_case`.
Both routes now converge at `simulation.case.SimulationCase` after samples are
fixed and before model matrices are allocated. The frozen case owns the model
parameter schema, immutable initial-state indices, sampled field and canonical
time grid, representation/axes, and propagation controls. Model construction
receives the frozen schema directly and does not inspect the original simulation
mapping.

The final mapping boundary is closed rather than permissive: an unknown name is
an error, and known names are accepted only by the model, field route, and
algorithm that consume them. TwoLevel and VibLadder expose scalar coupling and
therefore reject `polarization`. M-resolved LinMol generated fields require
`polarization`; M averaging may only use its documented fixed-linear input.
`split_interaction` exists only for M-resolved LinMol with
`algorithm="split_operator"`; all other combinations reject it. Removed names
retain dedicated migration errors. Python config loading filters imported module
objects only, leaving other names visible to this strict boundary.

Generated simulation fields use named, serialization-safe discriminators.
`envelope_kind` resolves only the four existing three-argument envelopes:
`gaussian`, `gaussian_fwhm`, `lorentzian`, and `lorentzian_fwhm`.
`modulation_kind` is explicitly `none` or `sinusoidal`. Custom callables and
the two-width Voigt functions cross the already-sampled injection boundary;
construction never invents a second width or silently falls back to Gaussian.

The two routes are mutually exclusive. External injection rejects pulse
generation keys and performs no resampling or correction. Optional
scalar/Jones decomposition metadata is carried only when explicitly known; it
preserves the legacy generated `helicity_projected` approximation and is never
invented for arbitrary Cartesian samples. Numerical convergence assessment is
a separate application service, not a configuration fallback.

That service is concretely
`simulation.convergence.assess_simulation_convergence`. It executes two
otherwise identical cases on caller-selected same-endpoint grids, applies the
same caller-provided observable to both population results, and returns an
immutable `ConvergenceReport`. Its standard metric is maximum absolute
elementwise difference. Observable name, observable function, and finite
nonnegative tolerance are required; generated cases differ only by `dt`, while
external cases differ only by same-kind sampled fields. It cannot write,
resample, refine, retry, or replace the requested calculation.

Each model package owns a frozen parameter dataclass. Validation is complete
before matrix allocation, while derived quantities such as Morse `N` remain
instance-local properties. LinMol representation is a required enum-like
choice rather than a boolean.

Under D-056, configured optimization enters through a closed YAML/dict document
validated by `optimization.config` and algorithm-specific key ownership in
`optimization.options`. Model construction then reuses the frozen production
schemas. Every algorithm requires its ordered two-axis adapter, Krotov requires
an explicit generated or sampled field branch, and sampled data must match the
canonical field grid exactly. Required output and plot sections have explicit
API/CLI override precedence. This is the strict migration boundary; persistence
schema versioning and final public result types remain Phase 7 work.

Result files require a schema version independent of package version. A loader
must either parse a known schema or raise an actionable error. It must not guess
array meaning from key presence.

## 12. Performance constraints

Architecture abstractions stop before hot loops.

Before and after each solver migration, record:

- median wall time after JIT warmup;
- peak resident memory or allocated trajectory size;
- dense and sparse matrix dimension;
- backend and dependency versions;
- norm/trace error;
- final-state numerical difference.

A performance regression larger than 10% must be investigated. It may be
accepted only when documented with a correctness, memory, or maintainability
benefit and user approval.

Benchmarks are not ordinary unit tests and should run in a dedicated job or
marker.

## 13. Forbidden target patterns

- Generic `**kwargs` across public solver boundaries.
- Global mutable physical parameters.
- Unit strings inside numerical kernels.
- Backend selection repeated independently by model, dipole, and solver.
- Runtime fallback from requested GPU to CPU.
- Importing package-root re-exports from inside the package.
- Factories that guess capabilities after allocating matrices.
- Boolean combinations that produce different undocumented return types.
- Runners that contain model Hamiltonian or dipole formulas.
- Serialization without a schema version.
- Catch-all exception handlers that convert physics errors into success.
