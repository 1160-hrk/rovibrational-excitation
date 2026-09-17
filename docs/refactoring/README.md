# Refactoring source of truth

Last verified: 2026-09-17
Branch: `refactor/v0.3`
Behavioral baseline: `613ce93`

This directory is the authoritative planning and contract reference for the
v0.3 refactor. It is written for both maintainers and Codex. User-facing
documentation under `docs/` may describe the current released API and can lag
behind this refactor; conflicts must be resolved using implementation evidence,
tests, and the decision log.

## Documents

| Document | Purpose | Update trigger |
|---|---|---|
| `PHYSICS_CONTRACTS.md` | Equations, units, time grid, states, Morse rules, coupling, backend capabilities | Any physical or numerical contract change |
| `DECISIONS.md` | Accepted decisions and questions that still require the user | Every resolved or newly discovered ambiguity |
| `TARGET_ARCHITECTURE.md` | Target package tree, dependency rules, typed contracts, old-to-new mapping | Any architecture or ownership change |
| `EXECUTION_PLAN.md` | Ordered phases, task IDs, acceptance criteria, commit rules | At the start and completion of every phase |
| `API_INVENTORY.md` | Current exports, CLI/config routes, factories, examples, and v0.3 disposition | Any public/internal entry-point change |
| `UNIT_BOUNDARY_AUDIT.md` | Remaining explicit-unit debt, safe order, and user-confirmation items | Any physical input boundary migration |
| `VALIDATION_INVENTORY.md` | P1.2-B audit of standalone diagnostics, replacements, and unresolved scale-policy constants | Any legacy validation disposition or recovered scientific intent |
| `PHASE5_ACCEPTANCE_AUDIT.md` | Row-by-row CPU/CUDA numerical-engine acceptance status and transfer debt | Any solver capability, backend transfer, or Phase 5 status change |
| root `AGENTS.md` | Mandatory operating instructions and document routing | When workflow or required checks change |

## Mission

The refactor aims to make the package:

- physically auditable;
- explicit about units, time grids, normalization, algorithms, and backends;
- modular without hiding hot numerical loops behind costly abstractions;
- testable at model, kernel, workflow, and user API boundaries;
- safe to change without relying on backward compatibility;
- reproducible from configuration through serialized results.

“Complete” does not mean that every module is maximally abstract. It means that
ownership and dependencies are clear, unsupported states are rejected, and
physics changes are detected by tests.

## Current verified baseline

### Tests and quality

| Item | Baseline |
|---|---:|
| Pytest | 1232 passed, 10 skipped (1242 collected) |
| Measured branch coverage | 77% |
| Mandatory CI coverage floor | 47% |
| Ruff findings (active source, tests, examples, benchmarks, scripts) | 0 |
| Files failing format (same active scope) | 0 |
| Historical `examples/archives/` | Explicitly excluded by D-044 |
| Optimization module coverage | 8-90% |
| Spectroscopy coverage | 90% |
| `simulation/runner.py` coverage | 69% |
| RK4 Schrödinger coverage report | 20% |

These rows were last verified locally on 2026-09-16. On 2026-09-16, D-071
through D-073 make device-native CUDA plus a real-GPU run mandatory for v0.3,
define independent optimization/spectroscopy reference construction, and fix
the exact minimal typed root API. P6.3-c implements D-075: `LinMolParameters`
is model-owned, the unused mapping wrapper is removed, and the stateless
dipole implementation is private while retaining exact parity coverage.
Strict mypy covers 41 named modules and P6.3 is complete. P6.3-b implements
D-074: the LinMol basis,
dipole implementation, and model builders now share
`models/linear_molecule/`, the former owners are removed, and the generic
dipole factory drops LinMol to avoid a reverse dependency. All D-070 values
and the M-average workflow remain unchanged. P6.3-a supplied the nine move
guards, while 29 existing physics cases retain independent authority over
selection rules, M averaging, Morse behavior, and propagation.

P6.4-a/D-076 audits the remaining experimental SymTop skeleton before removal.
It has no production caller, differs from the independently referenced D-053
model in basis order/filtering, anharmonic semantics, and transverse phases,
and its dense and CSR dipole routes both fail direct execution. Three new
guards record those differences; no production formula changes.
P6.4-b/D-077 removes that broken legacy basis/dipole, factory, and `jmk`
helper without a compatibility shim. Direct tests now use the production
`models.symmetric_top` owner; D-053 physics references remain unchanged.
The three legacy-only audit tests were replaced with two ownership guards and
one obsolete factory test was removed, yielding 1217 passing tests. Strict
mypy (41 modules), active example smoke tests, sdist/wheel build, Twine, and
isolated wheel import pass; optional CUDA tests remain unverified.
P6.4-c/D-078 assigns the sole production linear-rotor kernel to LinMol,
moves the independent Wigner reference under tests, and removes an unused
rotation export. The shared dipole base and vibration formulas remain separate
follow-up units; no transition formula or production matrix path changes.
P6.4-d/D-079 moves the shared harmonic/Morse function files unchanged to
`models.vibration`, retaining one function object for LinMol and VibLadder.
SymTop remains independent. The old path is removed; CUDA remains unverified.
P6.4-e/D-080 moves the unchanged concrete dipole cache/unit/persistence mixin
to `models.dipole_base` and adds a type-only `core.dipole.DipoleOperator` for
propagation and spectroscopy. Both dynamics reverse imports are gone. A
pre-move test freezes conversion, cache identity, SI views, and the legacy
fallback. Strict mypy now covers 42 modules; CUDA remains unverified.
P6.4-f/D-081 removes the now-empty `dipole` package and its eager root import.
The wheel contains no old package files; physics and numerical paths are
unchanged. Phase 6 remains open for the two exact `models -> dynamics.problem`
imports, to be characterized separately in P6.5.
P6.5-a adds direct model-component projection characterization without changing
source. Its identity, axis-order, and metadata-snapshot guards pass alongside
the full CPU suite; CUDA remains unverified.
P6.5-b/D-082 moves the byte-identical coupling/model contract definitions to
`core.model`. Dynamics keeps the same class objects; models has no upper-layer
imports. Strict mypy covers 43 modules, and active examples plus wheel checks
pass. Phase 6 acceptance audit and real CUDA verification remain separate.
P6.6-a characterizes the remaining simulation-owned fixed-M LinMol basis
before its ownership-only move. Basis order, mapping, Hamiltonian, and invalid
M behavior are frozen; the M-average workflow is unchanged.

P6.2-c
implements D-069:
the frozen schema now belongs to `models/vib_ladder`, and the unused mapping
and stateless dipole wrappers are removed after a complete caller audit. The
remaining typed builders use the same validation, unit conversion, basis,
Hamiltonian, and dipole implementation. All D-067 values remain unchanged,
P6.2 is complete. P6.2-b implements
D-068:
the VibLadder basis, dipole class, stateless builder, and production builders
now share `models/vib_ladder/`; the three former owners are removed and the
generic dipole factory rejects the moved model rather than creating a new
reverse dependency. All D-067 values remain unchanged. P6.2-a implements D-067:
nine new contracts freeze VibLadder parameter/unit projection, basis/state
order, Hamiltonian, scalar-z coupling, harmonic dipoles, dense/CSR storage,
cache identity, and all current builder paths. Together with the existing
independent Morse and propagation references, 61 focused cases pass without a
source implementation change. P6.1-d implements D-066:
the frozen schema joins `models/two_level`, TwoLevel is removed from the legacy
generic dipole factory, and the unused mapping/stateless builders are deleted.
All D-063/D-064 numerical references remain unchanged, completing P6.1.
P6.1-c implements D-065:
the basis, dipole, stateless dipole builder, and production builders now have
one owner under `models/two_level/`; all three superseded paths are removed and
the D-063/D-064 numerical references remain unchanged. P6.1-b implements D-064:
reduced Planck's constant is derived once as `H/(2*pi)`, all runtime aliases
share it, and Hamiltonian conversions use the central converter. The approved
correction changes the active typed TwoLevel population by at most
`1.142e-13`; dense/CSR disagreement remains `6.78e-20`. Four new contracts
cover constant authority and energy/dipole round trips. P6.1-a implements D-063:
the TwoLevel production parameter projection, basis/state mapping, Hamiltonian,
dense/CSR dipoles, scalar-x coupling, and builder parity are fixed by eight
pre-move characterization cases. Its O-013 finding is resolved by D-064.
P5.4-a implements D-062:
all CPU-verifiable Phase 5 rows now have executable evidence. NumPy CSR split
is exactly equal to dense spectral execution, prepared split kernels exactly
match public results, and an existing device state crosses result finalization
by identity. The dead missing-Numba fallback is removed because Numba is a
required dependency. Phase 5 remains open: current CuPy RK4 and split helpers
round-trip through host memory, and real CUDA parity is unverified.
P5.1-c implements D-061:
the dense Liouville RK4 kernel reuses the exactly shared right/next-left
endpoint Hamiltonian. Multiple dimensions and both output modes are bitwise
equal to the retained pre-change Numba loop. The single-thread benchmark
records modest 1.019x-1.044x median speedups and explicitly distinguishes
analytical eliminated allocation traffic from process RSS. A slower
sub-ulp-different full-buffer experiment was rejected. The complete suite is
1188 passed with 10 optional-GPU skips and branch coverage remains 75%.
P5.1-b implements D-060:
validated Liouville wrappers now delegate to a prevalidated dense NumPy/Numba
kernel. Field-stage indices, RK4 order, stride writes, shapes, and caller-array
immutability are fixed. This move makes no speed claim. Strict mypy covers 37
named modules. P4.3-q implements D-059: Local control requires
an explicit `seed_field` or `none` initialization. The seed branch requires a positive direct-amplitude value/unit
pair and segment count while preserving the active 1000 V/m, five-segment
field, trigger, clipping, grid, endpoint, and RK4 behavior. The no-seed branch
injects nothing and raises before propagation when the existing mode-specific
initial trigger identifies a zero-control fixed point; it never falls back.
Strict mypy now covers 36 named modules. P4.3-p implements D-058:
Local control gain now requires a finite positive value/unit pair and converts
to canonical `(V/m)^2 fs` before the unchanged update equations. The active
example preserves `1e21` canonically as `1000 (GV/m)^2 fs`. Result
diagnostics distinguish the field-fluence proxy from the objective and expose
field scale and clipping without changing the frozen Local grid or control
logic. P4.3-o implements D-057:
optimizer option values now reject implicit type conversion, mode fallback,
duplicate axes, invalid spectral constraints, and suppressed Local failures.
Nonnegative spectral alpha uses exact `1+alpha` division, while valid Local
time/index/update behavior and Krotov references remain unchanged. Strict mypy
now covers 35 named modules. P4.3-n implements D-056:
optimization documents now use closed root and per-algorithm schemas, require
explicit ordered control axes, honor required YAML output/plot policy with
documented API precedence, and raise top-level requested plotting failures.
Thirteen incomplete v0.2 YAML files are archived without invented dipoles;
three current-schema configs remain active with explicit dipole units and axes.
All optimizer kernels and local time/index contracts remain unchanged, and
strict mypy covers 34 named modules. P4.3-m implements D-055 by moving the
incomplete v0.2 optimization YAML files to an explicit archive without
inventing missing dipoles. P4.3-l implements D-054:
optimization now consumes the same frozen LinMol, VibLadder, and TwoLevel
parameters and model-owned basis/Hamiltonian/dipole builders as normal
simulation. Strict value/unit pairs and exact quantum-number state tuples
replace legacy optimizer model names and implicit M repair. Basis order, H0,
SI dipoles, every optimizer time/index contract, and the stored Krotov result
remain characterized; optimizer kernels are unchanged. Incomplete tracked
legacy optimizer configs receive no invented dipole values. P4.3-k implements
D-053:
normal simulation now has an independent rigid parallel-band SymTop model with
signed `|v,J,K,M>` ordering, explicit CH3F ortho/para filtering, two rotational
constants, two vibration-rotation couplings, and Cartesian rank-one Wigner
dipoles. NumPy dense/CSR RK4 and strict scaling are verified; CuPy, split
operator, all-isomer pure states, and optimization raise explicitly. The
legacy direct SymTop route and every optimizer numerical kernel are unchanged.
P4.3-j began D-052 with a strict model-layer symmetry foundation and named
presets for `H2`, `D2`, `T2`, `HD`, and `CH3F` without inferred constants or
weights. P4.3-i implements D-051:
the local optimizer now names its direct field limit and seed amplitude
`field_max_v_per_m` and `seed_amplitude_v_per_m`. Exact characterization keeps
the defaults, seed-before-componentwise-clipping order, stored and propagated
field arrays, odd grid, shared endpoints, slices, indices, and RK4-consumed
prefix unchanged. Class-D optimizer quantities remain unresolved. P4.3-h
implements D-050:
Krotov initial fields require an explicit generated/sampled source, physical
value/unit pairs, and strict exact-grid sampled injection. Frozen seed samples,
update indices, and V=0 to V=3 fidelities are unchanged. P4.3-g unit 3 implements
D-049 and completes the low-level field-unit boundary: generated pulses require
time, carrier, and direct amplitude units, while GDD/TOD are complete optional
pairs or exact zero. Frozen nonzero-dispersion samples remain unchanged and
cross-unit waveforms agree within conversion roundoff. Unit 2 implements D-048
for direct construction and canonical fs/V/m storage; unit 1 implements D-047
for arbitrary arrays. Optimizer grids, indices, and values remain unchanged.
P4.3-f implements D-046:
spectroscopy conditions, wavenumber grids, and device resolution now require
exact unit labels and numerical code consumes frozen canonical values. All
spectroscopy formulas and routing remain unchanged. P4.3-e implements D-045:
every normal-simulation scalar physical input now has an explicit paired unit,
and frozen quantity schemas convert once to fs, V/m, C*m, fs^2, fs^3, or
rad/fs. Direct mappings, Python files, batch expansion, and saved parameter
JSON retain the caller values and labels. The stale-label `ParameterProcessor`
and its route-dependent value mutation are deleted. Equivalent ps/MV/cm and
canonical fs/V/m inputs retain sampled-field and population agreement.
P4.3-d implements D-043:
spectral modulation uses a unit-aware physical
delay and explicit phase/amplitude multipliers, GDD/TOD use physical Taylor
coefficients, intensity is cycle-averaged input to peak field, and unit
validation has no heuristic warning or raw-attribute fallback. D-044 reduces
the supported example set to three typed smoke-tested scripts; former v0.2
scripts are explicit archives. Active source, tests, examples, benchmarks, and
scripts are Ruff-clean and format-clean. The preceding D-041 unit 7 added
explicit, report-only convergence assessment.
The caller supplies both grids, the named observable, and tolerance; maximum
absolute difference is reported without changing either calculation. Unit 6
made the final normal-simulation mapping strict. Unknown
keys, model/field/algorithm-inapplicable keys, dummy scalar polarization, and
misplaced split selectors now fail before allocation. Imported Python modules
are excluded from parameter mappings, while non-module helper values remain
visible to strict validation. Generated and externally injected fields converge
to the immutable `SimulationCase`; normal model and fixed-M construction consume
the frozen model schema directly.
Exact field/population references remain unchanged, and strict mypy now covers
23 named modules. The preceding D-042 unit introduced frozen LinMol,
VibLadder, and TwoLevel parameter schemas with neutral frequency names and
required paired units. Model input is validated before allocation and projected
once to the unchanged low-level rad/fs constructors. PHz, THz, wavenumber, and
rad/fs forms retain Hamiltonian, dipole, and fixed-M propagation equivalence.
Generated and injected field parity remains exact. Direct basis, propagation
kernels, optimization paths, and `LocalOptimizerLegacyGridV1` remain unchanged.
Phase 3 completed under P3.2-b; all discovered modules import and the top-level
import graph has no cycles.
P3.2-a had audited the Phase 3 structure and removed the empty
simulation manager, obsolete writable-array time-grid adapter, and two uncalled
`ParameterProcessor` construction helpers. The cleanup routes callers through
the already canonical immutable `TimeGrid`, removes the last `core -> fields`
reverse dependency, and changes no numerical kernel or physical formula.
P3.1-h had moved all plotting helpers unchanged from `plots/` to
`visualization/`, while root import remained independent of optional Matplotlib.
P3.1-g had moved checkpoint, serialization, and summary persistence
unchanged from `simulation/` to `io/`. File formats, filenames, deduplication,
and overwrite behavior remain fixed by contract tests. P3.1-f had moved model
construction unchanged to `models/` and kept the unchanged M-average
propagation workflow at `simulation/m_average.py`.
P3.1-e had moved generic validation unchanged to `core/validation.py`, P3.1-d
had moved nondimensionalization unchanged to `dynamics/scaling/`, and P3.1-c had moved propagation unchanged to `dynamics/`, P3.1-b had moved the unchanged electric-field implementation to
`fields/`, and P3.1-a had moved the unchanged `Hamiltonian` implementation to
`core/operators.py`. These checkpoints built and checked the distributions and
verified the new and removed module paths in isolated wheel installs. The
preceding P2.5 checkpoint reran the seven 4001-point CPU workloads through the
typed public boundary: every final state was exactly equal to the committed
Numba-CSR reference. The fixed 0.10 ms result-contract overhead is material
only for the two-level dense micro-workload and is documented in D-039. The 47%
gate intentionally starts at the accepted Phase 0 baseline; raise it in a
dedicated coverage checkpoint after target ownership and omit policy are
stable. The old README claim of 63% coverage is stale.

### Largest source hotspots

| File | Physical lines | Main concern |
|---|---:|---|
| `spectroscopy/absorbance_calculator.py` | 1,049 | Multiple response/spectrum responsibilities despite 90% measured coverage |
| `simulation/runner.py` | 628 | Construction, execution, multiprocessing, output, error handling |
| `dynamics/scaling/converter.py` | 562 | Strict scaling, array conversion, and object preparation |
| `dynamics/algorithms/rk4/schrodinger.py` | 478 | Dense, sparse, CPU, GPU, validation paths in one module |
| `fields/field.py` | 478 | Field state, pulse construction, polarization, unit conversion |
| `dynamics/utils.py` | 416 | Backend, units, field mapping, nondimensional preparation |

Line count alone does not require splitting; mixed responsibility and poor
testability do.

## Known repository hygiene problems

- P1.1 removed tracked coverage databases, historical runner-test output, and
  the Notebook checkpoint; saving runner tests now use pytest temporary paths.
- P1.2-A removed the approved root print scripts, two empty RK4 files, the
  disabled old-API split test, and one obsolete migration validator after
  migrating or confirming overlap for their useful assertions.
- P1.2-B removed the approved standalone diagnostics and redundant pytest
  wrapper after recording every collected replacement in
  `VALIDATION_INVENTORY.md`; the ignored generated PNG files were preserved.
- P1.3 removed `dipole/rot/jm_old.py` after Wigner-3j equivalence testing,
  removed unused deprecated builder wrappers and the redundant old-basis demo,
  and retained one spectroscopy archive as explicit Phase 7 migration evidence.
- The competing nondimensional compatibility modules have been removed; the
  strict converter remains pending its target-package move.
- P1.5-B corrected spectroscopy Jones-bra detection, enabled every selected
  Cartesian component, removed the implicit post-projection orientational
  `1/3`, and restricted Doppler broadening to the two transition-specific
  routes.
- P1.5-C replaced the implicit vibrational mask with required `pump_probe`
  (`V_i == V_j`) and `unfiltered` modes. Same-V rotational/M coherence is
  retained, discarded density norm is reported, missing labels raise, and
  post-probe radiation/PFID remains unfiltered. The combined polarization,
  broadening, pathway, and failure contracts are anchored by 26 focused tests.
- P1.6 consolidated duplicate CI workflows and validated mandatory Ruff, limited
  mypy, Python 3.10-3.13 pytest, physics/contracts, 47% branch coverage, build,
  and clean-wheel import gates locally and in GitHub Actions run #47.
  Refactor-branch pushes and explicit manual runs execute that same policy;
  `main` requires the aggregate `Required CI gates` check with strict branch
  synchronization and administrator enforcement.
- Several README examples reference removed or moved APIs.

Each item must be classified as migrate, replace, archive outside the package,
or delete. Do not delete a legacy implementation until unique formulas have
been compared with the replacement.

## Completed preparatory work

| Commit | Result |
|---|---|
| `7ce9419` | Modularized simulation construction and hardened propagation contracts |
| `af21fbe` | Aligned density propagation options, backend honesty, time return, validation |
| `613ce93` | Unified `-mu E` sign and added physical density matrix validation |
| `3b081e1` | Added the reproducible, non-blocking Phase 0 benchmark recorder |
| `6e154ec` | Replaced the Python/SciPy sparse RK4 loop with explicit Numba CSR propagation |
| `93ee9eb` | Made Cartesian and helicity-projected split interactions explicit and physically tested |
| `7d14fda` | Required physical inputs and consolidated strict nondimensionalization |
| `e4102d0` | Made spectroscopy conditions, exact/approximate/auto routes, broadening, and execution reports explicit |
| `82c8d76` | Enforced mandatory quality, physics, coverage, build, and wheel-import CI gates locally |
| `d5c56fc` | Enabled pre-PR refactor-branch and explicit manual CI execution |
| `62e6bfd` | Pinned Ruff and passed every remote required gate in Actions run #47 |
| `834f8ef` | Corrected complex spectroscopy polarization and passed Actions run #49 |
| `874b1c4` | Made pump-probe V-pathway selection explicit and passed Actions run #51 |
| `53bfb2c` | Introduced the typed TimeGrid for normal simulation and passed Actions run #53 |
| `965dcda` | Preserved the exact local-optimizer legacy time layout |
| `b211610` | Made dimensional NumPy RK4 backward direction explicit for Krotov |
| `873ad6e` | Made model coupling and complete physical propagation input one immutable `PropagationProblem` |

These commits are the starting point, not the final architecture.

## Phase status

| Phase | Name | Status |
|---|---|---|
| 0 | Physics characterization baseline | Complete — P0.1-P0.7 CPU baseline recorded; CUDA remains unverified |
| 1 | Repository and CI normalization | Complete — local and GitHub gates pass; `main` requires `Required CI gates` |
| 2 | Typed propagation contracts | Complete — P2.1-P2.5; one typed problem/options input and one backend-explicit endpoint-complete result |
| 3 | Target package migration | Complete — P3.1-a through P3.2-b establish target owners, remove superseded paths, and eliminate top-level cycles |
| 4 | Units and nondimensionalization | Complete for decided contracts — Class-D optimizer values and adaptive integration explicitly deferred |
| 5 | Numerical dynamics engine | In progress — CPU acceptance verified by P5.4-a; backend-native CuPy execution and real-CUDA parity remain |
| 6 | Model consolidation | In progress — P6.1-P6.5-b complete; P6.6 acceptance audit remains |
| 7 | Simulation, optimization, spectroscopy decomposition | Pending |
| 8 | Public API, documentation, and release | Pending |

Status must be updated only when the acceptance criteria in
`EXECUTION_PLAN.md` are met.

## Conflict resolution

When code, tests, released documentation, and these documents disagree:

1. Determine which behavior the current tests actually enforce.
2. Compare the behavior with `PHYSICS_CONTRACTS.md`.
3. Check `DECISIONS.md` for an explicit user decision.
4. If the answer changes physics or scientific interpretation, ask the user.
5. Add the resolution to `DECISIONS.md` before or with implementation.
6. Update stale public documentation after tests pass.
