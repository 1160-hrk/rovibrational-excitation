# Physics and numerical contracts

Last verified against source and tests: 2026-08-27
Baseline commit: `613ce93`

## Scope and authority

This document records scientific behavior that must survive refactoring. It is
more authoritative for the v0.3 refactor than old examples or README text.
Changing a decided contract requires:

1. explicit user approval;
2. a decision-log entry;
3. a regression or characterization test;
4. an isolated physics-change commit.

A file move or interface cleanup must not change any contract in this document.

## 1. Hamiltonian and evolution equations

The interaction Hamiltonian convention is:

~~~text
H(t) = H0 - sum_a mu_a E_a(t)
~~~

For scalar coupling:

~~~text
H(t) = H0 - mu_axis E_scalar(t)
~~~

The internal propagation Hamiltonian is expressed as angular frequency, so
`hbar` is absorbed after conversion to `rad/fs` or into nondimensional units.

Schrödinger evolution:

~~~text
d psi / dt = -i H(t) psi
~~~

Liouville-von Neumann evolution:

~~~text
d rho / dt = -i [H(t), rho]
~~~

Schrödinger and Liouville must use the same `-mu E` sign. For a pure initial
state `rho0 = |psi0><psi0|`, both solvers must agree within the expected
integration tolerance.

Regression anchor:
`tests/contracts/test_density_solver_contracts.py::test_liouville_matches_schrodinger_for_a_pure_state`.

## 2. Canonical units at the propagation boundary

The current dimensional propagation boundary uses:

| Quantity | Canonical kernel unit |
|---|---|
| Time | fs |
| Free Hamiltonian `H0` | rad/fs |
| Electric field | V/m |
| Dipole coupling matrix | rad/fs/(V/m) |
| Wavefunction | dimensionless complex amplitude |
| Density matrix | dimensionless complex matrix |

Input objects may accept J, eV, cm^-1, THz, Debye, C*m, intensity units, and
other documented units. Conversion must happen before entering a numerical
kernel. A kernel must not inspect unit strings or model classes.

Target rule:

~~~text
external units -> validated domain object -> one conversion boundary
               -> canonical arrays -> numerical kernel
~~~

No parameter may be converted more than once. Conversion functions must not
mutate caller-owned arrays or model parameters.
Intensity inputs are cycle-averaged. Their canonical conversion returns peak
electric-field amplitude as `sqrt(2*I*mu_0*c)`.

Strict propagation-unit validation checks only formal access to canonical units,
compatible matrix/sample shapes, and a finite positive field-grid step. It does
not classify values by a typical molecular range, downgrade failures to
warnings, or fall back to raw unit-ambiguous attributes. Numerical adequacy is
reported only by an explicit convergence calculation.

Public normal-simulation scalar inputs retain their submitted value and an
individual explicit unit in the caller-owned mapping. The mapping is never
rewritten during loading or execution and is saved with those original pairs.
Frozen typed values convert once to the internal canonical system: time in fs,
peak electric field in V/m, dipole moment in C*m, GDD in fs^2, TOD in fs^3,
and angular frequency in rad/fs. Numerical code receives only canonical values.

Frequency-bearing public inputs use a neutral quantity name and a required
`*_units` field. `Frequency` converts a finite scalar exactly once to canonical
angular frequency in `rad/fs`. Hz through PHz and wavenumber inputs do not
contain `2π`; `rad/s`, `rad/ps`, and `rad/fs` inputs do. The generated-field
carrier keys are `carrier_frequency` and `carrier_frequency_units`; the old
unit-ambiguous `carrier_freq` configuration key is rejected. FFT bins remain
ordinary frequency in cycles/fs, so only an explicit `cycles_per_fs` view is
passed to spectral modulation.

Nondimensional propagation is a separate, explicit transformation. It must
produce a scale object sufficient to convert time and observables back to
physical units.

## 3. Time grid and RK sampling

`FIELD_INTERVALS_PER_PROPAGATION_STEP = 2` is a physical/numerical contract.

Definitions:

- `field_dt`: spacing between adjacent electric-field samples.
- `propagation_dt`: one state-vector or density-matrix update.
- `propagation_dt = 2 * field_dt`.
- Each RK4 update consumes field values at the left endpoint, midpoint, and
  right endpoint.
- For `n_steps` propagation updates, the field grid length is
  `2 * n_steps + 1`.
- Both endpoints must be present.

For dimensional propagation:

~~~text
field_time[j] = t_start + j * field_dt
state_time[k] = t_start + k * propagation_dt
~~~

The configured span must satisfy:

~~~text
(t_end - t_start) / (2 * field_dt) is an integer
~~~

Invalid, non-finite, zero, or negative time steps must fail before propagation.

Legacy low-level return-time behavior:

- Full trajectory starts at `t_start`.
- With `sample_stride = s`, adjacent returned states are separated by
  `s * propagation_dt`.
- Final-state-only propagation returns a one-element time array containing
  `t_end`.
- Dimensional and nondimensional paths must return the same physical time axis
  in femtoseconds.

Legacy low-level kernels record the initial state and states at steps divisible
by the stride. If `n_steps` is not divisible by the stride, their regular
trajectory omits the endpoint. The typed propagation boundary always appends
the exact endpoint in that case, producing one shorter final output interval
without changing any integration step or field sample. Low-level shapes remain
characterized during the Phase 2 migration.

Backward dimensional RK4 is an explicit direction, not a decreasing public
field grid. `PropagationDirection.BACKWARD` reverses the field samples, sends
`-propagation_dt` to the unchanged RK4 kernel, and reports state times from
`t_end` toward `t_start`. Strings and numeric signs are rejected. Backward
split-operator and nondimensional modes are unsupported until independently
characterized and must raise.

Primary implementation anchors:

- `core/time.py` (`TimeGrid`, the canonical and sole normal-simulation
  time-grid owner; the legacy writable-array adapter was removed in P3.2-a)
- `dynamics/utils.py`
- `dynamics/schrodinger.py`
- `dynamics/liouville.py`

### 3.1 Strict nondimensional generator

The nondimensional path analyzes the complete active generator in SI units
before converting arrays. Let epsilon_0 be the smallest eigenvalue of H0,
Delta_H its spectral span, mu_ref the largest operator 2-norm among the dipole
components selected by coupling_axes, and E_peak the peak active field
magnitude. Cartesian coupling uses max_t sqrt(sum_a |E_a(t)|^2); scalar
coupling uses max_t |E_scalar(t)|.

~~~text
H0' = (H0 - epsilon_0 I) / E_ref
mu_a' = mu_a / mu_ref
E_a' = E_a / E_peak
lambda_num = mu_ref E_peak / E_ref
tau = (t - t_start) / (hbar / E_ref)
E_ref = max(Delta_H, mu_ref E_peak)
~~~

When an explicit positive energy_scale_J is supplied, only the last E_ref
selection is replaced. All other quantities and diagnostics retain their
physical definitions. The physical ratio mu_ref E_peak / Delta_H is never
replaced by lambda_num: it is None for a driven gapless problem.

Boundary behavior is part of the physics contract:

| Input state | Result |
|---|---|
| Ordinary ElectricField with zero samples | error; caller must choose ZeroField |
| ZeroField and nonzero free span | field scale inactive; field-free propagation |
| Driven field and zero active dipole operator | error |
| ZeroField and zero dipole operator | both interaction scales inactive |
| H0 proportional to identity, nonzero offset | use absolute offset only to carry phase |
| Completely zero generator | low-level error; no characteristic scale exists |
| Non-finite or non-Hermitian operator | error before scaling |

Inactive scales are represented by None, never by a nonphysical divisor.
Normalized arrays for inactive terms are exact zeros. Scale metadata records
the value, source (derived, explicit, or inactive), and derivation method.

Energy-origin centering changes a state vector by a global phase. For every
returned physical elapsed time Delta_t, the high-level Schrodinger path
multiplies the centered result by

~~~text
exp(-i epsilon_0 Delta_t / hbar).
~~~

The initial trajectory state therefore has phase one, sampled trajectory
phases use the actual propagation stride, and final-only output uses the field
endpoint. Density matrices receive no phase operation because it cancels
between ket and bra. Absolute wavefunction parity is tested, not inferred from
population parity.

The electric-field grid remains the only integration-grid source. No
nondimensionalization function may resample it. Legacy auto_timestep,
target-accuracy recommendations, 1000 fs caps, and invented 1 fs, 1 Debye, or
1e8 V/m scales are forbidden. A convenience time-array builder accepts an
explicit positive dt whose interval count divides the duration; otherwise it
raises rather than rounding or extending the endpoint.

Implementation anchors:

- fields/field.py: ZeroField;
- dynamics/scaling/converter.py: validation, centering, and scale derivation;
- dynamics/scaling/scales.py: values and provenance;
- dynamics/schrodinger.py: global-phase restoration;
- tests/contracts/test_strict_nondimensional_contracts.py: reference contracts.
### 3.2 Explicit physical inputs

Missing and zero are distinct. A physical zero is accepted only when supplied
as `0.0`; no constructor or runner may invent a model constant or pulse width.

Required simulation fields are model-specific:

| Model | Required physical/model fields |
|---|---|
| LinMol | `V_max`, `J_max`, `vibrational_frequency` + `_units`, `anharmonic_shift` + `_units`, `rotational_constant` + `_units`, `vibration_rotation_coupling` + `_units`, `dipole_scale` + `_units`, `potential_type` |
| VibLadder | `V_max`, `vibrational_frequency` + `_units`, `anharmonic_shift` + `_units`, `dipole_scale` + `_units`, `potential_type` |
| TwoLevel | `energy_gap`, `energy_gap_units`, `dipole_scale`, `dipole_scale_units` |
| SymTop | `molecule`, `nuclear_spin_isomer`, `V_max`, `J_max`, all six frequency quantities with explicit `_units`, `dipole_scale` + `_units`, and `potential_type` |

Every simulation case also states `duration` explicitly. The removed
`pulse_duration` name is rejected. Direct basis construction requires
LinMol `omega`, `B`, `alpha`, and `delta_omega`; VibLadder `omega`
and `delta_omega`; or TwoLevel `energy_gap`. Production SymTop construction
uses the frozen schema in the table above; its six frequency quantities are
vibrational frequency, anharmonic shift, perpendicular/parallel rotational
constants, and perpendicular/parallel vibration-rotation couplings. The legacy
`core.basis.SymTopBasis` is not its input path. Direct legacy dipole construction
requires `mu0`. Krotov optimization requires a
positive finite `duration_initial` before any model or field work begins.

For raw-array nondimensionalization, `H0_units` and `time_units` are
required keywords. For object-based nondimensionalization, `coupling_axes`
and `scalar_coupling` are required. These values cannot be inferred from
array shapes without changing the physical interpretation.

Regression anchors:

- `tests/test_basis_unified_units.py::test_basis_requires_physical_constants`;
- `tests/contracts/test_physical_input_contracts.py`;
- `tests/contracts/test_simulation_contracts.py`;
- `tests/test_simulation_models.py`.

## 4. Initial-state semantics

### 4.1 Normal simulation runner: coherent superposition

A list passed as `initial_states` to the normal simulation model builder is a
list of basis indices. The runner:

1. assigns amplitude `1` to every selected basis state;
2. forms an equal-amplitude, equal-phase coherent sum;
3. normalizes the resulting state vector.

An empty list is an error.

This behavior is intentionally coherent. It must not be converted to a
population sum.

### 4.2 Typed pure state

`PureState` requires a finite, nonempty, one-dimensional complex vector with
norm one within `100 * n * eps_float64`. It stores an exact defensive copy and
does not normalize caller input. Normalization requested by a workflow must be
performed explicitly before typed construction. `SchrodingerPropagator` accepts
only this type at its public boundary and passes `amplitudes` unchanged into the
existing array calculation.

### 4.3 Incoherent ensemble

An incoherent mixture constructs an `IncoherentEnsemble` from an iterable
of state vectors and passes that explicit type to `MixedStatePropagator`.

For each vector `psi_i`:

~~~text
raw_weight_i = <psi_i | psi_i>
normalized_state_i = psi_i / sqrt(raw_weight_i)
weight_i = raw_weight_i / sum_j(raw_weight_j)
rho(t) = sum_i weight_i |psi_i(t)><psi_i(t)|
~~~

Zero-norm vectors are skipped. An empty ensemble or an ensemble containing only
zero-norm vectors is an error. All vectors must have the same dimension.

This design allows callers to encode a desired raw weight `w_i` as
`sqrt(w_i) * normalized_psi_i`.

### 4.4 Explicit density matrix

The typed `DensityState` boundary requires trace one within the scale-aware
tolerance in Section 5. It never normalizes, clips, symmetrizes, or otherwise
repairs caller input.

`LiouvillePropagator` and `MixedStatePropagator` accept this explicit type and
pass its stored matrix unchanged into the existing Liouville array calculation. A raw square array is rejected; array shape
is never used to infer the state kind.

Optimization is a deliberate internal exception during migration: GRAPE, Krotov,
and the local optimizer pass intermediate ndarray results through the private
`_propagate_array` bridge. Constructing `PureState` between steps is forbidden
because validation or normalization could alter the characterized calculation.
The local optimizer preserves the same odd prefixes, shared endpoints, and array
objects.

### 4.5 Solver renormalization

Wavefunction renormalization is an explicit production policy. Typed options
must require the caller to select disabled or per-step renormalization and must
record that selection in the result. It must never be silently enabled because
it can hide integration error. Legacy low-level calls retain their
`renorm=False` default during migration.

## 5. Density-matrix validation

A density matrix must be:

- square;
- finite;
- Hermitian within numerical tolerance;
- positive semidefinite within numerical tolerance;
- have positive real trace.

For a complex128 density matrix `rho` with dimension `n`, the tolerance is:

~~~text
tol = 100 * max(1, n) * eps_float64 * ||rho||_2
~~~

where `||rho||_2` is the spectral norm.

Validation rules:

~~~text
abs(Im(trace(rho))) <= tol
||rho - rho_dagger||_2 <= tol
min(eigvalsh((rho + rho_dagger) / 2)) >= -tol
Re(trace(rho)) > 0
~~~

Values inside the threshold are accepted as numerical roundoff. The
implementation must not silently:

- clip negative eigenvalues;
- symmetrize the matrix;
- project onto the positive cone;
- replace the trace;
- otherwise modify the matrix.

`DensityState` additionally requires trace one at construction.
`MixedStatePropagator` performs no subsequent normalization or repair.

Primary implementation:
`core/validation.py`.

## 6. Vibrational ladder and Morse potential

### 6.1 Vibrational energy convention

The configured `omega` is `omega01`, the angular frequency of the fundamental
`v=0 -> 1` transition. The positive anharmonic shift `delta_omega` is the
decrease in adjacent transition frequency per vibrational level:

~~~text
x = v + 1/2
E_v = (omega01 + delta_omega) x - (delta_omega / 2) x^2
E_(v+1) - E_v = omega01 - v delta_omega
~~~

For `delta_omega = 0`, this reduces to the harmonic ladder
`E_v = omega01 (v + 1/2)`.

`VibLadderBasis.generate_H0()` and `generate_H0_with_params()` must use one
shared implementation of this formula. Their numeric results must agree when
given the same physical parameters and output units.

Primary implementation:
`core/basis/viblad.py`.

### 6.2 Morse bound levels

A Morse potential requires nonzero anharmonic shift. Therefore:

~~~text
potential_type == "morse" and delta_omega == 0 -> ValueError
~~~

The Morse level parameter is derived for each configured model or dipole
instance:

~~~text
N = (omega01 + delta_omega) / delta_omega - 1/2
~~~

It must never be global mutable state and must not be replaced by a fixed
`N=200`. A fixed value such as 200 may appear only in a test that explicitly
tests a chosen numerical example.

Maximum allowed vibrational basis index:

~~~text
V_max <= floor(N) - 1
~~~

A basis exceeding the limit is an error.

Primary implementation:
`dipole/vib/morse.py`.

Required characterization cases:

- zero anharmonicity with Morse is rejected;
- two model instances with different `omega01` or `delta_omega` do not leak
  Morse parameters into each other;
- boundary `V_max = floor(N) - 1` is accepted;
- the next level is rejected;
- harmonic and Morse transition elements agree in their documented limiting
  regime without forcing exact equality.

### VibLadder Phase 0 reference anchor

`tests/physics/test_vib_ladder_reference.py` fixes these energy parameters:

- `omega01 = 1.2 rad/fs`, `delta_omega = 0.08 rad/fs`, and `V_max = 4`;
- harmonic and anharmonic energies use independent closed-form references;
- stored and temporary-override construction must produce the same matrix.

Morse derivation uses two distinct pairs:

- `(omega01, delta_omega) = (1.0, 0.1) rad/fs`, giving `N = 10.5`;
- `(omega01, delta_omega) = (0.9, 0.2) rad/fs`, giving `N = 5.0`.

The first pair accepts `V_max = 9` and rejects `V_max = 10`. Constructing the
second instance before evaluating the first protects against shared mutable
Morse state. Adjacent Morse elements at `N=500` must be closer to their
harmonic values than at `N=50`; no arbitrary closeness threshold is imposed.

The workflow reference uses `V_max=2`, `omega01=0.37 rad/fs`,
`delta_omega=0.015 rad/fs`, `mu0=2e-29 C m`, a `5e8 V/m` constant field,
field spacing `0.001 fs`, propagation spacing `0.002 fs`, and final time
`0.1 fs`. It covers x, y, diagonal linear, and circular complex polarization.

Energy references use `atol=2e-15`; scalar-polarization trajectories use
`atol=2e-14`; physical time uses `atol=2e-15`; and dimensional/nondimensional
population parity uses `atol=2e-12`.

## 7. Model coupling and polarization

Current simulation model capabilities:

| Model | Coupling mode | Current implementation axis | Physical polarization behavior |
|---|---|---|---|
| LinMol | Cartesian | Configurable two-axis mapping, default `xy` | Depends on field polarization |
| TwoLevel | Scalar | `x` | Independent of supplied polarization direction |
| VibLadder | Scalar | `z` | Independent of supplied polarization direction |
| SymTop | Cartesian parallel-band coupling | Required explicit Cartesian axes | Polarization dependent; NumPy dense/CSR RK4 only |

For TwoLevel and VibLadder, `x` and `z` are current storage axes, not physical
polarization degrees of freedom. The target architecture should expose scalar
coupling directly so callers do not need a dummy polarization vector.

A supplied polarization may still be normalized and structurally validated by
the input layer. It must not change scalar-model excitation results.

### Molecular symmetry and nuclear-spin statistics

The D-052 foundation keeps four concepts separate:

1. geometric point group;
2. optional molecular permutation-inversion group;
3. rotational-state quantum numbers and vibronic symmetry;
4. a source-identified nuclear-spin classification policy.

A point-group label alone never determines allowed rotational states or
statistical weights. Initial policies apply only to the explicitly supported
totally symmetric vibronic manifold. They classify the caller's state without
changing `v`, `J`, or signed `K`.

| Preset | Point group | Sector rule | Statistical weight |
|---|---|---|---|
| `H2` | Dinfh | even J para; odd J ortho | 1; 3 |
| `D2` | Dinfh | even J ortho; odd J para | 6; 3 |
| `T2` | Dinfh | even J para; odd J ortho | 1; 3 |
| `HD` | Cinfv | one unfiltered sector | 6 for every J |
| `CH3F` | C3v, molecular group C3v(M) | K divisible by 3 ortho; otherwise para | unresolved until signed-K symmetry adaptation |

CH3F uses `abs(K) mod 3` only for sector selection. Its preset must raise if a
caller asks for a numerical weight. A zero weight may eventually remove a
state, while a positive weight is population data for a thermal or explicit
incoherent construction. A weight is never interpreted as a coherent
amplitude.

Molecule-name resolution is not a molecular-constant database. Presets contain
no Hamiltonian constants, dipoles, temperature, field parameters, or units;
those remain required physical inputs. Unknown molecule names never select a
generic symmetry or model.

The first production SymTop model is a rigid, nondegenerate parallel band in
`|v,J,K,M>` with signed K/M and deterministic storage order `v,J,K,M`. For
`x=v+1/2`, its angular-frequency Hamiltonian is

~~~text
E_vib(v) = (omega01 + delta_omega) x - delta_omega x^2 / 2
B_perp(v) = B_perp - alpha_perp x
B_parallel(v) = B_parallel - alpha_parallel x
E(v,J,K) = E_vib(v) + B_perp(v) J(J+1)
           + [B_parallel(v) - B_perp(v)] K^2.
~~~

External equation anchors are the NIST symmetric-top energy and selection-rule
[overview](https://physics.nist.gov/PhysRefData/MolSpec/Hydro/Html/sec2.html)
and the rank-one matrix element in Eq. 67 of
[Wall et al.](https://arxiv.org/abs/1305.1236). The implemented spherical form is

~~~text
<J'K'M'|D^1_{p0}*|JKM>
 = (-1)^(M'-K) sqrt[(2J'+1)(2J+1)]
   (J' 1 J; -M' p M) (J' 1 J; -K 0 K) delta_K'K.
~~~

The parallel-band direction cosine is the product of the standard M and K
Wigner-3j factors with `delta_K'K`. Cartesian conversion gives Delta K=0;
Delta M=0 for z and Delta M=+/-1 for x/y. The rank-one factors enforce
Delta J=0,+/-1 and remove J=0 to J=0. Harmonic vibration couples adjacent
levels; Morse vibration retains the accepted overtone elements and derives its
finite bound-level parameter per model instance.

Construction requires a named symmetric-top preset and exactly one pure
`ortho` or `para` sector. It filters the basis but never applies a statistical
weight. `nuclear_spin_isomer="all"` is rejected because one state vector must
not coherently combine nuclear-spin species; a future explicit incoherent
mixture may combine separately propagated sectors. CH3F is the first supported
preset. All constants and units remain caller supplied.

Independent SymPy Wigner-3j references cover low-J elements through J=3.
Hamiltonian references cover both vibration-rotation couplings and signed-K/M
degeneracy. Dense/CSR dipoles are exactly equal; dense/CSR and
dimensional/nondimensional RK4 populations agree. CuPy, split operator, and
optimization routes raise before allocation or propagation. The legacy
`core.basis.SymTopBasis` and `dipole.symtop` remain experimental and are not
used by this production model.

### Split-operator polarization reference

M 位相回転、Hermitian 性、Strang 分割、計算量の詳しい導出は
[docs/CARTESIAN_SPLIT_OPERATOR.md](../CARTESIAN_SPLIT_OPERATOR.md) を参照する。

`split_interaction="cartesian"` is the physical reference. It consumes the
real Cartesian field arrays and therefore uses exactly the RK4 generator
`H0 - mu_x Ex - mu_y Ey`. For M-resolved LinMol xy dipoles, the tested
identity is

~~~text
D(phi) mu_x D(phi)^dagger = cos(phi) mu_x + sin(phi) mu_y
D_nn(phi) = exp(i M_n phi).
~~~

The implementation uses the field midpoint, `hypot(Ex,Ey)` and
`atan2(Ey,Ex)`. The M rotations are elementwise, while the fixed `mu_x`
eigensystem requires two dense matrix-vector products per step. Tests at
`dt=0.02` and `0.01` confirm the expected factor-four reduction of the
second-order error relative to RK4.

`split_interaction="helicity_projected"` is a separate approximation. It
keeps the upper-triangular one-way part of the complex Jones-weighted
transition dipole and adds its adjoint. There is no factor one half.
`(1,+i)/sqrt(2)` selects Delta M=+1 and `(1,-i)/sqrt(2)` selects Delta M=-1
under the library convention. Non-Hermitian component dipoles, unnormalized
Jones vectors, and nonzero diagonal transition terms are rejected.

Sparse operator inputs are accepted for parity, but spectral eigenvectors are
dense; this path does not claim sparse-memory scaling.

### TwoLevel Phase 0 reference anchor

`tests/physics/test_two_level_reference.py` fixes the following parameter set:

- `H0 = diag(0, 0.37)` rad/fs;
- transition dipole `mu = 2e-29 C m`;
- field sampling interval `0.001 fs`, hence RK4 propagation interval `0.002 fs`;
- final physical time `0.2 fs` and constant driven field `5e8 V/m`.

It independently checks:

- field-free superposition phases against
  `psi_n(t) = exp(-i E_n t) psi_n(0)`;
- Liouville evolution against the outer product of Schrödinger evolution;
- constant-drive RK4 against
  `expm[-i (H0 - mu E) T]`, including the interaction sign;
- scalar-polarization independence for x, y, diagonal linear, and circular
  complex polarization inputs;
- physical time and populations across dimensional and nondimensional paths;
- NumPy dense/CSR parity and a real NumPy/CuPy final-state parity case.

CPU tolerances are near the scale of the chosen fourth-order step: analytic,
polarization, and dense/CSR comparisons use `2e-14`; density equivalence uses
`3e-13`; the constant-drive matrix-exponential reference uses `2e-13`; time
equivalence uses `2e-15`; and nondimensional population equivalence uses the
largest absolute tolerance, `2e-12`.

The CuPy comparison uses `rtol=2e-12`, `atol=2e-13`, is marked `gpu`, and must
execute on real CUDA hardware before CuPy parity is considered verified.

## 8. Coherent and incoherent observables

A coherent initial superposition evolves one state vector and includes
interference terms:

~~~text
rho_coherent = |sum_i c_i psi_i><sum_j c_j psi_j|
~~~

An incoherent mixture evolves components independently and sums density
operators:

~~~text
rho_incoherent = sum_i w_i |psi_i><psi_i|
~~~

These are not interchangeable. A function or runner must state which
semantics it uses in its name, type, or required options. No generic list or array input may silently switch meaning based on
shape. `MixedStatePropagator` requires `IncoherentEnsemble` or `DensityState`.

### 8.1 LinMol fixed-linear M average

Decision D-017 defines two distinct LinMol workflows:

- `representation="m_resolved"`: explicit `|v,J,M>` basis, Cartesian
  x/y/z coupling, and physically direction-dependent polarization response;
- `representation="m_incoherent_average"`: reduced `|v,J>` output with an
  incoherent average over separately propagated fixed-M blocks.

For a reduced initial state with rotational number `J0`,

~~~text
w_M = 1 / (2 J0 + 1),              M = -J0, ..., J0
P_(v,J)(t) = sum_M w_M |psi_(v,J,M)(t)|^2
~~~

With fixed linear polarization the quantization axis is chosen along the field,
so the internal interaction is `-mu_z E_scalar` and every block conserves M.
The `+M` and `-M` z-coupling blocks are identical. The implementation
therefore propagates `|M|=0,...,J0` with representative weights

~~~text
W_0 = 1 / (2 J0 + 1)
W_|M| = 2 / (2 J0 + 1),            |M| > 0
sum_|M| W_|M| = 1
~~~

The full explicit dimension and each block dimension are:

~~~text
D_full = (V_max + 1) (J_max + 1)^2
D_M = (V_max + 1) (J_max - |M| + 1)
~~~

Dense matrix work is consequently proportional to `sum D_M^2` for the
required representatives rather than `D_full^2`. No full M-resolved dipole
matrix is constructed by the reduced runner.

A fixed linear Jones vector may be real or may contain one common complex
phase. After normalization, the common phase is removed and the remaining
imaginary norm must not exceed `128 * epsilon_float64`. A physical relative
complex phase is circular or elliptical and must raise. `axes` must also raise
in this mode because the laboratory direction is the internal quantization
axis.

An equal-amplitude coherent superposition across v is allowed when all selected
states share one J. A selection spanning multiple J values raises rather than
silently discarding cross-J coherence. Such an initial condition requires an
explicitly specified incoherent ensemble or a future rotational density
contract.

The reduced result is a mixed-state population trajectory, not a state vector.
Saved results contain `representation="m_incoherent_average"`, `abs_m`,
`m_multiplicity`, `m_weight`, and one `psi_abs_m_<M>` per representative.
They intentionally contain no aggregate `psi`.

Regression anchor:
`tests/physics/test_linear_molecule_reference.py` compares the reduced result
against a full M-resolved incoherent reference, verifies weight normalization
and reduced work, and separates explicit-M Cartesian response from reduced
direction-independent response.

## 9. Algorithm and backend capability matrix

Current verified/implemented contract:

The typed source of truth is `PropagationOptions`, containing one
`ExecutionPolicy`, plus `validate_execution_capability`. Normal simulation
configuration requires backend, storage, algorithm, trajectory, stride,
scaling, and renormalization; one validated object controls model construction
and propagation. Public propagation also requires explicit Cartesian axes or a
scalar coupling axis. Split propagation requires the same explicit interaction
mode at construction and call time. Legacy booleans and keyword adapters remain
only inside private numerical/optimizer migration boundaries.

| State path | Algorithm | NumPy dense | NumPy sparse | CuPy dense | CuPy sparse |
|---|---|---:|---:|---:|---:|
| Pure state | RK4 | Yes | Yes | Yes when CuPy is installed | No |
| Pure state | Split operator | Yes | Accepted | Yes when CuPy is installed | No |
| Incoherent pure-state ensemble | Delegates to selected pure solver | Same as pure solver | Same as pure solver | Same as pure solver | No |
| Explicit density matrix | Liouville RK4 | Yes | No | No | No |

Additional constraints:

- Split operator requires diagonal `H0`.
- Liouville accepts only `backend="numpy"` and `algorithm="rk4"`.
- Unsupported sparse/backend combinations must raise before conversion.
- The typed incoherent-ensemble route supports split operator by propagating
  each pure component independently and summing the resulting projectors with
  normalized weights. Split propagation of an explicit density matrix remains
  unsupported. The legacy factory rejection is transitional.
- A backend name must govern both dipole construction and time propagation for
  a simulation case to avoid cross-backend array mismatches.
- `PropagatorFactory` requires typed state path, algorithm, execution policy,
  and renormalization choice; it never selects from polarization or sparsity.
- Low-level pure-state propagation returns shape `(saved_times, dimension)`.
  This includes final-only output, whose shape is `(1, dimension)`, on both
  NumPy and CuPy paths. Higher-level final-only APIs may remove that leading
  saved-time axis exactly once.
- Dipole helper `_xp` raises when requested CuPy is unavailable; it never
  substitutes NumPy.

### Numba CSR RK4 reference anchor

`tests/physics/test_sparse_rk4_reference.py` fixes the NumPy sparse contract:

- matrix storage is selected explicitly with `sparse=True`;
- SciPy sparse input without that selection raises before propagation;
- canonical CSR preparation does not mutate input or discard nonzero values;
- fused CSR application implements `-1j*(H0-mu_x*Ex-mu_y*Ey)@psi`;
- a deterministic 64-state Hermitian problem agrees with dense RK4 over the
  full trajectory to absolute tolerance `3e-13`;
- sparse trajectory and final-only paths return identical final states;
- renormalizing a zero or non-finite state raises instead of returning
  partially uninitialized output.

The tolerance covers accumulated floating-point ordering differences between
dense fastmath and strict CSR row reductions. It is not an operator-element
cutoff; the propagation layer applies no approximate sparsification.

A skipped CuPy test does not establish correctness. Keep capability wording
conditional until tested in a CUDA CI job.

Current anchors:

- `tests/contracts/test_solver_contracts.py` checks the CuPy final-only dispatch
  shape without requiring a GPU.
- `tests/physics/test_two_level_reference.py` contains the real NumPy/CuPy
  final-state parity case and is marked `gpu`.

### Solver invariant Phase 0 reference anchor

`tests/physics/test_solver_invariants.py` is the P0.6 independent reference.
It fixes the following deterministic problems and tolerances:

- RK4 global order uses free evolution with energies `(0, 1.7) rad/fs`,
  total time `2 fs`, and steps `0.2`, `0.1`, and `0.05 fs`. Error ratios
  must lie between 14 and 18 around the analytic fourth-order value 16.
- RK4 norm drift uses one eigenstate at `4 rad/fs`, step `0.3 fs`, and 20
  steps. With `renorm=False`, the norm must equal the fourth-order stability
  polynomial magnitude raised to the twentieth power and must visibly differ
  from one. With `renorm=True`, every saved norm agrees with one to `2e-15`.
- One nonautonomous RK4 step uses `E_left=0.2`, `E_mid=0.7`,
  `E_right=0.1`, and `dt=0.01 fs`. A direct four-stage calculation must
  agree to `2e-15` and fixes both left/mid/right sampling and `H0-mu E`.
- RK4 trajectory and final-only paths must return identical final vectors.
- Split operator uses 50 steps of `0.02 fs` with a sinusoidal field. All
  norms agree with one to `2e-14`; trajectory and final-only results agree
  exactly. A non-diagonal `H0` must raise before propagation.
- The physical-time reference uses a field grid from `1.0` to `1.5 fs`
  with `0.05 fs` field intervals. Propagation times advance by `0.1 fs`.
  With stride two, the current regular output is `(1.0, 1.2, 1.4) fs` and
  omits the `1.5 fs` endpoint; final-only output reports `1.5 fs`.
- Liouville propagation uses 40 steps of `0.01 fs`. Trace and Hermiticity
  errors remain below `3e-15` without projection or normalization.
- The density positivity boundary is derived directly from
  `100*n*eps*||rho||_2`: half the reference scale is accepted and twice the
  reference scale is rejected.
- Invalid RK4 backends, CuPy Liouville, and factory split-operator mixed
  states raise explicit capability errors.

The low-level endpoint, mixed split-operator factory, and renormalization checks
remain characterization anchors while D-026 is applied at the typed boundary.

### Optimization time and backward-propagation contract

GRAPE and Krotov use canonical `TimeGrid` instances beginning at zero. Their
configuration specifies the electric-field sampling interval directly as
`field_dt_fs`; one state-propagation step is exactly
`2 * field_dt_fs`. The requested total span must be exactly divisible by that
propagation step. A solver or configuration layer must never round the number
of steps, extend the endpoint, or reinterpret `field_dt_fs` as a propagation
interval.

The repository Krotov configurations use `field_dt_fs = 0.05` fs. This is the
explicit form of the historical `dt_fs = 0.1` fs propagation step and produces
the same field arrays. Local optimization is a deliberate exception under
D-027: it retains its versioned `np.arange` storage, segment endpoints, and
`sample_stride`, with the existing numerical value 0.1 fs renamed only to
`field_dt_fs`.

The local optimizer requires `gain` together with `gain_units`. The supported
unit labels are `(V/m)^2 fs`, `(MV/m)^2 fs`, `(GV/m)^2 fs`, and
`(TV/m)^2 fs`; the one canonical internal representation is `(V/m)^2 fs`.
The gain must be finite and strictly positive. For example, the historical
`1e21` canonical value is entered as `1000 (GV/m)^2 fs`. Conversion occurs
before the unchanged update expressions `E_a = gain * S * response_a`.
Because the projected dipole has units `rad/fs/(V/m)`, the gain has
field-squared-time units, with radians treated as dimensionless.

The local optimizer's direct component limit is `field_max_v_per_m`, and its
zero-drive seed is `seed_amplitude_v_per_m`. Both are explicitly V/m. The
historical defaults remain exactly `1e12 V/m` and `1e3 V/m`; a generated seed
is still evaluated before the existing componentwise clipping. The former
unit-ambiguous keys are errors. This rename does not change the odd storage
grid, shared endpoint ownership, segment midpoint or slices, lookahead index,
field values, or RK4-consumed prefix. `c_abs_min`, `drive_abs_min`, and
`shape_floor` retain unresolved Class-D dimensions and must not be assigned
units without a separate user decision.

Local reports `field_fluence_proxy = (1/gain) * sum_i
S(t_i) * (E_1(t_i)^2 + E_2(t_i)^2) * field_dt`. This is a diagnostic proxy,
not a claim that the expression is the exact optimization objective. It also
reports the maximum and RMS of the stored vector amplitude
`sqrt(E_1^2 + E_2^2)`, the fraction of control segments in which either
component changed under the existing componentwise clip, and the separate
per-axis reference scales `gain * max(abs(mu'_a))`. These diagnostics never
alter a field value, segment, index, endpoint, or propagation call.

GRAPE and Krotov optimization always consume every propagated state internally.
`output_stride` applies only after the final objective has been evaluated and
only to the returned trajectory. The initial state and exact endpoint remain in
the returned trajectory even when the stride does not divide the propagation
step count.

Krotov costates use explicit `PropagationDirection.BACKWARD`. For dimensional
NumPy RK4 this reverses both field component arrays and applies the negative
propagation interval used by the historical implementation, while the public
`ElectricField` time grid remains strictly increasing. Backward CuPy,
split-operator, and nondimensional propagation are unsupported and must raise;
they never fall back to forward propagation.

Reference anchors are
`tests/physics/test_optimization_time_reference.py`,
`tests/contracts/test_optimization_time_grid_contracts.py`, and
`tests/contracts/test_optimization_solver_time_contracts.py`.

The stored four-level Krotov regression workload is defined by
`configs/reference_krotov_viblad_v3.yaml` and
`benchmarks/krotov-v0-v3-v0.3.{json,npz}`. It records the current numerical
ability to transfer V=0 population to V=3 and verifies the optimized field with
a separate final forward propagation. It is a characterization reference, not
an independent derivation or validation of the Krotov update equation.

## 10. Spectroscopy evaluation contract

Experimental spectroscopy inputs are part of the physical problem. Temperature
`T`, pressure, optical length, coherence time `T2`, and per-molecule mass `m`
are required positive finite values with their own required unit labels. The
public pairs are `temperature/temperature_units`,
`pressure/pressure_units`, `optical_length/optical_length_units`,
`coherence_time/coherence_time_units`, and
`molecular_mass/molecular_mass_units`. The currently implemented boundary
accepts exactly K, Pa, m, ps, and kg per molecule; unsupported labels raise
instead of being inferred or converted by an undocumented formula.

`ExperimentalConditions` is frozen and exposes the canonical values consumed
by numerical code as `temperature_k`, `pressure_pa`, `optical_length_m`,
`coherence_time_ps`, and `molecular_mass_kg`. Number density remains
`pressure_pa / (k_B * temperature_k)`, and the coherence decay remains
`1 / (coherence_time_ps * 1e-12)`. Spectroscopy uses the constants from
`core.units.constants`; local rounded copies are forbidden.

All public spectral grids require `wavenumber_units="cm^-1"`. Device
broadening additionally requires `device_resolution/device_resolution_units`
as a pair, with `device_resolution_units="cm^-1"`. Radiation and PFID routes
have the same explicit grid-unit boundary. Internal exact/approximate routing
and every response formula consume the same canonical cm^-1 arrays as before.

For ordered Cartesian components `axes`, interaction and detection use

~~~text
mu_int = sum_a e_int[a] mu_a
mu_det = sum_a conj(e_det[a]) mu_a
~~~

Thus identical excitation and detection polarization gives a Jones bra-ket
contraction and is invariant under a global polarization phase. Every selected
axis contributes, and each finite nonzero Jones vector must have exactly
`len(axes)` components.
The projected per-molecule response is converted through
`chi = number_density * response / epsilon_0`. No unconditional `1/3` factor is
applied after polarization projection. Any isotropic orientational average must
already be represented by the density matrix and lab-frame dipole operators,
or be selected later through an explicit approximation policy.

The pre-probe pathway is always explicit:

~~~text
phase_matching = "pump_probe":
    P_ij = 1 when V_i == V_j, otherwise 0
    rho_selected = P * rho_pre_probe

phase_matching = "unfiltered":
    P_ij = 1 for every i, j
    rho_selected = rho_pre_probe
~~~

For the present pump-probe workflow, V labels the net vibrational absorption
and emission order and is therefore the selected phase-matching proxy. This
retains the complete equal-V blocks: rotational and M coherences with
`V_i == V_j` survive. It removes cross-V density entries before the probe
commutator. `pump_probe` requires a valid `basis.V_array`; absence or shape
mismatch raises, and no unfiltered fallback is permitted.

The reported discarded-density fraction uses the Frobenius norm:

~~~text
f_discarded = ||(1 - P) * rho_pre_probe||_F / ||rho_pre_probe||_F
~~~

It is defined as zero for an all-zero density matrix. This fraction describes
a physical pathway selection and is independent of the later commutator
threshold used only by `approximate_sparse`.

The selection applies only to absorption through `calculate`, whose input is
the density immediately before the probe interaction. `calculate_radiation_spectrum`
and `calculate_pfid_spectrum` instead consume a post-probe density directly and
must retain cross-V optical coherence; applying `P` there would erase the
radiating signal. This V-based contract is specific to the current pump-probe
workflow and is not a general wave-vector bookkeeping implementation.

For an angular-frequency grid, transition-specific Doppler broadening uses

~~~text
sigma_omega = |omega_0| sqrt(k_B T / (m c^2))
sigma_pixels = sigma_omega / delta_omega
~~~

Only `matrix` and `loop` accept Doppler broadening, and both broaden each
transition susceptibility before summation. Routes without the same
transition-specific kernel raise. Broadening requires a strictly monotonic
uniform grid and is applied at every positive resolved width; no fixed absolute
threshold decides whether the physics is skipped.

The response calculation policy is:

- `matrix`, `loop`, `2d`, and `chunked` are exact routes without Doppler. Their
  detection support removes only relative machine-roundoff noise below
  `eps * max(abs(mu_det))`, never an absolute physical-dipole threshold.
- `approximate_sparse` is a distinct opt-in route. It requires a relative
  threshold with `0 < threshold <= 1`, scales it by the largest relevant
  commutator magnitude, and reports the discarded commutator L2 fraction.
- `auto` is also opt-in. It requires a positive memory budget and chunk size,
  and reports whether `2d` or `chunked` actually ran.
- `2d`, `chunked`, `approximate_sparse`, and `auto` reject Doppler broadening
  until they share the transition-specific kernel; method selection may not
  change the physical line shape.
- A requested device function must be recognized and applied. Its resolution
  is required, positive, paired with its required cm^-1 label, and expressed
  on the explicitly labeled cm^-1 wavenumber grid.

`SpectroscopyCalculationReport` is the observable record of requested and
executed method, estimated allocation, explicit memory/threshold controls,
discarded commutator fraction, explicit phase-matching mode, discarded
pre-probe density fraction, and device-function application. Exact-route
agreement, pathway selection, and contract failures are anchored by
`tests/physics/test_spectroscopy_reference.py`. Independent experimental
spectra or sum rules remain required before decomposing the full spectroscopy
module.

## 11. Input validation principles

Parameters that define the physical problem must be required rather than
filled with extreme or arbitrary defaults. In particular:

- `t_start`, `t_end`, and field sampling `dt`;
- model-defining maximum quantum numbers;
- required model frequency or energy gap;
- dipole scale;
- initial state specification;
- pulse center, carrier frequency, amplitude, and duration as required by the
  chosen field construction.

Defaults are acceptable only for representation choices with a safe,
documented meaning, such as `backend="numpy"` or
`potential_type="harmonic"`.

Unknown keys and unsupported combinations must fail with the parameter name and
reason. No physical input may be ignored.

Decision D-041 strengthens the target normal-simulation boundary. Model kind,
initial states, LinMol representation, generated-field kind, and every
field-defining parameter are explicit. Scalar models receive a scalar field;
only M-resolved LinMol receives Cartesian components and polarization.

Python callers may inject sampled external fields. Their samples must match one
canonical `TimeGrid` exactly: finite uniform increasing time, odd sample count,
both endpoints, `2 * propagation_steps + 1` values, and
`propagation_dt = 2 * field_dt`. Field components must be finite and have the
exact required one-dimensional shape. Validation never resamples, trims, pads,
rounds, normalizes, or repairs the field.

The implemented sampled-field boundary accepts real values only, in V/m.
`ScalarField` stores one waveform. `CartesianField` stores two ordered components
and may additionally carry an explicitly supplied scalar/Jones decomposition.
That decomposition is provenance for the existing approximate
`helicity_projected` split interaction; it is never inferred from arbitrary
Cartesian samples. A missing decomposition therefore raises if that
approximation is requested, while exact Cartesian RK4 and split propagation
need only the two real components.

The legacy low-level `ElectricField.add_arbitrary_Efield` boundary requires a
`field_units` label on every supplied array and converts exactly once to its
internal V/m storage before the unchanged elementwise addition. Accepted
labels are direct electric-field amplitude units only. Cycle-averaged intensity
units are rejected because a signed, phase-bearing field waveform cannot be
reconstructed from intensity samples. Internal GRAPE, Krotov, and local
optimizer arrays are labeled `V/m`; this labeling does not alter their values,
time grids, endpoint slices, segment indices, or RK4 consumption.

Direct low-level `ElectricField` and `ZeroField` construction requires
`time_units`; the supplied increasing uniform time grid is converted exactly
once to internal fs. The constructor has no `field_units` argument because it
receives no field-valued input: the initially zero field and all later stored
samples use V/m. `ElectricField.from_time_grid` consumes the canonical fs grid
without another caller unit choice. Field-scale output requires an explicit
direct amplitude unit. The legacy-named `get_time_SI()` returns canonical fs,
not seconds; `get_time_in_units()` is the explicit conversion boundary.

Low-level `add_dispersed_Efield` calls require explicit units for duration,
center time, carrier frequency, and amplitude. Inputs are converted once to
fs, fs, cycles/fs, and V/m before the unchanged envelope, carrier, and
dispersion formulas run. Amplitude accepts direct electric-field units only;
cycle-averaged intensity cannot label an already defined signed amplitude.
GDD and TOD are independent optional value/unit pairs. Supplying either member
without the other is an error; omitting both members of a pair applies exactly
zero fs^2 or fs^3. No pulse sample, polarization decision, FFT operation, or
optimizer time grid is repaired or inferred by this boundary.

Generated pulses require the serialization-safe `envelope_kind` and
`modulation_kind` discriminators. Supported generated envelopes are `gaussian`,
`gaussian_fwhm`, `lorentzian`, and `lorentzian_fwhm`; `duration` retains the
unchanged width meaning of the selected envelope function. `t_center` is required.
Voigt and arbitrary callables are not guessed into the one-width schema and use
external sampled-field injection instead.

`modulation_kind` is `none` or `sinusoidal`. Sinusoidal modulation requires
a finite depth, a physical delay with an explicit time unit, and `phase` or
`amplitude` mode; its additive phase defaults safely to zero. With FFT
frequencies `f` and `f0` in cycles/fs, its argument is
`2*pi*delay_fs*(f-f0)+phase`. Phase mode applies
`exp(-i*depth*sin(argument))`; amplitude mode applies
`1+depth*sin(argument)` and requires `0 <= depth <= 1`. Zero depth is an
exact identity. No clip, absolute-value repair, or implicit normalization is
allowed.

Physical dispersion uses
`GDD*delta_omega**2/2 + TOD*delta_omega**3/6`, where
`delta_omega=2*pi*(f-f0)`, with the existing `exp(-i*phase)` convention.
The removed `envelope_func`, mixed-case `Sinusoidal_modulation`, and four
legacy `*_sin_mod` keys raise migration errors.

Generated pulses are evaluated by the characterized legacy generator and then
copied exactly into the typed field. Tests require bitwise-equal populations for
generated and externally injected TwoLevel, M-resolved LinMol, and M-averaged
LinMol cases. This boundary conversion changes neither field samples nor the
Hamiltonian evaluated at them.

Krotov initial fields have a separate explicit source discriminator. A
`generated` seed is the existing Gaussian-FWHM pulse and requires duration,
center, carrier frequency, and direct amplitude value/unit pairs plus a finite
nonzero two-component polarization. Optional GDD/TOD pairs are exact zero when
both members are omitted. A `sampled` seed requires a real finite two-column
array and a direct amplitude unit. It is converted once to V/m and must match
the canonical odd Krotov `TimeGrid` exactly. The boundary never chooses between
the two sources implicitly and never resamples, normalizes, repairs, or derives
a signed field from intensity. Krotov update indices and objective logic consume
the same field samples as before D-050.

Optimization model construction uses the same frozen physical parameter
schemas and model-owned basis/Hamiltonian/dipole builders as normal simulation.
Optimization state entries remain exact quantum-number tuples for `initial`
and `target`; they are not normal-runner basis indices and are never repaired by
adding or removing M. LinMol optimization currently requires the full
M-resolved basis. The separate M-block incoherent average is unsupported in
optimization and raises instead of being approximated by a no-M basis.

The optimizer retains its historical NumPy CSR construction and rad/fs
Hamiltonian projection. Krotov and local optimization retain their ordered
two-component control-axis projection, including the zero second component for
a VibLadder `zx` choice. This is a workflow adapter, not a change to the normal
scalar VibLadder coupling contract. No optimizer objective, gradient, time
grid, factor-of-two field index, or endpoint rule is changed by sharing model
construction. SymTop optimization remains unsupported pending an independent
objective/control reference.

Decision D-056 requires configured optimization to use an explicit ordered
two-axis adapter for every algorithm. D-057 requires those labels to be
distinct. Krotov and local accept exactly two lowercase Cartesian labels and
preserve their existing two-column projection; GRAPE accepts only `xy`, the
only route it implements. Invalid, missing, or unsupported axes raise and are
never replaced by `xy`. For scalar VibLadder and TwoLevel models these labels
only preserve the historical two-column optimizer representation and do not
create physical polarization dependence. The closed configuration boundary
does not alter any optimizer field sample, objective, update equation, time
grid, or endpoint rule.

Local evaluation is selected only by exact `target` or `weights` labels;
weight construction is selected only by `by_v`, `by_v_power`, or `custom`,
with reversal carried by its separate boolean. Requested lookahead requires a
finite real eigenvalue for every basis state and never silently disables itself.
These validation rules do not change the accepted Local update expression or
its frozen time/index layout.

For the monotonic Krotov spectral route, `alpha` is finite and nonnegative.
Each frequency component therefore uses the exact denominator `1 + alpha`,
which is at least one. No clipping or denominator floor is applied. Invalid
bands, units, modes, weights, scale, shape, or alpha values raise before they
can alter an update.

Time-grid consistency does not prove propagation accuracy. Accuracy assessment
requires the complete generator and observable and is therefore an explicit
convergence calculation with a caller-selected tolerance. It reports the
comparison and never changes the requested field or propagation interval.
The standard calculation compares equal-shaped coarse- and fine-grid observable
values using `max(abs(observable_coarse - observable_fine))`. The caller must
name and provide the observable, both grids, and a finite nonnegative tolerance
in the observable's units. There is no library-selected relative tolerance,
hidden safety factor, automatic refinement, or pass-triggered rerun.

## 12. Required physics test matrix

Every major solver or model migration must cover the applicable rows:

| Area | Required checks |
|---|---|
| TwoLevel | analytic free evolution, driven two-level reference, scalar-polarization independence |
| VibLadder | harmonic energies, anharmonic energies, Morse bound, transition rules |
| LinMol | state-index round trip, degeneracy/state ordering, rotational-vibrational energies, polarization response |
| Dipole | shape, Hermiticity, selection rules, known elements, dense/sparse agreement |
| RK4 | order/convergence, norm drift, left-mid-right field sampling, final/trajectory agreement |
| Split operator | norm conservation, diagonal-H0 rejection, RK4 agreement at small step |
| Liouville | trace, Hermiticity, PSD input validation, pure-state agreement |
| Mixed state | weight normalization, no coherent cross terms, component time agreement |
| Units | round trips and canonical-boundary equivalence |
| Nondimensional | physical/nondimensional observable and time equivalence |
| Backend | NumPy dense/sparse and real CuPy parity where supported |
| Spectroscopy | exact-route agreement, explicit approximation report, grid-derived broadening, device-function application |

Tolerance values must be justified by algorithm order, machine precision, and
problem scale. Do not use a loose constant solely to make a test pass.

## 13. Open physics/API decisions

The first four Phase 2 propagation questions were resolved by D-026. These
items still require user input before behavior changes:

1. Independent objective/gradient references and acceptable tolerances for
   optimization, plus spectroscopy references beyond the existing exact-route
   tests.
