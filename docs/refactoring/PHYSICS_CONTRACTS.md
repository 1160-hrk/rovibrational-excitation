# Physics and numerical contracts

Last verified against source and tests: 2026-08-11
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

Primary implementation anchors:

- `simulation/timegrid.py`
- `core/propagation/utils.py`
- `core/propagation/schrodinger.py`
- `core/propagation/liouville.py`

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

- core/electric_field/core.py: ZeroField;
- core/nondimensional/converter.py: validation, centering, and scale derivation;
- core/nondimensional/scales.py: values and provenance;
- core/propagation/schrodinger.py: global-phase restoration;
- tests/contracts/test_strict_nondimensional_contracts.py: reference contracts.
### 3.2 Explicit physical inputs

Missing and zero are distinct. A physical zero is accepted only when supplied
as `0.0`; no constructor or runner may invent a model constant or pulse width.

Required simulation fields are model-specific:

| Model | Required physical/model fields |
|---|---|
| LinMol | `V_max`, `J_max`, `omega_rad_phz`, `delta_omega_rad_phz`, `B_rad_phz`, `alpha_rad_phz`, `mu0_Cm`, `potential_type` |
| VibLadder | `V_max`, `omega_rad_phz`, `delta_omega_rad_phz`, `mu0_Cm`, `potential_type` |
| TwoLevel | `energy_gap`, `energy_gap_units`, `mu0_Cm` |

Every simulation case also states `duration` explicitly. The removed
`pulse_duration` name is rejected. Direct basis construction requires
LinMol `omega`, `B`, `alpha`, and `delta_omega`; VibLadder `omega`
and `delta_omega`; TwoLevel `energy_gap`; or SymTop `omega`, `B`, `C`,
`alpha`, and `delta_omega`. Direct dipole construction requires `mu0`;
LinMol, VibLadder, and SymTop dipoles also require `potential_type`. TwoLevel
does not accept this inapplicable option. Krotov optimization requires a
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

### 4.2 Incoherent ensemble

An incoherent mixture uses `MixedStatePropagator` with an iterable of state
vectors.

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

### 4.3 Explicit density matrix

The typed `DensityState` boundary requires trace one within the scale-aware
tolerance in Section 5. It never normalizes, clips, symmetrizes, or otherwise
repairs caller input.

During migration, an explicit square matrix passed through the legacy
`MixedStatePropagator` adapter is still normalized by its positive real trace,
while direct legacy `LiouvillePropagator` input is validated without trace
normalization. These transitional behaviors are not the final typed contract.

### 4.4 Solver renormalization

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

`MixedStatePropagator` performs its separately documented positive trace
normalization only after validation.

Primary implementation:
`core/propagation/algorithms/validation.py`.

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
| SymTop | Not integrated into the simulation model factory | Builder-specific | Must be characterized before migration |

For TwoLevel and VibLadder, `x` and `z` are current storage axes, not physical
polarization degrees of freedom. The target architecture should expose scalar
coupling directly so callers do not need a dummy polarization vector.

A supplied polarization may still be normalized and structurally validated by
the input layer. It must not change scalar-model excitation results.

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
semantics it uses in its name, type, or required options. No generic list input
may silently switch meaning based only on array shape except the currently
documented explicit square density-matrix dispatch in `MixedStatePropagator`.

### 8.1 LinMol fixed-linear M average

Decision D-017 defines two distinct LinMol workflows:

- `use_M=True`: explicit `|v,J,M>` basis, Cartesian x/y/z coupling, and
  physically direction-dependent polarization response;
- `use_M=False`: reduced `|v,J>` output with an incoherent average over
  separately propagated fixed-M blocks.

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
- Low-level pure-state propagation returns shape `(saved_times, dimension)`.
  This includes final-only output, whose shape is `(1, dimension)`, on both
  NumPy and CuPy paths. Higher-level final-only APIs may remove that leading
  saved-time axis exactly once.
- Existing dipole helper `_xp` can fall back to NumPy when CuPy is unavailable.
  This silent fallback is a known defect and must be replaced by an explicit
  availability error.

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

## 10. Spectroscopy evaluation contract

Experimental spectroscopy inputs are part of the physical problem. Temperature
`T`, pressure, optical length, dephasing time `T2`, and molecular mass `m` are
required positive finite values. Spectroscopy uses the constants from
`core.units.constants`; local rounded copies are forbidden.

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
  is required, positive, and expressed on the supplied wavenumber grid.

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

1. The intended production status and validated physics scope of SymTop.
2. Reference problems and acceptable tolerances for optimization and
   spectroscopy, which currently have insufficient automated coverage.
