# Physics and numerical contracts

Last verified against source and tests: 2026-07-31
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

Current return-time behavior:

- Full trajectory starts at `t_start`.
- With `sample_stride = s`, adjacent returned states are separated by
  `s * propagation_dt`.
- Final-state-only propagation returns a one-element time array containing
  `t_end`.
- Dimensional and nondimensional paths must return the same physical time axis
  in femtoseconds.

Current stride behavior records the initial state and states at steps divisible
by the stride. If `n_steps` is not divisible by the stride, the endpoint is not
included in a trajectory. Whether to always append the endpoint is still an
open design decision; do not change it without user approval.

Primary implementation anchors:

- `simulation/timegrid.py`
- `core/propagation/utils.py`
- `core/propagation/schrodinger.py`
- `core/propagation/liouville.py`

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

### 4.3 Explicit density matrix through MixedStatePropagator

An explicit square matrix passed to `MixedStatePropagator` is validated and
then normalized by its positive real trace before Liouville propagation.

Direct `LiouvillePropagator` input is validated but is not currently normalized
to trace one. Requiring unit trace in the direct API is an open decision.

### 4.4 Solver renormalization

Wavefunction renormalization is opt-in. The default is `renorm=False`.
Refactoring must not silently enable it, because it can hide integration error.
If enabled, its threshold and behavior must remain visible in solver options.

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
- `PropagatorFactory` currently rejects split operator for `state_type="mixed"`
  even though direct ensemble propagation can delegate to it. This discrepancy
  is an open API decision.
- A backend name must govern both dipole construction and time propagation for
  a simulation case to avoid cross-backend array mismatches.
- Low-level pure-state propagation returns shape `(saved_times, dimension)`.
  This includes final-only output, whose shape is `(1, dimension)`, on both
  NumPy and CuPy paths. Higher-level final-only APIs may remove that leading
  saved-time axis exactly once.
- Existing dipole helper `_xp` can fall back to NumPy when CuPy is unavailable.
  This silent fallback is a known defect and must be replaced by an explicit
  availability error.

A skipped CuPy test does not establish correctness. Keep capability wording
conditional until tested in a CUDA CI job.

Current anchors:

- `tests/contracts/test_solver_contracts.py` checks the CuPy final-only dispatch
  shape without requiring a GPU.
- `tests/physics/test_two_level_reference.py` contains the real NumPy/CuPy
  final-state parity case and is marked `gpu`.

## 10. Input validation principles

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

## 11. Required physics test matrix

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

Tolerance values must be justified by algorithm order, machine precision, and
problem scale. Do not use a loose constant solely to make a test pass.

## 12. Open physics/API decisions

These items require user input before behavior changes:

1. Whether trajectories must always append the final state when stride does
   not divide the number of propagation steps.
2. Whether direct Liouville input must have trace one, should be normalized, or
   may remain a positive scaled density operator.
3. Whether `PropagatorFactory` should support split operator for incoherent
   pure-state ensembles.
4. Whether renormalization should remain an exposed solver option or be
   restricted to diagnostics.
5. The intended production status and validated physics scope of SymTop.
6. Reference problems and acceptable tolerances for optimization and
   spectroscopy, which currently have insufficient automated coverage.
