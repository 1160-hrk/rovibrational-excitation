# Phase 7 optimization references

Last verified: 2026-09-27
Current checkpoint: P7.3-c, independent Local-control reference

## Purpose

This document records the slow, transparent scientific references required by
D-072 before optimization code is decomposed. Reference calculations live only
in tests and are never called by production code. A production formula may be
changed only after a disagreement is shown to the user and explicitly resolved.

## P7.3-a — GRAPE terminal-population gradient

### Discrepancy found

The former `optimization.grape` update was described in source as a
time-local heuristic. At propagated state `psi_i`, it used only

~~~text
imag(<target | -mu_a | psi_i>)
~~~

and copied that value to two neighboring field samples. It did not propagate a
terminal costate and did not differentiate the RK4 stages, the propagation
step, shared RK4 endpoints, or per-step normalization. It was therefore not
the gradient of the reported terminal target population.

On the fixed TwoLevel diagnostic used during P7.3-a, the second-iteration
update inferred from the old solver was compared with a central finite
difference of the actual objective:

| Quantity | Observed value |
|---|---:|
| Old update-vector norm | approximately `1.14e-9` |
| Central-difference gradient norm | approximately `6.34e-14` |
| Relative error | approximately `1.79e4` |
| Direction cosine | approximately `0.894` |

The finite-difference norm was stable for perturbations from `10` through
`1e5 V/m`; this was not a finite-difference step artifact. The user approved
replacement of the heuristic on 2026-09-24.

### Objective

Production GRAPE now minimizes the explicitly declared discrete objective

~~~text
J(E) = 1 - |<target | psi_N(E)>|^2
       + (lambda_a / 2) sum[q,a] E[q,a]^2.
~~~

The constant one has no effect on the gradient. The `lambda_a` term preserves
the former code's unweighted discrete L2 derivative `lambda_a * E`; it is not
reinterpreted as a time-integrated physical fluence. The dimensions and useful
scale of `lambda_a` and `learning_rate` remain Class-D questions and are not
inferred by this checkpoint.

### Forward discrete map

For one propagation step with left, midpoint, and right field samples, define

~~~text
A_q(E_q) = -i (H0 - E[q,0] mu_0 - E[q,1] mu_1)

k1 = A_left psi
k2 = A_mid   (psi + dt k1 / 2)
k3 = A_mid   (psi + dt k2 / 2)
k4 = A_right (psi + dt k3)

z       = psi + dt (k1 + 2 k2 + 2 k3 + k4) / 6
psi_new = z / ||z||.
~~~

This is the actual dense NumPy RK4 graph used by the optimizer: `dt` is
`2 * field_dt_fs`, midpoint samples are used by both `k2` and `k3`, and the
right sample of one step is the left sample of the next. The normalization is
performed after every propagation step, matching `renorm=True`.

### Exact discrete reverse pass

Production `optimization.grape_rk4.evaluate_discrete_rk4` stores the forward
stage state vectors and applies reverse-mode differentiation to that exact
graph. It reconstructs stage operators in the reverse pass instead of storing
three `dimension x dimension` matrices per time step, so the persistent tape
uses `O(n_steps * dimension)` rather than `O(n_steps * dimension^2)` memory.
For real objectives and complex state vectors, adjoints are defined by

~~~text
dJ = 2 Re(<bar_psi | dpsi>).
~~~

At the terminal state, with `c = <target | psi_N>`, the population part starts
from

~~~text
bar_psi_N = -c |target>.
~~~

For `psi_new = z / r`, `r = ||z||`, normalization is reversed as

~~~text
bar_z = bar_psi_new / r
        - z Re(<bar_psi_new | z>) / r^3.
~~~

Each matrix-vector stage is then reversed in the opposite order. Because

~~~text
dA_q / dE[q,a] = i mu_a,
~~~

one stage contributes

~~~text
dJ/dE[q,a] += 2 Re(<bar_k | i mu_a | stage_state>).
~~~

Both midpoint-stage contributions and contributions from adjacent steps are
accumulated into the same field sample. Finally, `lambda_a * E` is added.

This is a discrete adjoint of the implemented RK4 calculation, not a
continuous-time approximation and not an exact-matrix-exponential GRAPE
formula. Custom `propagator_func` values are rejected because no corresponding
discrete derivative exists at this boundary.

### Independent reference and tolerance

`tests/physics/test_grape_gradient_reference.py` contains an independent,
deliberately slow Python RK4 objective. It does not call the production
gradient or production propagator. Every field component is perturbed in both
directions and differentiated by central finite differences.

For perturbations `1e8`, `1e7`, and `1e6 V/m`, the relative error decreases
toward an observed `5.8e-9` plateau. The regression bound is `1e-7`: safely
above the observed plateau and substantially below D-072's provisional
`1e-5` target. The same test compares every normalized trajectory state and
the objective, while a second test verifies that the runner applies exactly
one computed gradient step and increases fidelity on the fixed TwoLevel case.

### Initial-field contract

For a diagonal `H0`, orthogonal initial and target basis states, and zero
field, terminal population is quadratic at the origin. Its first derivative is
therefore zero. The old heuristic moved away from zero only because it was not
the objective gradient.

GRAPE consequently requires an explicit initial field. It shares Krotov's
strict discriminator and value/unit boundary:

- `initial_field_kind: generated` requires duration, center, carrier,
  direct-amplitude value/unit pairs, and a finite nonzero two-component
  polarization; optional GDD/TOD remain complete pairs;
- `initial_field_kind: sampled` requires a real finite two-column array, a
  direct electric-field amplitude unit, and exact canonical field-grid length;
- neither route resamples, normalizes, repairs, or silently falls back to a
  zero field.

The shared implementation retains its historical internal class/module names;
renaming those symbols is structural follow-up work, not part of this physics
change.

### Deliberately unchanged behavior and remaining work

- GRAPE still supports only the ordered `xy` control adapter.
- Gradient descent still uses `E <- E - learning_rate * gradient`.
- `target_fidelity` is still checked against terminal fidelity, not the
  penalized objective.
- The existing `convergence_tol` branch observes a small fidelity change but
  does not stop (`pass`). Fixing that changes iteration count and final fields,
  so P7.3-a records but does not alter it without a separate decision.
- Krotov, Local, and spectral-constraint formulae are unchanged. Their D-072
  independent references remain P7.3-b through P7.3-d work.

## P7.3-b — standard Krotov one-iteration construction

### Discrepancy and resolution

The independent direct construction found that the former solver was not the
standard sequential first-order Krotov update. It normalized the backward
costate without restoring `|<target|psi_T>|`, evaluated every update from the
old forward trajectory, and multiplied the dipole matrix element by two. On
the fixed diagnostic, preserving the costate scale reduced the update norm by
roughly the terminal overlap factor; the former rapid improvement was therefore
not evidence for the formula. The user approved the D-106 split.

The former calculation moved unchanged to `legacy_batch_overlap`. Its stored
reference and spectral configuration also select that name. Standard `krotov`
uses a separate interval-control engine; no configuration silently changes
meaning.

### Standard construction

For interval `n`, `E[n]` is constant on `[t_n,t_(n+1))` and is represented at
the midpoint. An old forward trajectory and an overlap-scaled backward costate
are constructed under the old control. The new trajectory is then built
sequentially:

~~~text
chi_N = <target|psi_N> target
Delta E[n,a] = S(t_(n+1/2))/lambda_a
               Im(<chi_old[n]|-mu_a|psi_new[n]>)
E_new[n,a] = E_old[n,a] + Delta E[n,a]
psi_new[n+1] = U(E_new[n], control_dt) psi_new[n].
~~~

The minus sign is `dH/dE_a=-mu_a` for the repository-wide
`H=H0-sum_a mu_a E_a` convention. There is no extra factor two. Costates are
not normalized. Production uses one constant-H dense NumPy RK4 step for `U`
and does not renormalize the state; accuracy remains an explicit grid
convergence question.

### Independent oracle and tolerance

`tests/physics/test_krotov_iteration_reference.py` directly expands the RK4
stages and independently constructs the old trajectory, terminal costate,
negative-time backward trajectory, every sequential update, and the updated
trajectory. It does not call the production propagator. All four arrays agree
with production to absolute tolerance `2e-15`; terminal and final fidelity
checks use ulp-scale scalar tolerances. The costate terminal norm is explicitly
fixed to the overlap magnitude and explicitly differs from one.

### Units, input, and unsupported combinations

`lambda_a` has the required paired unit `lambda_a_units` and canonical value in
`1 / ((V/m)^2 fs)`. Equivalent MV/m, GV/m, and TV/m squared labels convert
once. Standard time uses `control_dt_fs`; state endpoints have length `N+1` and
midpoint controls have length `N`. Generated and sampled sources are selected
by `initial_control_kind`; sampled controls require exact shape `(N,2)`. The
old `field_dt_fs`, `initial_field_*`, custom propagator, spectral constraint,
and plotting schemas raise rather than being reinterpreted.

### End-to-end transfer and accuracy evidence

`tests/integration/test_standard_krotov_transfer.py` fixes two deterministic
problems:

- TwoLevel, 100 fs and 0.2 fs controls: the generated seed begins below 0.05
  target population and reaches above 0.999 with a monotone observed fidelity
  history.
- Harmonic VibLadder with `V_max=4`, initial `V=0`, target `V=3`, 500 fs and
  0.1 fs controls: `V=3` exceeds 0.985, every other level remains below 0.01,
  and the coarse trajectory norm error stays below `1.1e-3`. Repropagating the
  final control with each interval split in half puts the norm error below
  `4e-5` and changes the target population by less than `1.5e-3`.

On the current CPython 3.12.12 aarch64 environment, pytest reports 0.22 s for
the TwoLevel case and 5.79 s for the VibLadder case including its half-step
repropagation. These wall times are observational and are not enforced as
performance thresholds.

These are deterministic regression and grid-convergence checks, not universal
recommended pulse or penalty values.

## P7.3-c — direct Local update on the frozen legacy grid

### Scope and unchanged calculation

The reference covers both supported evaluation modes without changing
production code. For a segment beginning from `psi`, the optional lookahead
state is constructed exactly as

~~~text
psi_ref[j] = psi[j] exp(-i epsilon[j] tau),
tau = lookahead_fraction * (tlist[end - 1] - tlist[start]).
~~~

In `weights` mode, with diagonal evaluation operator `A`, the two direct
responses and fields are

~~~text
r_a = Im(<psi_ref | A (-mu_a) | psi_ref>)
E_a = gain S r_a.
~~~

In `target` mode they are

~~~text
c   = <target | psi_ref>
d_a = <target | (-mu_a) | psi_ref>
r_a = Im(conj(c) d_a)
E_a = gain S r_a.
~~~

These signs follow `H=H0-sum_a mu_a E_a`. No extra factor is introduced; the
existing production expressions above are retained. This checkpoint does not
assign dimensions or recommended values to `c_abs_min`, `drive_abs_min`, or
`shape_floor`.

The test independently reproduces the D-027 grid rather than calling the
production grid builder: `tlist=np.arange(0,1.0,0.1)`, segments `(0,4)` and
`(4,8)`, writes to `start+1:end+1`, propagates `start:end+1`, and performs
the final propagation only on the odd prefix `0:9`. Its direct Python RK4
expands all four stages under `H0-mu_x E_x-mu_y E_y` and normalizes after each
0.2 fs propagation step. It calls neither the production propagator nor a
production Local helper.

### Fixed diagnostic and observed agreement

The deterministic TwoLevel diagnostic uses a 0.47 rad/fs gap, `2.1e-29 C·m`
Cartesian dipole, one shaped 2e8 V/m seed segment, 20 `(GV/m)^2 fs` gain,
sine-squared shape, and a 0.5 lookahead fraction. The seed makes the first
segment nonzero; the second segment is therefore a genuine response-derived
update rather than another seed. No component clips.

Both `weights` and `target` production fields agree with the direct
reference. The largest absolute field differences are respectively
`3.73e-9 V/m` and `1.12e-8 V/m` on approximately `8.26e7 V/m` fields;
relative array-norm errors are `3.12e-17` and `1.22e-16`. The largest
trajectory difference is `2.23e-16` and the relative trajectory-norm errors
remain below `1.5e-16`. Regression comparisons use `rtol=2e-15` with
ulp-scale absolute trajectory tolerance.

The independent reference therefore finds no discrepancy requiring a physics
change. Seed trigger/sign, sine-squared sampling and floor, lookahead index,
componentwise clipping order, shared endpoint ownership, per-segment
propagation, final odd prefix, and stored tail remain exactly as previously
characterized. The authoritative test is
`tests/physics/test_local_update_reference.py`.

## Reference status

| Reference | Status |
|---|---|
| GRAPE central finite-difference gradient with step convergence | Complete — P7.3-a |
| Direct one-iteration Krotov construction | Complete — P7.3-b/D-106 |
| Direct Local update on frozen legacy grid | Complete — P7.3-c |
| Direct DFT/convolution spectral constraints | Pending — P7.3-d |
