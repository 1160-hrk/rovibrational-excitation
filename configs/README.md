# Current optimization configurations

Only the three YAML files in this directory are supported current-schema
optimization documents. Historical v0.2 documents live under
`examples/archives/v0_2_optimization_configs/` and must not be used as input.

## Included runs

- `reference_legacy_batch_overlap_viblad_v3.yaml`: stored V=0 to V=3
  `legacy_batch_overlap` reference.
- `example_legacy_batch_overlap_spectral_viblad_v3.yaml`: the same four-level physical model
  with an explicit legacy monotonic spectral constraint.
- `example_local_viblad_v3.yaml`: the frozen legacy local optimizer on the same
  explicit physical model.

Every document supplies a dipole value and unit. The value `0.3 D` belongs to
these examples; it is not a library default and is never inserted into another
configuration.

## Execution

~~~bash
rve-optimize --config configs/example_legacy_batch_overlap_spectral_viblad_v3.yaml
~~~

`output.dir` is required and is interpreted relative to the current working
directory when it is relative. `--out PATH` overrides it. `plot.enabled` is
required; `--no-plot` explicitly overrides a true value. A requested top-level
plot failure raises instead of converting the calculation to a reported
success.

CLI overrides use a complete dotted path, for example:

~~~bash
rve-optimize \
  --config configs/example_legacy_batch_overlap_spectral_viblad_v3.yaml \
  --override algorithms.legacy_batch_overlap.max_iter=1 \
  --out /tmp/rve-optimization-smoke \
  --no-plot
~~~

Unknown root, section, algorithm, and spectral-constraint keys are errors.

## Control axes and time samples

`control_axes` is required and ordered. For example, `zx` means that field
column 0 couples through `mu_z` and column 1 through `mu_x`; a generated legacy batch `initial_polarization: [1.0, 0.0]` therefore
drives only the z column. The local, standard Krotov, and legacy batch
optimizers accept two distinct lowercase labels from `x`, `y`, and `z`; duplicate pairs such as `xx` are errors. GRAPE currently accepts
only `xy` because no other route is implemented.

VibLadder and TwoLevel are physically scalar models. Their two columns and
axis labels are the preserved optimizer adapter, not a physical polarization
dependence. For the shown VibLadder `zx` setup, the x coupling is zero.

For GRAPE and `legacy_batch_overlap`, `field_dt_fs` is the field sampling
interval and one RK4 propagation step is `2 * field_dt_fs`. The exact field
length is `2 * n_propagation_steps + 1`. Standard `krotov` instead requires
`control_dt_fs`: it stores `N+1` state endpoints and exactly `N`
piecewise-constant controls at interval midpoints. These schemas are not
interchangeable. `output_stride` thins only returned states. The local optimizer
uses its separately frozen odd-grid, segment, shared-boundary, and endpoint
rules; do not translate its `sample_stride` to `output_stride`.

Local `eval_mode` is exactly `target` or `weights`. In weights mode,
`weight_mode` is exactly `by_v`, `by_v_power`, or `custom`; reversal is supplied
only through the separate boolean `weight_reverse`. Values are not lowercased
or converted from strings, so YAML booleans must be unquoted `true`/`false` and
iteration/stride counts must be integers.

Local `gain` and `gain_units` are both required. The recommended readable
unit is `(GV/m)^2 fs`; `1` in that unit equals `1e18 (V/m)^2 fs`.
The active example's `1000 (GV/m)^2 fs` therefore preserves the historical
canonical value `1e21 (V/m)^2 fs`. Gain must be finite and positive. The
result reports `field_fluence_proxy` only as a diagnostic, together with
canonical gain, vector field maximum/RMS, segment clipping fraction, and
separate per-control-axis reference field scales.

Local `initialization` is also required. For the standard basis-state transfer,
use an explicit starter field:

~~~yaml
initialization:
  method: seed_field
  amplitude: 1000.0
  amplitude_units: V/m
  max_segments: 5
~~~

The amplitude is a positive magnitude in a direct electric-field unit;
intensity units are invalid. This branch retains the existing mode-specific
trigger and applies the seed before the componentwise field limit. To request
no injected starter field, provide exactly `initialization: {method: none}`.
That branch checks the initial `weights` response against `drive_abs_min`, or
the initial `target` overlap against `c_abs_min`, and raises before propagation
when the configured condition is a zero-control fixed point. It never silently
falls back to a seed. Results report the selected method and seeded segment
count.

GRAPE minimizes
`1 - fidelity + (lambda_a / 2) * sum(field_samples**2)` and differentiates the
actual normalized dense NumPy RK4 steps. Its generated or sampled initial
field is required; the solver does not invent a zero seed. GRAPE supports only
`control_axes: xy` and rejects `propagator_func` because a custom propagator
has no matching discrete derivative.

Spectral bands must be nonempty finite center/positive-width pairs with a
supported frequency unit. Mode is `pass` or `stop`, combination is `max` or
`sum`, and weights apply only to `sum`. The spectral alpha scale and optional
sum weights are nonnegative; invalid values raise rather than being clipped or
repaired.

## Python-supplied fields and interval controls

YAML is not mandatory when samples are constructed in Python. GRAPE and
`legacy_batch_overlap` accept old-grid field samples through
`initial_field_kind: sampled`; standard Krotov accepts interval samples through
`initial_control_kind: sampled`. The two shapes and key sets are deliberately
distinct.

~~~python
config["algorithms"]["grape"] = {
    "control_axes": "xy",
    "initial_field_kind": "sampled",
    "initial_field_samples": samples,
    "initial_field_units": "V/m",
}
~~~

~~~python
config["algorithms"]["krotov"] = {
    "control_axes": "zx",
    "lambda_a": penalty_value,
    "lambda_a_units": "1 / ((GV/m)^2 fs)",
    "initial_control_kind": "sampled",
    "initial_control_samples": interval_controls,
    "initial_control_units": "V/m",
}
~~~

For GRAPE/legacy, `samples` must be a finite real `(2*N+1, 2)` array matching
the canonical field grid. For standard Krotov, use
`initial_control_samples`, `initial_control_units`, and a finite real `(N,2)`
array matching the interval count. Values are copied and converted once to V/m;
the boundary never resamples, trims, pads, normalizes, or derives a signed field
from intensity. Standard Krotov also requires a positive `lambda_a` with
`lambda_a_units`, preferably the readable `1 / ((GV/m)^2 fs)` representation.
Its current route rejects spectral constraints and plotting explicitly.
