# Current optimization configurations

Only the three YAML files in this directory are supported current-schema
optimization documents. Historical v0.2 documents live under
`examples/archives/v0_2_optimization_configs/` and must not be used as input.

## Included runs

- `reference_krotov_viblad_v3.yaml`: stored V=0 to V=3 Krotov reference.
- `example_krotov_spectral_viblad_v3.yaml`: the same four-level physical model
  with an explicit monotonic spectral constraint.
- `example_local_viblad_v3.yaml`: the frozen legacy local optimizer on the same
  explicit physical model.

Every document supplies a dipole value and unit. The value `0.3 D` belongs to
these examples; it is not a library default and is never inserted into another
configuration.

## Execution

~~~bash
rve-optimize --config configs/example_krotov_spectral_viblad_v3.yaml
~~~

`output.dir` is required and is interpreted relative to the current working
directory when it is relative. `--out PATH` overrides it. `plot.enabled` is
required; `--no-plot` explicitly overrides a true value. A requested top-level
plot failure raises instead of converting the calculation to a reported
success.

CLI overrides use a complete dotted path, for example:

~~~bash
rve-optimize \
  --config configs/example_krotov_spectral_viblad_v3.yaml \
  --override algorithms.krotov.max_iter=1 \
  --out /tmp/rve-optimization-smoke \
  --no-plot
~~~

Unknown root, section, algorithm, and spectral-constraint keys are errors.

## Control axes and time samples

`control_axes` is required and ordered. For example, `zx` means that field
column 0 couples through `mu_z` and column 1 through `mu_x`; a generated
Krotov `initial_polarization: [1.0, 0.0]` therefore drives only the z column.
The local and Krotov optimizers accept two distinct lowercase labels from `x`,
`y`, and `z`; duplicate pairs such as `xx` are errors. GRAPE currently accepts
only `xy` because no other route is implemented.

VibLadder and TwoLevel are physically scalar models. Their two columns and
axis labels are the preserved optimizer adapter, not a physical polarization
dependence. For the shown VibLadder `zx` setup, the x coupling is zero.

For Krotov and GRAPE, `field_dt_fs` is the field sampling interval and one RK4
propagation step is `2 * field_dt_fs`. The exact field length is therefore
`2 * n_propagation_steps + 1`; `output_stride` thins only the returned
trajectory and never the internal optimizer states. The local optimizer uses
its separately frozen odd-grid, segment, shared-boundary, and endpoint rules;
do not translate its `sample_stride` to `output_stride`.

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

Spectral bands must be nonempty finite center/positive-width pairs with a
supported frequency unit. Mode is `pass` or `stop`, combination is `max` or
`sum`, and weights apply only to `sum`. The spectral alpha scale and optional
sum weights are nonnegative; invalid values raise rather than being clipped or
repaired.

## Python-supplied Krotov fields

YAML is not mandatory when field samples are constructed in Python.
`run_from_config(config_dict, ...)` accepts a NumPy array through this explicit
branch:

~~~python
config["algorithms"]["krotov"] = {
    "control_axes": "zx",
    "initial_field_kind": "sampled",
    "initial_field_samples": samples,
    "initial_field_units": "V/m",
}
~~~

`samples` must be a finite real array of shape `(n_field_points, 2)` whose
length exactly matches the canonical Krotov time grid. Values are copied and
converted once to internal V/m. The boundary never resamples, trims, pads,
normalizes, or derives a signed field from intensity.
