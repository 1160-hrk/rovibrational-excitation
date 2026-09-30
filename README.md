# rovibrational-excitation

[![CI](https://github.com/1160-hrk/rovibrational-excitation/actions/workflows/ci.yml/badge.svg)](https://github.com/1160-hrk/rovibrational-excitation/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[日本語](README_JP.md)

A Python library for time-dependent rovibrational quantum dynamics driven by
laser fields. The current development version is `0.3.0.dev1`; its API is
intentionally incompatible with v0.2.

The library provides explicit-unit model construction, generated or externally
sampled electric fields, typed propagation choices, optimal-control methods,
linear-response spectroscopy, batch execution, strict result/checkpoint
schemas, and plotting helpers.

## Current status

CPU calculations are the verified production path. CI runs Python 3.10–3.13,
the complete CPU suite, physics references, branch coverage, supported
examples, type checks, and clean-wheel imports; see the
[CI workflow](.github/workflows/ci.yml).

Real-CUDA execution is not yet verified. The low-level RK4 and split-operator
implementations now keep their arrays on device and return backend-native CuPy
arrays. CPU-backed graph tests are not CUDA evidence: mandatory real-GPU parity,
norm, shape, transfer, and performance checks remain. A requested CuPy backend
never silently falls back to NumPy.

## Supported physical models

| Model | Coupling and current scope |
|---|---|
| Two-level | Scalar coupling; explicit gap and dipole units |
| Vibrational ladder | Scalar harmonic or Morse ladder; Morse requires nonzero anharmonicity |
| Linear molecule | Harmonic or Morse vibration with rotation; Cartesian M-resolved propagation or scalar-z incoherent M averaging |
| Symmetric top | Rigid parallel band in signed `|v,J,K,M>` order; CH3F ortho/para filtering; NumPy dense/CSR RK4 only |

Symmetric-top simulation requires explicit axes, all constants and units, and
exactly one nuclear-spin isomer sector. CuPy, split operator, all-isomer pure
states, and optimization are rejected explicitly for this model.

## Numerical capability summary

- Pure-state Schrödinger propagation supports RK4 with NumPy dense or CSR
  storage. The Numba CSR path performs sparse matvecs inside the compiled RK4
  loop.
- Split-operator propagation provides the exact Cartesian interaction route and
  an explicit helicity-projected approximation. The interaction mode is always
  required; it is never selected by fallback.
- Incoherent mixtures use a dedicated mixed-state propagator and normalized
  statistical weights.
- Density-matrix/Liouville propagation supports NumPy dense RK4 only.
- The electric-field sampling interval is half a propagation step. Canonical
  RK4 fields therefore contain `2 * n_steps + 1` samples.
- The interaction convention is `H(t) = H0 - mu E(t)` throughout.

See [physics contracts](docs/refactoring/PHYSICS_CONTRACTS.md) and the
[Cartesian split-operator explanation](docs/CARTESIAN_SPLIT_OPERATOR.md) for
the fixed conventions and limitations.

## Installation

Python 3.10 or newer is required.

```bash
pip install rovibrational-excitation
```

For the current source checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,io,plot]"
```

The optional `gpu` extra installs `cupy-cuda12x`; select it only for a matching
CUDA environment and keep the unverified status above in mind.

## Quick start

This generated-field example is executed directly by the README contract test.
Every physical scalar carries an explicit unit, and the caller explicitly
selects the generated-field route with `field=None`.

```python
# README_SMOKE
import numpy as np

from rovibrational_excitation import run_simulation_case

params = {
    "basis_type": "twolevel",
    "energy_gap": 0.2,
    "energy_gap_units": "rad/fs",
    "dipole_scale": 3.0e-30,
    "dipole_scale_units": "C*m",
    "t_start": -10.0,
    "t_start_units": "fs",
    "t_end": 10.0,
    "t_end_units": "fs",
    "dt": 0.1,
    "dt_units": "fs",
    "duration": 4.0,
    "duration_units": "fs",
    "t_center": 0.0,
    "t_center_units": "fs",
    "envelope_kind": "gaussian_fwhm",
    "modulation_kind": "none",
    "carrier_frequency": 0.05,
    "carrier_frequency_units": "PHz",
    "amplitude": 1.0e8,
    "amplitude_units": "V/m",
    "initial_states": [0],
    "backend": "numpy",
    "storage": "dense",
    "algorithm": "rk4",
    "return_traj": True,
    "sample_stride": 1,
    "nondimensional": False,
    "renorm": False,
    "save": False,
}

population = run_simulation_case(params, field=None)
np.testing.assert_allclose(population.sum(axis=1), 1.0, rtol=1e-9, atol=1e-11)
print("final populations:", population[-1])
```

For externally supplied waveforms, construct `TimeGrid` and `ScalarField` or
`CartesianField` explicitly. Samples are copied and converted once to the
canonical V/m representation; they are never resampled, padded, normalized, or
repaired. See
[`example_external_scalar_field.py`](examples/example_external_scalar_field.py).

## Public API

The package root deliberately exports only:

```text
__version__, ElectricField, TimeGrid, ExecutionPolicy,
PropagationProblem, PropagationOptions, PropagationResult,
run_simulation_case
```

Use explicit subpackages for the rest:

- `rovibrational_excitation.models` — schemas and model construction;
- `rovibrational_excitation.core` — low-level states, operators, units, and
  model-independent contracts;
- `rovibrational_excitation.fields` — sampled fields, envelopes, and
  modulation;
- `rovibrational_excitation.dynamics` — propagators and capability contracts;
- `rovibrational_excitation.optimization` — GRAPE, standard Krotov,
  `legacy_batch_overlap`, and local control;
- `rovibrational_excitation.spectroscopy` — standard absorption and typed
  complex analyzer response;
- `rovibrational_excitation.visualization` — optional plotting helpers.

There are no compatibility shims for the old v0.2 root convenience names.

## CLI and supported examples

Run the executable parameter template without writing results:

```bash
rve-simulate examples/params_template.py --no-save
```

List or run the three supported examples:

```bash
python examples/launcher.py --list
python examples/launcher.py --run quickstart --quick
python scripts/smoke_examples.py
```

Only the top-level files listed in [examples/README.md](examples/README.md) are
supported. `examples/archives/v0_2/` is historical migration evidence and is
not executed or repaired.

For optimization, start from one of the three strict current configurations:

```bash
rve-optimize --config configs/example_local_viblad_v3.yaml --no-plot
```

See [configs/README.md](configs/README.md) for the different time-grid
contracts, required seed/penalty/gain units, and Python-supplied field routes.

## Spectroscopy

Standard absorption uses a typed projection and returns mOD. A Cartesian
analyzer can instead return the projected complex molecular response. Analyzer
intensity/absorbance and a production thermal-state constructor are not
implemented; the library raises rather than inventing a reference field or
measurement convention.

## Results and restart

Saved simulations use immutable result generations selected by an atomic
`result_current.json` pointer. Checkpoints use the corresponding versioned pair
and validate the complete ordered expanded run before resume. Invalid,
unversioned, corrupt, or different-run data raises; no legacy fallback or
implicit repair is attempted. See [result storage](docs/RESULT_STORAGE.md).

## Development checks

```bash
pytest -q
coverage run --data-file=/tmp/rve-coverage \
  --source=src/rovibrational_excitation --branch -m pytest -q
coverage report --data-file=/tmp/rve-coverage --show-missing --fail-under=47
ruff check --no-fix src tests examples benchmarks scripts
ruff format --check src tests examples benchmarks scripts
mypy
```

The current local checkpoint is 1514 passing CPU tests with 16 optional-GPU
skips and 81% measured branch coverage. A skipped GPU test is not CUDA
evidence.

## Documentation

- [Documentation index](docs/README.md)
- [v0.3 migration guide](docs/MIGRATION_V0_3.md)
- [Parameter reference](docs/PARAMETER_REFERENCE.md)
- [Time propagation](docs/TIME_PROPAGATION.md)
- [Unit system](docs/UNIT_SYSTEM.md)
- [Sweep specification](docs/SWEEP_SPECIFICATION.md)
- [Version and release process](docs/VERSION_MANAGEMENT.md)
- [Changelog](CHANGELOG.md)

## License

[MIT](LICENSE)
