# Examples

Only the three top-level `example_*.py` files documented below are supported
examples. They use the current typed simulation boundary, are linted, and run
as smoke tests in CI.

## Quick start

Run the smallest generated-field example:

```bash
python examples/launcher.py --run quickstart --quick
```

List all supported examples:

```bash
python examples/launcher.py --list
```

## Supported examples

- `example_typed_twolevel.py`: generated Gaussian field and dense NumPy RK4.
- `example_typed_spectral_modulation.py`: unit-aware physical delay,
  sinusoidal spectral phase modulation, GDD, and TOD.
- `example_external_scalar_field.py`: direct injection of a validated
  `ScalarField` on an exact odd-length `TimeGrid`.

All three examples use `save=False`, complete in seconds, check population
normalization, and avoid plotting or persistent output.

Run the same smoke suite used by CI:

```bash
python scripts/smoke_examples.py
```

## Historical material

`examples/archives/` contains migration evidence and scripts written against
older APIs. In particular, `archives/v0_2_scripts/` contains the former
top-level examples and their dedicated helpers. Some require removed calling
conventions, undefined experiment-specific constants, or external compiled
extensions.

Archived files are intentionally:

- not listed or executed by `examples/launcher.py`;
- excluded from Ruff and CI smoke tests;
- not repaired by guessed physical or optimization parameters.

Use archived code only as historical reference. Before restoring one as a
supported example, migrate it to the current public API, add a bounded quick
mode, and include it in `scripts/smoke_examples.py`.
