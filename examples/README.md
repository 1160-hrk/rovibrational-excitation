# Examples

Only the top-level example files documented below are supported. They use
the current typed simulation boundary, are linted, and run as smoke tests
in CI. This catalog is generated only from top-level example_*.py files;
helpers, generated outputs, and archives are never scanned.

## Quick start

Run the smallest generated-field example:

~~~bash
python examples/launcher.py --run quickstart --quick
~~~

List all supported examples:

~~~bash
python examples/launcher.py --list
~~~

## Supported examples

- [example_external_scalar_field.py](example_external_scalar_field.py): Two-level propagation with externally sampled scalar field injection.
- [example_typed_spectral_modulation.py](example_typed_spectral_modulation.py): Two-level propagation with unit-aware spectral modulation.
- [example_typed_twolevel.py](example_typed_twolevel.py): Minimal typed two-level propagation.

All supported examples use save=False, complete in seconds, check population
normalization, and avoid plotting or persistent output.

The parameter-file template is also executed without saving by the smoke
suite, but it is not an additional example module:

~~~bash
python -m rovibrational_excitation.cli.simulate examples/params_template.py --no-save
~~~

Run the same smoke suite used by CI:

~~~bash
python scripts/smoke_examples.py
~~~

## Historical material

examples/archives contains migration evidence and scripts written against
older APIs. In particular, archives/v0_2_scripts contains the former
top-level examples and their dedicated helpers. Some require removed
calling conventions, undefined experiment-specific constants, or external
compiled extensions.

Archived files are intentionally:

- not listed or executed by examples/launcher.py;
- excluded from Ruff and CI smoke tests;
- not scanned by this catalog builder;
- not repaired by guessed physical or optimization parameters.

Use archived code only as historical reference. Before restoring one as a
supported example, migrate it to the current public API, add a bounded
quick mode, and include it in scripts/smoke_examples.py.

Verify that this generated catalog is current with:

~~~bash
python examples/tools/build_index.py --check
~~~
