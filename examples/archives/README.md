# Historical examples

This tree contains unsupported migration evidence grouped by the release line
whose API it used. It is not scanned by the example launcher, Ruff, or the CI
smoke suite.

Current executable examples are the top-level files documented in
[`../README.md`](../README.md). Current optimization YAML is limited to the
three documents in [`../../configs/`](../../configs/).

Do not promote an archived script or configuration by filling in guessed
physical values. Migrate it to the current typed API, add an executable bounded
reference, and move it out of this tree in a separately reviewed change.
