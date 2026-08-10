# Validation

Scientific correctness is validated by collected pytest tests, especially
`tests/physics/`, `tests/contracts/`, and `tests/integration/`.

Run the correctness suite with:

```bash
pytest -q
```

Reproducible runtime and memory measurements live in `benchmarks/` and
`tests/performance/`. The disposition and replacement evidence for the
removed standalone diagnostics are recorded in
`docs/refactoring/VALIDATION_INVENTORY.md`.

Generated images left in this directory are local diagnostic outputs and are
not authoritative reference data.
