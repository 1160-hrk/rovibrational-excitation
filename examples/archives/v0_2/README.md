# v0.2 historical material

This directory is a non-executable snapshot of material written for the v0.2
API. D-115 consolidates the previously separate archive locations here without
changing file contents or calculation logic.

- `scripts/` contains former examples, helper modules, a notebook, experimental
  Local-optimizer runs, the external C++ RK4 example, and spectroscopy migration
  evidence.
- `optimization_configs/` contains incomplete v0.2 YAML documents. They are not
  accepted by the current schema and deliberately receive no invented dipole or
  other physical value.

Everything below this directory is excluded from supported example discovery,
Ruff, and smoke execution. Use it only as historical reference until an item is
individually migrated and tested.
