# Archived v0.2 optimization configurations

These files are retained only as historical records of earlier optimization
experiments. They use removed model keys, rely on older defaults, and omit an
explicit dipole scale. They are not supported input for `rve-optimize`.

Current runnable configurations live in `configs/`. Do not copy a physical
constant from these files or silently migrate one: create a current-schema
configuration with an explicit `dipole_scale` and `dipole_scale_units`.
