# Strict nondimensionalization

This package has one scaling contract. It transforms the complete active
Hamiltonian generator and never invents a time step, field strength, dipole
moment, or energy scale.

## Public entry points

- `nondimensionalize_from_objects`: preferred path for `Hamiltonian`,
  dipole, and `ElectricField` objects. The active coupling axes and scalar
  versus Cartesian coupling must be stated explicitly.
- `nondimensionalize_system`: array path. Hamiltonian and time units must be
  stated explicitly.
- `nondimensionalize_with_SI_base_units`: array path when all inputs are
  already in SI base units.
- `determine_SI_based_scales`: derive scale metadata without transforming a
  time grid.
- `create_dimensionless_time_array`: convert an explicit, exactly divisible
  physical grid; it never adjusts `dt`.
- `analyze_regime`: neutral diagnostics with no universal weak/strong
  thresholds.

## Definition

For the selected active dipole components,

```text
H0' = (H0 - epsilon0 I) / Eref
mu' = mu / muref
E' = E / Epeak
tau = t / (hbar / Eref)
lambda = muref Epeak / Eref
Eref = max(spectral_span(H0), muref Epeak)
```

A caller-supplied positive `energy_scale_J` may replace the final `Eref`
selection. Scale provenance is recorded in `NondimensionalizationScales`.

An ordinary zero-valued `ElectricField` is rejected as ambiguous. Use
`ZeroField` for field-free evolution. Inactive scales are represented by
`None`, and a completely zero generator is rejected.
