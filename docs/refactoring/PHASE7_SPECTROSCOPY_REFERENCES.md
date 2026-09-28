# Phase 7 spectroscopy references

Last verified: 2026-09-28
Current checkpoint: P7.4-a1 analytic two-level and single-coherence references

## Purpose

P7.4 must split `spectroscopy/absorbance_calculator.py` by scientific
responsibility without using the existing implementation as its own oracle.
The references in this document are deliberately slow or closed-form,
test-only calculations. Production code never calls them.

If a reference disagrees with production, no production formula is changed
until the competing conventions and a recommendation are presented to the
user. Structural moves remain calculation-neutral.

## Existing ownership boundary

The current public spectroscopy boundary accepts a density matrix supplied by
the caller. It does **not** construct a thermal state. Temperature currently
enters number density, coherence decay inputs, and Doppler width; it does not
silently replace the caller's density matrix with Boltzmann populations.

For the two-level reference, the test independently constructs canonical
Boltzmann weights,

~~~text
p_n = exp(-(E_n - E_min) / (k_B T)) / sum_m exp(-(E_m - E_min) / (k_B T)),
~~~

and supplies `diag(p)` to the public calculator. This fixes how a thermal test
state is prepared without claiming that production owns thermal-state
generation.

## P7.4-a1 analytic response references

For energies `0` and `hbar * omega_0`, real transition dipole `mu`, population
difference `Delta p = p_0 - p_1`, and coherence decay `gamma`, the independent
per-molecule response is

~~~text
R(omega) = mu^2 Delta p / hbar * [
    1 / (omega - omega_0 - i gamma)
    - 1 / (omega + omega_0 - i gamma)
].
~~~

Both resonant and counter-rotating terms are retained. The test then applies
the documented susceptibility and absorbance conversion directly, including
the complex square root; it does not call a private production response
helper. The public loop route agrees to a maximum relative discrepancy below
`2.2e-16` on the fixed grid.

For a single post-probe coherence `rho_10`, the accepted radiation/PFID
transform convention is independently evaluated as

~~~text
R_rad(omega) = i mu rho_10 / (omega - omega_0 - i gamma).
~~~

The public radiation result agrees below `1.6e-16` relative discrepancy, and
PFID is exactly the same stored array value under the current public contract.
This reference fixes the denominator orientation, Fourier sign, coherence
phase, transition-index order, and radiation sign before any code movement.

## Remaining required references

The following P7.4 references remain before the corresponding production
responsibility moves:

- transition-specific Doppler broadening against a direct normalized Gaussian
  convolution, including the Lorentzian-to-Voigt limit;
- Gaussian and sinc-family device-function normalization and grid convention;
- exact response-route equivalence against the analytic response rather than
  only against another production route;
- response-to-absorbance limiting behavior and any defensible sum/area rule;
- an explicit decision whether a future thermal-state constructor is in
  scope. It does not exist today and will not be inferred during refactoring.

## Evidence

`tests/physics/test_spectroscopy_analytic_reference.py` contains no import from
the production implementation module beyond the public calculator and data
types under test. Its expected response, Boltzmann weights, frequency
conversion, susceptibility, refractive index, and absorbance are constructed
directly from authoritative constants.

P7.4 decomposition is not yet authorized for responsibilities whose independent
reference remains open. P7.4-a1 changes no production calculation. The focused
spectroscopy suite passes 35 tests; the full suite passes 1398 tests with 10
optional-GPU skips (1408 collected), and branch coverage remains 80%.
