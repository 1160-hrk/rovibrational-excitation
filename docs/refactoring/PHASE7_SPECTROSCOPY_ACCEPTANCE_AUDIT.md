# Phase 7 spectroscopy acceptance audit

Verified: 2026-09-29
Status: **Numerical ownership accepted; D-123 constructor migration in progress.**

## Scope

This audit covers the current spectroscopy facade, experimental conditions,
response routes, broadening/device functions, radiation/PFID transform,
observable conversion, calculation report, dependency direction, and explicit
failure policy. It does not add a thermal-state constructor or change any
formula, threshold, polarization, pathway, or grid.

## Ownership

| Responsibility | Owner | Independent authority |
|---|---|---|
| Required experimental value/unit pairs | `spectroscopy.conditions` | D-023, condition contracts |
| Exact and explicit approximate response kernels | `spectroscopy.response` | D-112/D-113 analytic routes |
| Doppler and instrument convolution | `spectroscopy.broadening` | direct normalized convolutions |
| Radiation/PFID frequency response | `spectroscopy.transform` | single-coherence analytic transform |
| Response-to-mOD conversion | `spectroscopy.observables` | weak-susceptibility and zero limits |
| Immutable execution report | `spectroscopy.report` | facade/identity contracts |
| Validation, projection, dispatch, cache/report lifecycle | `AbsorbanceCalculator` | public spectroscopy contracts |

Kernel owners import no simulation, optimization, model, I/O, CLI, plotting,
or dynamics layer. The calculator remains orchestration rather than a second
formula owner.

## Numerical acceptance

- `matrix`, `loop`, `2d`, and `chunked` remain exact routes.
- The characterized resonant chunked accumulation difference remains below the
  fixed `2e-12` bound.
- Only `approximate_sparse` uses the required scale-relative threshold and
  reports its discarded commutator L2 fraction.
- `auto` requires both budget and chunk size and reports the executed route.
- Doppler remains transition-specific and restricted to matrix/loop.
- Gaussian, sinc, and sinc-squared device kernels retain their grids, boundary
  behavior, and discrete normalization.
- Pump-probe selection remains the explicit equal-V pre-probe mask; radiation
  and PFID remain unfiltered post-probe calculations.
- Production still consumes caller-owned density and creates no thermal state.

## Failure and fallback audit

No spectroscopy module contains a broad catch, print-only failure, warning
downgrade, ignored method option, automatic memory budget, fixed Doppler skip,
or hidden exact-route threshold. Units, phase matching, method, and
method-specific controls fail explicitly.

## Constructor migration

D-123 resolves the target semantics. Scalar coupling takes its internal axis
from `SystemModel` and rejects a Cartesian projection. Cartesian coupling
requires `CartesianProjection`, whose axes are typed and whose interaction ket
is finite, nonzero, and normalized. The named `standard_absorption` path uses
the same physical probe ket for detection, so circular detection remains the
adjoint rotating operator. All four exact routes are bitwise equal to the old
explicit same-polarization construction.

D-124 removes default axes, case normalization, and the invented first-axis
interaction ket from direct construction and its factory. Explicit arbitrary
`pol_det` remains only until callers migrate to a typed analyzer complex
response. Analyzer intensity/absorbance is a separate D-123 observable and is
not yet implemented.


D-125 establishes the required internal split without changing results: every
numerical route returns angular frequency plus complex molecular response, and
the facade performs the existing mOD conversion once before optional device
convolution. This is a prerequisite only; analyzer complex response is not yet
public and explicit legacy `pol_det` remains.


D-126 exposes this boundary through immutable `ComplexResponseSpectrum` and
`calculate_complex_response`. The result records the cm^-1 grid, projected
per-molecule response in C^2 m^2 / J, and calculation report; it never applies
the mOD conversion or device function. A typed arbitrary analyzer is still the
remaining constructor migration.

## Remaining release work

After the temporary constructor is removed and analyzer capability remains truthful, P7.4 can receive its final acceptance decision.
Phase 8 must still rewrite the stale spectroscopy facade examples together with
the English/Japanese READMEs, verify public snippets, and publish the exact
capability/limitation matrix. Real-CUDA evidence remains a separate release
gate and is not supplied by this CPU audit.
