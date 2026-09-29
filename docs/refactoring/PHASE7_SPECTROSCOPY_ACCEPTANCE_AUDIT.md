# Phase 7 spectroscopy acceptance audit

Verified: 2026-09-29
Status: **Numerical ownership accepted; constructor strictness decision open.**

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

## Open constructor decision

Three current behaviors choose a physical polarization when the caller omits or
case-varies input:

1. `axes` defaults to `"xy"`;
2. `axes.lower()` silently accepts uppercase/mixed case;
3. `pol_int=None` creates a unit Jones ket on the first selected axis.

D-024 explicitly accepts only `pol_det=None`: it means the same physical ket as
`pol_int` and detection applies the analyzer bra. The recommended v0.3
resolution is therefore to require `axes` and `pol_int`, reject non-lowercase
axes, and retain `pol_det=None`. This is an API/physical-input behavior change
and requires explicit user approval before implementation. Until then, an
executable acceptance-debt test records the current behavior without endorsing
it as the final API.

## Remaining release work

After the constructor decision, P7.4 can receive its final acceptance decision.
Phase 8 must still rewrite the stale spectroscopy facade examples together with
the English/Japanese READMEs, verify public snippets, and publish the exact
capability/limitation matrix. Real-CUDA evidence remains a separate release
gate and is not supplied by this CPU audit.
