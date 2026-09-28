# Phase 7 spectroscopy references

Last verified: 2026-09-28
Current checkpoint: P7.4-a2 independent spectroscopy references complete

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

## P7.4-a2 broadening and device references

The transition-specific Doppler reference begins with the analytic complex
Lorentzian above and independently constructs the normalized discrete Gaussian
kernel on the actual angular-frequency grid. Its test parameters make
`sigma_omega / delta_omega = 3` exactly through the documented

~~~text
sigma_omega = |omega_0| sqrt(k_B T / (m c^2)).
~~~

Direct convolution agrees with the production filter in the boundary-free
interior at the `5e-14` relative test bound. The normalized kernel and the
broadened imaginary-response sum are conserved at floating-point precision.
The accepted `omega_0 == 0` branch is exactly unchanged. This is a transparent
discrete Lorentzian-to-Voigt reference; it does not call SciPy or a production
broadening helper to build the expected array.

Device-function references independently construct a Gaussian kernel from the
declared FWHM and the existing odd-length sinc/sinc-squared grids. A centered
unit impulse produces the direct convolution, and every kernel has unit sum at
floating-point precision. These tests fix the current offset convention and
normalization without asserting that another endpoint convention would be
physically superior.

## Exact routes and observable limit

All four exact response routes are now compared with the analytic two-level
answer, not merely with one another. `loop`, `matrix`, and `2d` agree at the
original `5e-14` relative bound. At the resonant peak, `chunked` exhibits an
observed `1.51e-12` relative difference from sparse/dense accumulation order;
its fixed bound is `2e-12`. This is recorded floating-point behavior, not an
authorization to change an expression.

The response-to-absorbance conversion also agrees with its weak-susceptibility
limit at `2e-10` relative tolerance for dimensionless susceptibilities of
`1e-10` to `3e-10`, and maps an exact zero response to zero. A universal
absorbance-area sum rule is not asserted: the public observable contains a
complex square root, a frequency prefactor, and a caller-selected finite grid.
The valid conservation statement is instead applied at the normalized
response-broadening kernel, where the discrete imaginary-response sum is
preserved.

A future thermal-state constructor remains a possible new capability, not a
missing production owner. Its physical definition is unnecessary for moving
the current caller-density calculations and will not be inferred during this
refactor.

## Reference disposition

D-072 spectroscopy coverage is complete for the current production behaviors:
caller-supplied thermal test input, response poles, transform sign and phase,
absorption/radiation/PFID observables, Doppler and device broadening, exact
routes, and defensible normalization limits. Covered responsibilities may now
move structurally with before/after parity tests.

## Evidence

`tests/physics/test_spectroscopy_analytic_reference.py` contains no import from
the production implementation module beyond the public calculator and data
types under test. Its expected response, Boltzmann weights, frequency
conversion, susceptibility, refractive index, and absorbance are constructed
directly from authoritative constants.

P7.4-a1 and P7.4-a2 change no production calculation. The analytic-reference
file passes ten tests; the focused spectroscopy suite passes 43. The full suite
passes 1406 tests with 10 optional-GPU skips (1416 collected), branch coverage
remains 80%, and the spectroscopy monolith reaches 94%.
