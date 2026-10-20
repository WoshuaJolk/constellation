# Segment-composition and inverse campaign

**Full Fold Kepler/FKT objective: not completed.** This directory contains
reproducible identities, interval certificates, and a conditional cost
calculation. It does not contain a completed fast segment hierarchy.

Read the [derivation](../segment_theory.md) for the hypotheses, formulas and
limits. The [results](results.json) preserve the original near/distant/stress
domains, fitted endpoint witnesses, independently evaluated identities, and
outward-rounded whole-domain inverse-conditioning bounds.

## Reproduce and verify

From the repository root, with the `test,research,accelerate` extras installed:

```bash
OPENBLAS_NUM_THREADS=1 python -m research.kepler_transform.segment_experiments /tmp/new-fkt-segment-results
OPENBLAS_NUM_THREADS=1 python -m research.kepler_transform.verify_segment_campaign /tmp/new-fkt-segment-results
OPENBLAS_NUM_THREADS=1 python -m pytest -q tests/test_kepler_segments.py
```

The output directory must not already exist. The verifier recomputes the rank
and inverse certificates, checks source/input hashes, and replays the real-pixel
identity. NumPy/SciPy/Arb/Numba versions can affect numerical results; the saved
environment records the execution versions. Exact certificate endpoint
comparisons are intentionally strict.

## What was established

* **Endpoint ambiguity:** interval derivatives certify rank four for the
  endpoint map and rank six when a midpoint is added, at an interior state of
  each original domain. The implicit-function theorem gives a local
  two-dimensional endpoint fiber with varying midpoint. The independently
  checked stress witness has a midpoint separation of about 7.25 arcseconds.
* **Exact state composition:** the new chart uses actual observed angular
  rates. Rebase composition and score additivity retain light time and observer
  motion. Bound, unbound, and parabolic states are exercised. An independent
  DOP853/light-time calculation tests the coordinate definition.
* **A possible cost mechanism:** hidden range variables affect sufficiently
  short segments at order h². A conditional anisotropic cell count is favorable.
  The needed continuous chart-cover, interpolation and transition bounds are
  not supplied by that count.
* **Inverse conditioning:** a recomputed complex-time certificate bounds a
  unit-norm input direction over every orbit in the original near box.
  With 33 distinct epochs over one day, the certified real response-field
  inverse amplification exceeds 1e17. The construction uses nonzero spatial
  frequencies and survives conjugate symmetry. This concerns the Fourier
  response-field operator, not an asserted nullspace of arbitrary finite raw
  detector images.
* **Real data:** three archived ZTF exposures, 81 pixels each, and 64 orbit
  queries verify the segment identity with the recorded full TPV WCS and
  independent direct pixel statistics. The original Gaussian PSF and variance
  approximations remain. This is not a discovery or performance claim.

The exact near-null vectors require phase coefficients defined in exact
arithmetic. Float64 cancellation bottoms out at approximately 1e-12; that
residual does not verify the much smaller mathematical bounds. Both are
recorded separately.

## Remaining blockers

The full forward segment hierarchy has not been built and certified, so there
is no demonstrated new total-cost advantage. General detector coverage and
errors still need integration with whichever forward construction succeeds.
An inverse must specify its observable subspace, regularization and conditioning;
neither an adjoint nor a finite-query iterative solve establishes the general
inverse objective. These are unresolved research requirements.
