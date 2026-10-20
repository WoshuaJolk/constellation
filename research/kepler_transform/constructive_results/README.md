# Constructive cascade and hierarchy campaign

**Full FKT: not complete.** The new result is a verified construction of the
coordinate maps needed by the cascade, a repair for intermediate-grid
underresolution, and their integration into full-domain segment hierarchies.
See the [derivation and constructive next steps](../constructive_factorization.md).

## Established results

- Interval variational flow encloses derivatives throughout physical state
  boxes, including observer motion and light time in the declared model.
- Real and complex contraction certificates establish all 64 partial
  coordinate inverses on accepted patches and determine child padding.
- Spatial subdivisions preserve the whole original six-dimensional domain.
  An independent exact-rational check verifies the final atlas covers.
- The verifier replayed **99 real/complex transfer certificates** across the
  saved campaign and checked **62 source files** by SHA256.
- A complete bottom-up recurrence uses these transfers, separable refinement,
  an algebraic adjoint and a finite nodal regularized inverse. The coordinate
  inverse certificates do not establish the full data-reconstruction inverse.
- The full repository suite reported **377 passed, 1 skipped**.

The saved source hashes from the previous segment-field campaign remain intact.

## Refinement repair

The child function is an exactly represented multilinear polynomial using
two nodes per coordinate. Errors below compare *all constructed parent nodes*
against independently evaluated physical rebasing of that polynomial.

| Parent order | Refine child before changing coordinates | Maximum nodal error |
| ---: | --- | ---: |
| 3 | No | 1.11e-3 |
| 3 | Yes | 5.05e-7 |
| 5 | No | 1.11e-3 |
| 5 | Yes | 9.47e-11 |

This isolates a real construction issue: parent refinement alone does not
resolve the intermediate hybrid fields. Separable child refinement fixes it
without a six-dimensional interpolation stencil. These are finite numerical
checks, not uniform floating-point error certificates.

## Complete original-domain hierarchies

All original state-box widths, time spans and cadences are retained.
The observer in these experiments is the declared analytic circular model.
The data are synthetic Fourier coefficients, not additional archive pixels.

| Domain | Exposures | Retained patches | Retained grid nodes | Construction, including refused attempts |
| --- | ---: | ---: | ---: | ---: |
| Near, 8 days | 8 | 1 | 5,175 | 7.17 s |
| Distant, 30 days | 32 | 1 | 22,887 | 31.61 s |
| Stress, 8 days | 8 | 10 | 51,750 | 99.74 s |

The three-node-per-coordinate setting checks geometry and graph integration;
it is **not** a sufficient likelihood-resolution rule. With 128 held-out
states, the maximum response errors divided by source-coefficient l1 norm are:

| Domain | Frequency bound 0.03 / arcsec | Frequency bound 1 / arcsec |
| --- | ---: | ---: |
| Near | 5.74e-3 | 5.40e-1 |
| Distant | 1.06e-2 | 3.40e-1 |
| Stress | 6.97e-2 | 6.28e-1 |

Query-adjoint discrepancies are below `1.8e-15` in these runs. That verifies the
algebra of the implemented operator, not its accuracy against the full physical
likelihood. No archive-scale speedup is claimed.

## Sufficient orders and counting argument

For short transfers at frequency bounds of 1 / arcsec, the new complex
certificates provide computable whole-domain interpolation bounds. The
conservative single-cell calculation selects uniform orders **239–241** for
an exact-arithmetic error of `1e-8` per unit coefficient l1 norm. This implies
roughly `1.9e14` nodes on a uniform six-dimensional grid. **These grids are not
allocated, and this calculation is not an affordable solution.**

The derivation gives a frequency-aware local-cell construction to pursue
instead. It also proves a cadence-independent counting inequality under an
explicit candidate resolution law:

```
sum_segment G(span) <= (2N-1) G(0) + D [G(total_span)-G(0)].
```

This supports an input-plus-output architecture if sufficient intermediate
grids obey that law with controlled constants. Establishing and executing
those grid rules remains necessary.

## Artifacts and reproduction

- `local.json`: whole-domain certificates, sufficient-order calculations,
  separable-refinement experiment and conditional cadence-count checks.
- `hierarchy.json`: all original-domain graphs, every retained certificate,
  construction costs, refusals, numerical errors and adjoint checks.
- `verification.json`: exact atlas-cover checks and certificate replays.
- `environment.json`: runtime versions and test result.

From the repository root, using new output directories:

```bash
OPENBLAS_NUM_THREADS=1 python -m research.kepler_transform.constructive_experiments replay-local --mode local
OPENBLAS_NUM_THREADS=1 python -m research.kepler_transform.constructive_experiments replay-hierarchy --mode hierarchy
mkdir replay-constructive
cp replay-local/results.json replay-constructive/local.json
cp replay-hierarchy/results.json replay-constructive/hierarchy.json
OPENBLAS_NUM_THREADS=1 python -m research.kepler_transform.verify_constructive_campaign replay-constructive
OPENBLAS_NUM_THREADS=1 python -m pytest -q tests/test_kepler_cascade_domain.py
```

The runner captures source hashes before execution and refuses provenance if
research source changes during the run. The verifier recomputes the geometric
certificates; it does not upgrade numerical approximation errors into proofs.
