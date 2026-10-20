# Segment-field construction campaign

**Full objective: not complete.** This campaign constructs the actual shared
segment fields that were missing from the preceding composition experiment.
It then tries a cascade of one-dimensional remaps to avoid the six-dimensional
tensor stencil. The [derivation](../segment_field_theory.md) explains the
validated ODE, both constructions, their error accounting and remaining gaps.

## What the records establish

- Whole-box reception-coordinate flow enclosures cover the unchanged near,
  distant and stress domains. These use interval Picard inclusion, not a sample
  maximum.
- Parent fields are built from stored child fields. No parent resums the
  original exposure stack.
- The finite interpolation graph has a tested adjoint and a specified
  Tikhonov inverse, checked against a dense reference.
- The cascade reduces tensor stencil storage substantially. Its mixed-coordinate
  inverses are checked numerically at every constructed node, but have no
  uniform invertibility/analyticity certificate.
- Three actual ZTF frames are processed with their full TPV WCS, finite Gaussian
  pixel integrals, masks, variances and profiled backgrounds. Refining angular,
  rate and hidden coordinates brings the sampled relative sup errors of both
  likelihood sufficient statistics below `1e-8`.
- The broad-orbit experiments remain slower than compiled direct evaluation
  once construction is included. Their accuracy is also insufficient for the
  full objective. A sufficient whole-domain tensor interpolation bound is
  recomputed and retained even though it fails the `1e-8` target.

The synthetic Fourier trials include an exploratory low-frequency band of
`0.03 / arcsec` and a separate `1 / arcsec` trial. The low-frequency results
must not be described as a complete one-arcsecond-resolution transform.
The real-data check spans about 12 minutes and only three exposures; it does
not demonstrate affordable all-archive processing.

## Measured results

The synthetic error is the maximum held-out response error divided by the
source coefficient l1 norm. The timing ratio is direct root-node evaluation
plus direct held-out evaluation, divided by graph construction, application and
query time. A ratio below one means the new construction is slower.

| Trial | Frequency band, /arcsec | Sampled error | Total timing ratio |
| --- | ---: | ---: | ---: |
| Near, tensor | 0.03 | 2.94e-3 | 0.0226 |
| Distant, tensor | 0.03 | 1.68e-2 | 0.0248 |
| Stress, tensor | 0.03 | 4.85e-2 | 0.0263 |
| Near, refined cascade | 0.03 | 8.92e-7 | 0.00377 |
| Near, adaptive cascade, 32 frames | 0.03 | 3.16e-4 | 0.0308 |
| Near, physical-frequency tensor trial | 1.0 | 6.73e-1 | 0.0355 |

For the ZTF checks, the errors below are
`max(abs(approx-reference))/max(abs(reference))` over 128 held-out states.
They are not pointwise relative-error bounds or uniform-domain certificates.

| Angular / hidden / rate order | Numerator error | Information error |
| --- | ---: | ---: |
| 6 / 2 / 3 | 1.67e-3 | 4.31e-4 |
| 10 / 2 / 3 | 1.10e-6 | 1.34e-5 |
| 14 / 2 / 3 | 1.29e-7 | 6.63e-8 |
| 18 / 3 / 4 | 2.98e-10 | 1.10e-10 |

The last row establishes numerical accuracy on these samples, while the
synthetic table explicitly fails the joint speed-and-accuracy objective.
The final regression suite reported **355 passed, 1 skipped**.

## Files

- `forward.json`: unchanged-domain tests of both backends, a physical-frequency
  failure case, and a short-segment adaptive-order heuristic. Includes all
  geometry iterations, stored transfer entries, construction/application/query
  times, compiled direct timings and separate interpolation-error diagnostics.
- `real.json`: four order-refinement trials on the archived ZTF pixels.
  Source and input hashes are included.
- `verification.json`: source/input hash verification, recomputed whole-box
  covers, a recomputed uniform tensor error bound and real-image replay.
- `environment.json`: package versions and regression-test counts.

Source hashes are captured before each campaign and checked again afterward.
The runner refuses to certify a campaign if its research source changed during
execution. Timings vary by machine and load; successful numerical checks do not
turn them into an asymptotic or novelty proof.

## Reproduce

Run from the repository root, using new output directories:

```bash
python -m pip install -e '.[test,research,accelerate]'
OPENBLAS_NUM_THREADS=1 python -m research.kepler_transform.segment_field_experiments replay-forward --mode forward
OPENBLAS_NUM_THREADS=1 python -m research.kepler_transform.segment_field_experiments replay-real --mode real
mkdir replay-fields
cp replay-forward/results.json replay-fields/forward.json
cp replay-real/results.json replay-fields/real.json
OPENBLAS_NUM_THREADS=1 python -m research.kepler_transform.verify_segment_field_campaign replay-fields
OPENBLAS_NUM_THREADS=1 python -m pytest -q tests/test_kepler_segment_field.py
```

To replay the committed certificates and input checks:

```bash
OPENBLAS_NUM_THREADS=1 python -m research.kepler_transform.verify_segment_field_campaign research/kepler_transform/segment_field_results
```

The verifier writes `verification.json`; numerical timing fields can change.
It never upgrades the cascade's sampled checks into a uniform certificate.

## Library surface

`SegmentTransform` is available in
`research.kepler_transform.segment_field`. It accepts Fourier exposure fields
or a custom linear leaf kernel, constructs a bounded-domain representation,
and provides `transform`, `adjoint`, `operator` and `regularized_inverse`.
The returned field supports arbitrary in-domain queries and their adjoint.
The inverse explicitly operates on the finite root-nodal graph.

This is research API, not a replacement for the production `fold_kepler`
contract. The missing full-objective work is uniform cascade-map/error
certification, affordable high-accuracy resolution across broad domains,
floating-point and detector error accounting, and the scope and conditioning
of a scientifically useful inverse.
