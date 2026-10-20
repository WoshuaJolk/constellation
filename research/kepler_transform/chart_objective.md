# Broad-coverage campaign

The original full objective remains unchanged. This campaign targets its
continuous-domain coverage bottleneck. A newly discovered object is not required.

After numerical coordinate pilots, but before the recorded benchmarks, fix:

* Near domain: reference heliocentric distance 3 AU, 8-day stack; inverse
  topocentric axial distance 0.5 ± 0.05 AU^-1, fractional radial velocity
  center ± 0.001/day, two angular coordinates ±10 arcsec and two angular-rate
  coordinates ±2 arcsec/day. Six-dimensional, no energy-class filter.
* Distant domain: reference heliocentric distance 41 AU, 30-day stack;
  inverse axial distance 0.025 ± 0.00625 AU^-1 (32–53.33 AU axial distance),
  fractional radial velocity center ±0.0001/day; same angular widths.
* Stress domain: near domain with inverse-distance width 0.25 and
  fractional-radial-velocity width 0.01/day. Retain it even if it fails.
* Maximum continuous coordinate error: **0.0005 arcsec per coordinate**.
  This is an absolute geometry tolerance, not a detection significance.
* Cover budget: 16 leaves and 600 seconds (near/distant); 16 leaves and
  180 seconds (stress). Time limits are soft boundaries between subdivisions.
* Structured workload: eight outer points per resolved cell, 131,072 inner
  queries per outer point. Also test independently random full-6D queries.
* Reference: all-conic retarded geometry with per-exposure type-3 NUFFTs.
  Stronger comparison: the same certified polynomial geometry with per-frame
  NUFFTs. Report common construction/certification in both complete costs.
* Record every leaf, rejected parent, unresolved volume, selected exact-Arb
  score audits, query counts, model parameters, versions and source hashes.

The adaptive cover must cover the exact original root; no orbit class or
unresolved cell may be dropped. Report direct fallback separately. Geometry
certification, analytic phase truncation, sampled numerical accuracy and full
arithmetic certification are distinct claims.

Passing this campaign is not automatic completion of the original objective.
Generic globally efficient outer-coordinate synthesis, finite Fourier numerical
error bounds, calibrated real exposure likelihoods, and broad archive workloads
still require their own evidence. At least a 2x total speed advantage is the
target against the physical reference; any time-accumulation claim also needs
an advantage against the stronger same-geometry comparison.
