# Global Fold Kepler: mathematical construction, certificates and failed shortcuts

**The full catalogue objective is not complete. No mathematical breakthrough
or sublinear-in-resolved-trajectories result is claimed.** This campaign aimed
at the global operator rather than further tuning a screened trajectory bank.
It produces a stronger bounded forward result and explicit reasons why several
proposed full-objective shortcuts fail.

## What was established

The new library constructs a continuous response field over **all six orbital
coordinates in the declared box before any query orbits are supplied**.
It sums arbitrary supplied Fourier response coefficients at arbitrary
exposure times. The orbit model includes all conic energy signs, moving-observer
parallax and retarded emission time.

The [derivation](../global_theory.md) proves a uniform exact-arithmetic error
bound for the entire interpolation hierarchy, including interpolation
instability at every level. Complex-time Arb/Acb certificates then establish
the required bounds over the unchanged near, distant and stress domains.
Analysis subdivisions cover every failed parent cell; no difficult orbit is
removed. The source-frequency limit remains one inverse arcsecond in each
coordinate.

For every orbit `q` and every allowed source point `s`, the resulting
exact-arithmetic kernel approximation obeys

```
|K_approx(q,s)-K(q,s)| <= epsilon,
|F_approx(q)-F(q)| <= epsilon * sum_m |data_m|.
```

This is an absolute operator bound. It is not relative accuracy near a zero,
an interval bound on the fast floating-point code, or a proof about arbitrary
detector likelihood preparation.

| Domain | Sufficient nodes `(time,kx,ky)` | Uniform kernel bound | Measured relative score error against independent solver |
| --- | ---: | ---: | ---: |
| Near, 8-day span | `(67,38,33)` | `7.878e-9` | `6.735e-13` |
| Distant, 30-day span | `(91,54,52)` | `8.962e-9` | `1.276e-13` |
| Broad near stress, 8-day span | `(293,117,36)` | `7.479e-9` | `7.629e-13` |

The first numerical column comes from whole-cell interval bounds, not sampled
errors. The last column checks 199 corners, boundaries and other query points
against a separate Python universal-variable propagator. Separate 80-digit
tests challenge the complex derivative enclosures using an unscaled universal
anomaly solver, independent of the regularized interval implementation.
Finite samples are checks of the implementation, never the uniform proof.

The observer follows the specified analytic circular orbit, and projection
uses one fixed tangent plane. These mathematical hypotheses are explicit.
This campaign uses **synthetic Fourier response fields**, not newly downloaded
web-repo images or a real-archive detection demonstration. The preceding
[chart campaign](../chart_results/README.md) has the real ZTF pixel checks;
those do not validate this new global algorithm automatically.

## What prevents calling it the requested breakthrough

The full hierarchy has
`P_l=2^(6*l+3*(L-l))` paired blocks and `G=2^(6L)` output leaves.
The conditional construction cost is
`(M+G+Q) polylog(B/epsilon)` for input size `M`, bandwidth `B`, and `Q`
explicit query values, assuming the stated quantitative regularity and kernel
oracle costs. **`G` is not absent from the theorem.** If `N` denotes resolved
trajectories at the requested physical resolution, the cost is not proved
sublinear in `N`. Merely naming an implicit field does not eliminate that cost.

The executed examples use depth one and large certified anisotropic orders,
not an empirical verification of the asymptotic depth/bandwidth regime.
Exact frequency separation reduces gate storage, but the final fields still
occupy about 86 MB, 262 MB and 1.264 GB respectively. The stress execution
peaked at about 3.49 GB process RSS. These constants matter.

The unrestricted linear transform also has a genuine nullspace: adding
constant exposure fields whose constants sum to zero changes no orbital
response. An adjoint is implemented; it is not an inverse. A useful inverse
must specify the identifiable subspace, gauge and conditioning. We did not
silently replace that requirement with an adjoint.

Remaining full-objective gaps are an economical representation at general
survey resolution, arbitrary validated observer/projection and detector
inputs, complete numerical error accounting, and a properly defined stable
inverse usable by the science pipeline. The production `magictools
fold_kepler` command retains its existing finite-bank meaning.

## Three mathematical routes were pursued

1. **Global demodulated interpolation.** The forward construction and
   quantitative certificates work on the declared domains. The six-dimensional
   resolution cost remains. This adapts established butterfly ideas; an
   implementation and application do not establish novelty.
2. **Exact dynamical regularization.** The all-energy Sundman identity makes
   the dynamics linear in regularized time, but the physical exposure clock
   depends on radius, radial velocity and energy. A single shared-clock FFT
   shortcut therefore fails. The identity did prove useful in the complex-time
   certificates. See the [derivation and obstruction](../regularization_attempt.md).
3. **An exact finite Fourier-feature lift.** Even a regular circular Kepler
   family has infinite phase separation rank: its tangent coordinates contain
   the linearly independent functions `tan(omega*t)`. A Vandermonde argument
   rules out an exact fixed-size shared phase lift. It does not rule out
   approximate or hierarchical transforms. See the
   [counterexample](../finite_features_obstruction.md).

The theory also gives a polynomial-degree lower bound for an unremoved
oscillatory phase residual. These are obstructions to specific proposals,
not a general computational impossibility theorem. They are elementary
working derivations, not claims of newly discovered mathematics or independent
specialist review.

## Failed attempts are retained

At the same physical bandwidth, the initial low-order hierarchy produced:

| Domain | Depth 1, order 16: relative error | Depth 1, order 32: relative error |
| --- | ---: | ---: |
| Near | `2.999e-3` | `1.126e-12` |
| Distant | `4.997e-1` | `3.610e-3` |
| Stress | `6.176e-1` | `1.516e-1` |

Depth-two attempts at orders 8 and 12 also failed the physical accuracy
target. Low-frequency controls succeeded but were not promoted to physical
success. [All 18 initial attempts](initial_attempts/complete.json), their
arrays, dimensional counts and executed core source snapshots are retained.
The snapshot hashes match the original run's source manifest.

Direct complex interval bounds were initially far too wide. Substituting an
exact on-root observer/velocity cancellation made useful bounds possible.
The initial stress certificates still failed the right-half-plane square-root
test in two root cells; the
[failed proof records](failed_proof_attempts/stress-rho2.json) preserve that
failure. Analysis-only subdivision then covered the same domain successfully.
An attempted global all-time collision-free/periapsis condition failed on the
distant box; it was replaced by a proved observation-branch condition, not
by removing zero-angular-momentum states.

## Timing is supporting evidence, not the mathematical result

These comparisons include the selected plan's construction, application,
query evaluation, and all three saved ellipse/order certificate attempts
for that domain. Module startup and independent validation are excluded.
The compiled direct solver is warmed; its reference calculation evaluates all
16 input Fourier modes at every actual exposure. No speed threshold is used
as a test or a completion criterion.

| Workload | Queries / exposures | New method including proof search | Compiled direct | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Near | 199 / 8 | 4.743 s | 0.00259 s | Much slower |
| Distant | 199 / 32 | 12.404 s | 0.00956 s | Much slower |
| Stress | 199 / 8 | 67.245 s | 0.00274 s | Much slower |
| Near, dense cadence | 32,839 / 1,024 | 11.768 s | 46.098 s | 3.92× faster on this synthetic workload |

The last case reduces geometry evaluations from 33,627,136 to 2,278,421,
including construction and evaluation. Its field is still built before queries
are selected. Its independently checked relative score error is `1.288e-12`.
This demonstrates useful exposure-count reduction, **not sublinear physical
output-resolution cost** or archive-scale performance. Timings are one saved
execution per workload on the development host.

## Reproduce and use the library

From the repository root, with enough memory for the stress case:

```bash
python -m pip install -e '.[test,research,accelerate]'
OPENBLAS_NUM_THREADS=1 python -m pytest -q \
  tests/test_kepler_global.py tests/test_kepler_complex_certificate.py

# Check hashes, cover accounting, error arithmetic and saved score comparisons;
# then independently rebuild all nine interval proofs:
OPENBLAS_NUM_THREADS=1 python -m research.kepler_transform.verify_global_campaign \
  --recompute-proofs

# Create new proof and execution files; existing outputs are refused:
OPENBLAS_NUM_THREADS=1 python -m research.kepler_transform.certify_global \
  new-global-proofs --partition-depth 6
OPENBLAS_NUM_THREADS=1 python -m research.kepler_transform.certified_global_experiments \
  new-global-proofs/near-rho2.json new-global-execution/near.json
OPENBLAS_NUM_THREADS=1 python -m research.kepler_transform.certified_global_experiments \
  new-global-proofs/distant-rho2.json new-global-execution/distant.json
OPENBLAS_NUM_THREADS=1 python -m research.kepler_transform.certified_global_experiments \
  new-global-proofs/stress-rho1.5.json new-global-execution/stress.json
OPENBLAS_NUM_THREADS=1 python -m research.kepler_transform.certified_global_experiments \
  new-global-proofs/near-rho2.json new-global-execution/near-1024frames.json \
  --frames 1024 --random-queries 32768
```

`SourceButterfly(phase, sources, output_dimension=6, depth=1,
order=(67,38,33), max_gate_bytes=8_000_000_000)` builds the near-domain
representation. `sources` are normalized `(time,kx,ky)` coordinates in
`[-1,1]^3`; the `ContinuousKeplerPhase` supplies the explicitly declared
physical chart, time span and bandwidth. `plan.transform(coefficients)`
returns a field; `field.evaluate(normalized_orbits, accelerated=True)` queries
it. `plan.adjoint(normalized_orbits, values)` applies the finite-query adjoint.
Changing the physical model, domain, bandwidth, depth or order requires a new
proof. The generic plan does not automatically attach or claim a certificate.

The historical initial implementation can be reconstructed in a separate
checkout of parent commit `1f8449742ec7be58a871229ff8581f1605bf5377` by adding
the four files in `initial_attempts/executed_sources` under
`research/kepler_transform`, then running `global_experiments`.
Use the current implementation for new work; historical timings belong to
their recorded sources.

`proofs/` contains every quantitative root/cell certificate, `executions/`
contains outputs and independent references, and `manifest.json` binds the
artifacts and source files by SHA-256. The [main theory](../global_theory.md)
links the primary literature and gives the complete assumptions and counts.
