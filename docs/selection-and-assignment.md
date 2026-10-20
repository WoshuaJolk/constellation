# Calibration uncertainty, finite selection functions, and exact assignment

Version 0.5 continues priorities 5–9: `learn_null`, `weigh_stamp`, `map_complete`,
`inject_all`, and `match_all`. The guarantees below apply to explicit input models
and finite candidate sets. The algorithms use established probability and
combinatorial optimization; this release does not claim a mathematical breakthrough.

## Brightness and calibration uncertainty in learn_null

Each hypothesis may now carry `orbit_id` and nonnegative `reference_flux`.
Supply the full **joint prior** over orbit and brightness in `prior_weight`.
All hypotheses must have an orbit label if any do. The tool retains the full
joint posterior and sums it to `orbit_posterior`; it does not optimize away
unfavorable brightness alternatives or invent a brightness prior. Use
`"ranking_target": "orbit"` to rank future exposures by expected information
about the orbit after marginalizing brightness. The default `"hypothesis"`
continues to rank information about the entire discrete state.

For a measured selection efficiency with finite independent calibration trials,
add `calibrations`:

```json
{
  "id": "selection-a",
  "recovered": [8, 2],
  "injected": [10, 10],
  "beta_prior_alpha": 0.5,
  "beta_prior_beta": 0.5
}
```

Counts are ordered like `hypotheses`. The two prior parameters are positive and
default to Jeffreys' Beta(½,½) prior. Counts are integers from zero to 10⁹.
An observed row or future candidate can use
`{"id": "exposure-id", "calibration_id": "selection-a"}` instead of a fixed
probability vector. Given k recoveries in n independent calibration trials,
the conditional efficiency posterior is Beta(a,b), with
`a=prior_alpha+k`, `b=prior_beta+n-k`.

If m non-detections reuse the same calibration, their joint likelihood is

`E[(1−p)^m] = B(a,b+m)/B(a,b) = product_j (b+j)/(a+b+j)`.

The implementation sums stable logarithms of the final product. It does **not**
raise the posterior mean miss probability to the mth power. The next detection
probability is `a/(a+b+m)`. Calibration posteriors are reported conditional on
each hypothesis. For example, two non-detections with an otherwise unmeasured
Jeffreys efficiency have likelihood 3/8, not 1/4.

The same calibration ID asserts a shared unknown efficiency and conditional
independence of the real outcomes given it. Different IDs assert independent
efficiency parameters. Calibration trials must represent the same selection,
brightness, rate and coverage condition; overlapping image placements are not
automatically independent Bernoulli trials. Do not use the 15 overlapping live
example placements as binomial counts without establishing this assumption.
Known off-chip probability zero should use an exact probability vector, not an
uncertain calibration. Calibration images must be independent of evaluated data.

```bash
magictools learn_null examples/null-calibration.json -o calibrated-posterior.json
```

## Empirical models in weigh_stamp

`weigh_stamp` accepts `psf_image` / `psf_hdu`, using the same native-pixel measured
PSF as `force_track` for moving, stationary and dipole models. The extended-source
and narrow-streak families remain the documented Gaussian alternatives.

Use `stamp_radius_pixels` (integer 1–63) to extract a stamp around original-image
`x,y`, including a correspondingly cropped variance and mask. The output records
exclusive-stop `stamp_bounds`; full-frame FITS hashes remain in provenance.
This works directly on a ZTF prepared multi-extension image.

An optional `empirical_artifacts` list accepts objects with `id`, `image` and
optional `image_hdu`. Each FITS plane must match the **final stamp shape** and
be registered at the intended artifact position. Supply background-subtracted,
independently estimated templates; selecting a template from the candidate's
own noise invalidates the evidence interpretation. Negative lobes are retained;
positive template mass is normalized to one **before masking**. The amplitude
prior is therefore in units of positive-lobe flux. The new `empirical_artifact`
class averages evidence over its supplied templates with uniform weights. If
explicit class priors are supplied, include this class too. At most 128 templates
are supported. This adds instrument-specific models but does not supply a
validated real-artifact corpus or a calibrated real/bogus probability.

## map_complete: exact selection for a finite experiment

```bash
magictools map_complete examples/selection.json -o exact-selection.json
```

The example supplies exposures, a weighted population, and explicit cadence
rules. An exposure has `id`, a unique physical `observation_id`, a `surveys` list,
and `mjd_utc`. A shared physical exposure belongs to several surveys in **one
row**; entering it twice is rejected. Each object has `id`, nonnegative `weight`,
and a `detection_probability` vector in exposure order. Weights define the finite
population distribution and are normalized. Detection probabilities must already
include the relevant usable footprint, photometric selection and calibration;
catalogue limiting magnitude alone does not provide these probabilities.

Each discovery rule has `id`, `surveys`, `min_visits_per_night`, `min_nights`, and
optional `max_night_span`. It selects all exposures belonging to any listed
survey, counts distinct detections within a night, then requires enough
qualifying nights. Nights are `floor(MJD_UTC − boundary_hours/24)`, with
`night_boundary_utc_hours=12` by default. A bounded span means the difference
between first and last qualifying **night indices**, not exact elapsed exposure
hours. All possible observing-night windows are checked. The same exposure
cannot count twice within a combined rule. A cadence-qualified object is not
automatically orbit-linked; actual pipeline acceptance needs a richer rule/model.

For every object, `map_complete` enumerates all binary detection patterns with
probability `product_i p_i^d_i (1−p_i)^(1−d_i)`, evaluates every rule on the same
pattern, and sums probabilities of each rule and their union. This preserves
dependence from shared exposures and overlapping rules. It does not add survey
completenesses or assume independent survey discoveries. Probability-mass
normalization is checked. The result is exact to floating-point arithmetic under
**conditionally independent per-exposure detections**. Weather, shared templates,
calibration errors, blends and linking can violate that assumption.

Exact enumeration is limited to 20 exposures and a 50-million work budget.
Exceeding a limit fails explicitly and recommends the Monte Carlo tool. Up to
32 rules and 10,000 finite population objects are supported, subject to the work
budget. The reserved rule name `union` denotes selection by any supplied rule.

## inject_all: population selection Monte Carlo

```bash
magictools inject_all examples/selection.json -o simulated-selection.json
```

The same manifest can add `draws_per_object` (default 10,000; range 100–1,000,000),
`seed` (default 2026), and `confidence` (default 0.95). For every object, this draws
the same independent Bernoulli exposure experiment used by the exact tool,
reusing each realization across all survey rules. It reports successes, estimated
selection probabilities, exact Clopper–Pearson intervals for each object's union,
and a weighted Hoeffding interval for the finite population's union. The latter
uses radius `sqrt(log(2/delta) * sum(w_i^2) / (2*N))` because object simulations
are independent. These are Monte Carlo error intervals, not uncertainty in
selection calibration or in the supplied population. Intervals are marginal,
not simultaneous across every reported object. Work is capped at 100 million
rule–exposure draws and results are reproducible for a fixed software environment.

This is population-level **selection simulation**, not additional pixel-level
image injection. `inject_tracks` performs the latter. A scientifically validated
end-to-end population experiment needs both, with efficiencies appropriate to
the full discovery process.

Optional `population_size` is the **mean of a Poisson population**, not a fixed
integer census. Poisson thinning gives expected selected count `population_size`
times the weighted union probability. Optional `observed_count` adds its Poisson
count log-likelihood. Monte Carlo likelihoods use the point efficiency estimate
and explicitly say so; calibration and population uncertainty are not integrated.
Impossible positive counts at zero mean yield `zero_probability=true` and a
`null` log-likelihood rather than invalid JSON infinity. These are count-only
likelihoods, not fits to orbital-property distributions.

## match_all: finite assignment with an integer proof certificate

```bash
magictools match_all examples/matching.json -o assignment.json
magictools verify_match assignment.json -o certificate-check.json
```

Supply `detections` with `id` and integer `unassigned_score`, and `candidates`
with `id`, unique `detection_ids`, and integer `score`. Each candidate represents
an already generated, physically evaluated orbit/association hypothesis. Scores
are additive utilities: selected-candidate scores plus scores of unassigned
detections. Integer-scaled log-likelihoods are possible, but any quantization is
the caller's responsibility and changes the objective being certified. Floating
scores are rejected. `score_units` records the chosen meaning.

The optimizer selects optional candidates so each detection belongs to at most
one. It supports 4,096 detections, 128 candidates, and `node_limit` up to 100,000
expanded branch nodes (default 100,000). It may leave detections unassigned.
It does not generate orbital hypotheses, require all detections to be explained,
or assign a probability that a link is astrophysically correct.

The guarantee follows from three elementary facts:

1. Start with all detections unassigned. A candidate's benefit is its score minus
   the unassigned scores it replaces. Nonpositive-benefit optional candidates
   can be omitted without lowering the maximum objective.
2. Every feasible solution either includes a chosen pivot candidate or excludes
   it. Including it removes every overlapping candidate. The binary certificate
   tree covers both possibilities at every expanded node.
3. Partition remaining candidates into cliques of pairwise-conflicting candidates.
   At most one member of each clique can be selected. The current score plus
   the sum of clique-wise maximum benefits is therefore an upper bound.

Each certificate leaf contains the clique partition, and each branch contains
its pivot and both children. The verifier rebuilds the remaining set along each
path, checks every pairwise conflict, confirms no candidate was omitted or
duplicated, checks the proposed assignment's feasibility, and recomputes bounds
using Python integers. It does not rerun the optimizer or trust a solver's status.
The certificate embeds its finite input problem; provenance separately identifies
the original manifest. Verification proves that embedded problem, so compare it
to the intended input before relying on a certificate from elsewhere.

`status="optimal"` means the verified upper bound equals the feasible objective.
If the work limit is reached earlier, `status="bounded"` reports the incumbent,
an upper bound, and a nonzero integer gap. A partially searched tree is never
reported as an optimal solution. The result supplies one optimum, not a proof
of uniqueness. The guarantee remains exact even for integer scores above 2⁵³.

The example defeats greedy assignment: one conflicting candidate scores 10,
whereas two compatible candidates score 6 each. The optimum is 12. With
`node_limit=1`, the tool honestly returns objective 10, upper bound 12, gap 2.
Random small problems are checked against independent exhaustive subset search;
tampered bounds, invalid cliques, omitted candidates and duplicate assignments
are rejected by tests. This is exact finite set packing, not the catalogue's
unrestricted global physical-completeness theorem.
