# solve_itf

Status: experimental reference engine. The full-ITF objective below is unchanged
by the deliberately small first implementation. Neither a fast synthetic example
nor a finite orbit bank establishes completion.

## Objective

Given a versioned complete ITF snapshot, infer which tracklets belong to the
same orbiting bodies, their orbital distributions, and competing global
assignments. Return a searchable ranked representation within 180 seconds on
declared hardware, accounting for possibilities excluded from detailed search.

This is catalogue-level inference over recorded detections. It does not recover
sources absent from ITF and does not replace Kepler's all-trajectory pixel
transform or FitSky's joint pixel inference. Their research engines, objectives,
observations and archived results remain unchanged.

### Acceptance contract

- Name and hash the input snapshot. Report its observation/tracklet count,
  time span, sources of uncertainty, time scales and observer conventions.
- Declare the orbital domain, dynamics, error model, association/count priors,
  selection assumptions and tolerated approximation errors before benchmarking.
- Cover all tracklets in that declared snapshot. Limits must reject an input
  or produce an explicitly incomplete result, never silently take a subset.
- Rank linkage events and mutually consistent global assignments. Unlinked
  tracklets and bad measurements must remain possible explanations.
- Support progressive output and queries for any specified combination.
  Do not try to materialize every subset or every possible orbit.
- Distinguish exact group membership from "these tracklets share a source,
  possibly with others." These are different posterior events.
- Quantify omitted alternatives. Normalizing probabilities over a shortlist is
  insufficient. Continuous-domain exclusions and assignment truncation require
  separate accounting.
- Report cold construction, cache construction, input processing, propagation,
  indexing, inference, result construction and query cost separately.
- Target <=180 seconds for a full prepared-snapshot search and scientific
  iterations, including invalidated cache rebuilds. Download latency is separate.
  Report the cold cost and memory as well; do not hide work in preparation.
- At the deadline, report success only if the accuracy contract is met.
  Otherwise return the current results and unresolved bounds with an incomplete
  status. An empty shortlist must not imply an empty sky.
- Compare against strong established linking methods under equivalent hardware,
  input, caching and accuracy requirements.
- Validate independent dynamics, exhaustive small controls, deliberately
  ambiguous scenes, hidden known-object identities, and a frozen full snapshot.

The benchmark hardware, real ITF snapshot, scientific domain and numerical
tolerances have not yet been selected. The complete 180-second target remains
unmeasured and unachieved.

## First executable milestone

Install this branch and generate an independent synthetic 45-day scene:

~~~sh
python -m pip install '.[test]'
constellation schema solve_itf
constellation example solve_itf --output itf-example
constellation validate solve_itf itf-example/input.json
constellation solve_itf itf-example/input.json --output itf-run
constellation inspect itf-run
~~~

The equivalent run spelling is:

~~~sh
constellation run solve_itf itf-example/input.json --output itf-run
~~~

There is no new dependency extra and no hosted inference service. Existing
directories are never overwritten. The shared agent runner hashes the manifest,
stores the resolved input, and writes report.json, provenance.json and run.json.
A successful command means the report was produced; inspect its scientific
status and completeness fields.

### Admitted input

This reference admits 1--12 tracklets, 2--32 observations each, and 1--4096
explicit weighted orbital states. Larger inputs are rejected, not truncated.
These are reference-engine limits, not an acceptable limit for the objective.

Each observation needs:

- A globally unique observation ID, tracklet ID and observatory identifier.
- Reception MJD in TDB and observer heliocentric ICRS position in AU.
- RA/Dec in degrees and isotropic small-angle uncertainty in arcseconds.

The orbit prior contains positive weights and six-component Cartesian states
[rx, ry, rz, vx, vy, vz] in AU and AU/day at epoch_mjd_tdb, in the same frame.
Weights are normalized. They are a declared finite prior, not automatically
valid quadrature of a continuous prior. Splitting a state into duplicate states
whose weights sum to its original weight leaves inference unchanged.

The observatory ID is provenance; no observatory lookup, UTC-to-TDB conversion,
aberration correction, debiasing or raw MPC80/ADES ingestion is performed here.
Do not copy timestamps or apparent coordinates into these fields without
preparing them in the stated convention. A synthetic observer is not an MPC
observatory ephemeris.

No candidate fitting or orbit proposal construction occurs: supplied states are
evaluated using Kepler's unmodified universal-variable propagator and
reception-observer light time. Propagation failures abort the calculation rather
than remove prior mass.

### Declared statistical model

For observed unit direction u and predicted unit direction v, use a normalized
von Mises-Fisher density on the sphere, with kappa = 1/sigma_rad^2:

    f(u | v) = kappa / [2*pi*(1-exp(-2*kappa))] * exp(kappa*(u.v-1)).

The unlinked baseline is independent uniform directions, 1/(4*pi), for every
observation. The tracklet-orbit likelihood ratio R[j,r] is the product of these
density ratios over observations in tracklet j. Calculations use logarithms and
squared unit-vector differences, avoiding angular-wrap and near-unity dot-product
cancellation.

This background is intentionally simple and is not a fitted distribution of
real ITF tracklets. It ignores selection, correlated astrometry, calibration and
motion within the unlinked baseline. Its probabilities are conditional reference
results, not calibrated real-survey association probabilities.

For a set partition P of the tracklets, define block potentials:

    w({j}) = 1                                       (unlinked)
    w(S) = a * sum_r pi[r] * product_{j in S} R[j,r]  (|S| >= 2)
    W(P) = product_{S in P} w(S)
    Pr(P | data, declared model) = W(P) / sum_Q W(Q).

Here a is link_activity, a positive potential per linked group. It is not
automatically an astrophysical population rate. This defines a finite
conditional partition model; it is not presented as a derived survey
selection/count model.

A tracklet appears exactly once in each partition. Overlapping competing
candidates contribute through different global assignments. Different groups
draw independently from the finite orbit prior. An unlinked singleton is a
baseline state, not an inferred orbital posterior or a claim that it is false.

### Unexplored mass accounting

The search uses canonical set-partition prefixes: the next tracklet either
joins one existing block or starts a new block. This creates disjoint subtrees
covering every partition exactly once, without an arbitrary pair gate.

Let k tracklets be assigned in a prefix P. For every prefix singleton i define

    C_i = max(1, a * sum_r pi[r] R[i,r]).

For each unassigned tracklet j define

    B_j = max(1,a) * max(1,max_r R[j,r]).

Every completed partition in that subtree has weight at most

    H(P) = W(P) * product_{prefix singletons i} C_i
                * product_{unassigned j} B_j.

Why: adding observations to an already linked block increases its potential
by at most their maximum likelihood ratios. Maturing a prefix singleton costs
at most a*E[R_i] times those ratios. A group made entirely of suffix tracklets
costs at most a times their maxima. The B factors cover all such groups and
unlinked singletons, conservatively.

The exact number of completions for r remaining tracklets and b existing blocks is

    F(0,b) = 1
    F(r,b) = b*F(r-1,b) + F(r-1,b+1).

Therefore F(r,b)*H(P) bounds that subtree's total weight. Summing these bounds
over the disjoint frontier gives U, an upper bound on unexplored weight. If L is
the sum of evaluated complete-partition weights, then

    Z in [L, L+U]
    unexplored posterior mass <= U/(L+U).

For an event with evaluated weight A,

    Pr(event) in [A/(L+U), (A+U)/(L+U)].

For an individual completed assignment of weight w, the tighter interval is
[w/(L+U), w/L], because it cannot reappear in an unexplored subtree.

The logarithmic unresolved-mass ceiling is also returned. A nonempty frontier's
reported positive ceiling is floored at the smallest positive float if exponentiation
underflows, rather than reported as zero.

These are analytic truncation bounds evaluated in float64. No directed-rounding
or numerical error certificate is claimed. They exclude neither bad modes nor
difficult propagations, but they do not account for orbits absent from the
declared finite prior. The ceilings can be very loose; reducing that looseness
without losing coverage is an important next research problem.

### Reading a report

- finite_prior_complete: every assignment in this small finite-prior model was
  evaluated. It does not mean the full scientific objective is complete.
- incomplete: a time/node budget or requested residual-mass threshold stopped
  enumeration. Look at unresolved_mass_upper and each probability interval.
- ranked_candidates: exact linked-group events, sorted by lower probability
  bound. Overlapping intervals do not establish their ordering.
- ranked_assignments: the strongest completed global assignments, with linked
  groups and unlinked singletons. returned_assignment_order_resolved compares
  them against an upper bound on every unexplored assignment's score.
- queries: probabilities that all listed tracklets share one group, possibly
  with additional tracklets. Set queries in the manifest before running.
- orbit_probabilities_given_block: discrete orbital posterior conditional on
  that exact group. The corresponding IDs/states are included in the report.
- top_k limits output, not search. Omitted already-evaluated rows are counted
  separately from unencountered candidate events.
- full_itf_search_complete and continuous_orbit_coverage_certified remain false.
  excluded_continuous_orbit_mass_upper is null, meaning unknown, not zero.
- A timeout during likelihood construction returns no rankings, mass bound 1,
  and [0,1] for requested query events. Partial geometry is not normalized.

Budgets are checked between propagation batches and partition expansions.
max_seconds defaults to 180, but this is a cooperative engine budget, not a hard
wall-clock guarantee. JSON reading, schema admission, writing, and process
startup are additional CLI costs. An in-flight numerical kernel cannot be
interrupted by this implementation. No three-minute full-snapshot claim follows.

### Validation

tests/test_solve_itf.py compares with independently enumerated labelled
assignments, then deduplicates partitions. It checks partial-search enclosures,
global exclusivity, multimodal alternatives, prior-weight splitting, permutation,
extreme scores and deadlines. Orbital predictions are checked against independent
DOP853 integration and Brent light-time roots. The runnable example uses analytic
circular motion and a separate scalar light-time solution over 45 days.

The solve_itf reference workflow runs these checks on Python 3.10 and 3.12,
installs the distribution, and executes the CLI outside the checkout. It retains
JUnit, input and report artifacts. These controls do not substitute for real ITF
data or a full-snapshot performance benchmark.

## Next milestones

1. Select and hash a real snapshot; establish tracklet grouping, observatory and
   time conventions, uncertainty metadata, duplicate handling and licensing.
   Build an ingestion audit before claiming search coverage.
2. Develop continuous orbit support from each short arc, including distance and
   radial velocity uncertainty. Compare against independent orbit solvers.
3. Build a compact overlap index with exclusion accounting. Measure construction,
   memory, query cost and residual-bound tightness as counts and spans grow.
4. Replace the uniform background and independent-error model with justified
   selection, tracklet, outlier and shared-calibration models. Validate ambiguous
   associations under those models.
5. Compare against this exhaustive reference and established linkage algorithms.
   Preserve original benchmarks and failures.
6. Attempt the frozen full-snapshot 180-second contract, including result access.
   If coverage or accuracy is unresolved, report that explicitly.

No Kepler or FitSky research code is modified by this milestone.
