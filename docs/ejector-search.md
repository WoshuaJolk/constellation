# Search a catalogue and certify a bounded encounter question

`ejector_search` fits every requested-size group in a supplied astrometric
catalogue, repeats the **whole group/time search** on declared rotation controls,
independently checks fitted trajectories, and looks for source counterparts at
the historical encounter. `ejector_bound` can then decide the existence question
for each group with continuous-time interval enclosures, rather than interpreting
an optimizer failure or a small Monte Carlo sample as an exclusion.

The [committed real-data experiment](../research/ejector_search/README.md)
decides every triple in an 18-star observed sample and a 21-object recovery
sample. These are completed **finite, model-conditional decision problems**.
They do not complete the original all-survey `find_ejector` objective, identify
a new ejection, or introduce a new mathematical theorem.

```bash
python -m pip install -e '.[accelerate,research]'
magictools ejector_search survey.json -o survey-result.json
magictools ejector_bound survey.json --refine-nodes 2000 --verify-witnesses -o certificate.json
```

The optional `accelerate` extra compiles the same kick–drift–kick equations with
Numba, without fastmath. Without it, the fitter uses NumPy. The interval solver
requires `python-flint` from the research extra and does not use the fitted
leapfrog trajectory as a proof of physical flow.

## Input and outputs

Use [observed.json](../research/ejector_search/results/observed.json) as a complete
example. Fields shared with [`ejector_fit`](ejector-fit.md) are `schema_version: 1`,
`epoch_label`, `frame`, `potential`, `objects`, and `fit`. The measurement order
is parallax, pmra*cos(dec), pmdec, RV. Angular positions are fixed. Every object
requires a measured RV and a positive-definite covariance. The search has:

* `group_size`: 3–6. Every combination is accounted for.
* `lookback_bounds_myr`: positive lower/upper ages, upper at most 100 Myr.
* `radius_pc`: every member must be within this radius of the arithmetic centroid.
* `report_chi2_max`: maximum total squared Mahalanobis displacement to report.
  This is also the interval solver's **bounded domain**, not a discovery p-value.
* `max_hypotheses`: explicit resource cap, at most 100,000. Exceeding it aborts
  before output. The catalogue is never silently shortened.
* `workers`: 1–32 parallel fit processes.

`fit.starts_myr` has one age per row; all declared starts are attempted.
`controls` requires `method: "proper_motion_rotation"`, `n_controls`, and `seed`.
`catalogue_description` must describe the supplied selection. `archive_output`
names a **new** directory relative to the manifest. Optional `counterparts`
names the source-catalogue JSON described below.

The search archives every group's fit status/cost, all control ledgers, nominal
tracebacks for every star, fitted candidates, DOP853 checks, source matches,
input hashes, and a code snapshot. A `complete.json` sentinel appears only after
the requested search finishes. The path library contains nominal states;
the observation likelihoods carry uncertainty, not those single paths.

The bounds command writes a separate `<archive_output>-bounds` directory with
expanded position enclosures and a certificate. Its outcomes for each group are:

* **Excluded:** no allowed encounter in the stated continuous model domain.
* **Certified witness:** at least one allowed encounter, verified by interval
  propagation and an interval bound on measurement cost.
* **Unresolved:** neither conclusion was established. A resource limit, failed
  enclosure, or unsuccessful fit always leaves uncertainty explicit.

`all_groups_decided_under_model` is true only when no group remains unresolved.
It answers existence per group; it does **not** enumerate every possible origin
or mode of a compatible group. `full_objective_complete` stays false.

## Why the exclusions cover continuous time and measurement uncertainty

For a group with total cost at most C, every individual measurement displacement
satisfies `|delta_j| <= sqrt(C * covariance_jj)`. This gives conservative initial
boxes even when observations are correlated. Parallax is restricted to the
declared positive lower bound. Observation-to-state conversion uses recorded
float64 frame coefficients interpreted as fixed model inputs.

All subsequent enclosure arithmetic uses 128-bit
[Arb balls](https://python-flint.readthedocs.io/en/latest/). The supported softened
Galactic force fields are smooth. For a state box Y, negative time step h, and
candidate tube Z, the code verifies Picard self-inclusion:

```
Y + [h, 0] f(Z) ⊆ Z,       f(x, v) = (v, a(x)).
```

The tube contains the entire segment for every allowed initial state. If A
encloses acceleration throughout that tube, endpoints are enclosed by
`x0 + h*v0 + h²*A/2` and `v0 + h*A`. The next step starts from these boxes.
A failed enclosure is replaced by an unbounded outer set, never a discarded
trajectory. Conversion to stored binary64 endpoints is checked against the
Arb bounds before use.

At every time segment, expand each stellar position box by the encounter radius
along every axis. A centroid-radius encounter requires these expanded boxes to
share a center. If they have no intersection in every segment, an encounter is
excluded. Overlap does not establish one.

Unresolved groups can be subdivided in latent Gaussian coordinates. An Arb
Cholesky factor encloses the factor of the exact supplied covariance. The root
box covers the entire `sum(z²) <= C` ball. Every split covers its parent, and a
branch is discharged only by a cost lower bound, impossible parallax domain,
or disjoint continuous trajectory tubes. **Every branch must be discharged to
exclude a group.** The certificate retains all terminal leaves and pending boxes.
The verifier separately checks leaf coverage using exact rational volumes and
non-overlap, in addition to replaying the interval calculations.

Positive witnesses are proposed by a stricter common-center fit, then checked
with independent Arb propagation, a bound on centroid radius, and a bound on
the total measurement cost. The exact-center optimizer result itself is not an
existence proof. The witness checks a positive-radius encounter at one exact
represented age; it does not certify equality at mathematical radius zero.

This uses established interval-analysis and branch-and-bound ideas. The
implementation, Arb, and the stated physical model are trusted; it is not a
proof-assistant-verified program. Frame and Galactic-potential uncertainty,
angular uncertainty, binaries, stellar/cluster interactions, and unobserved
sources lie outside this certificate. Fit `max_sigma` restrictions are relaxed
in the enclosure search, making exclusions conservative for the fitter's domain.

## Controls and source identification

Each control rotates every star's proper-motion vector and its complete
observation covariance coherently. Position, parallax, RV and proper-motion
magnitude remain fixed. The reported search-wide rank is
`(1 + controls_with_minimum_cost <= candidate_cost) / (B + 1)`, with ties counted
against the candidate. All groups and starting ages are searched again.

This assumes independent uniform proper-motion directions conditional on the
retained scalar quantities. It is not established for a literature-selected
runaway sample or a real anisotropic Galactic population. Ranks are **control
diagnostics**, not calibrated astrophysical false-association probabilities.
Choices made across runs and models are not included in one run's rank.

A counterpart catalogue has schema 1, matching `epoch_label`, `description`,
`coverage`, `objects`, and an `excluded` ledger. Each object has `id`, six-column
`icrs` (RA, Dec, distance, pmra*cos(dec), pmdec, RV), positive `match_radius_pc`,
and optional metadata. Sources are propagated **backward to the encounter**,
not matched at their present spatial position. All within-radius matches and
the five nearest sources are reported. Missing RVs are never filled with zero.

The committed Hunt–Reffert adapter accounts for 7,167 published objects, of which
5,749 have usable nominal 6D states and tidal radii. It retains RV counts,
errors, dispersion, membership, ages and aliases. These are nominal matches,
with no covariance or cluster-evolution marginalization. A tidal radius is a
geometric association scale, not an astrometric error or a compact ejector size.
No match is never evidence that an ejector is dark.
