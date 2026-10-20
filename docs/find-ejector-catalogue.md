# `find_ejector`: continuous catalogue searches

Schema 2 removes the old 128-star and stored-time-grid search limits. It accepts
joint six-dimensional draws, integrates every supplied star, writes all
tracebacks to disk, and finds **every qualifying fixed-size group and its time
intervals in the piecewise-linear trajectories**. It searches encounters between
orbit snapshots, including encounters with no accepted grid point. Results are
not truncated to a ranked shortlist. A resource cap aborts the search.

This is a functional catalogue search, **not completion of the original all-survey
`find_ejector` objective or a demonstrated astronomical breakthrough**. Input
coverage, unknown radial velocities, population priors, Galactic dynamics,
source visibility, and distinguishing ejection from other shared motion remain
scientific requirements. A kinematic group is not an ejection discovery.

The [real-data campaign](../research/ejector_catalogue/README.md) includes the
actual web-repo Gaia field, archive enrichment, an uncapped all-sky selection,
raw data, provenance, and reproducible searches. Schema 1 remains supported.

## Prepare Gaia observations

`constellation ejector_prepare prepare.json` writes a joint-draw NPZ and an audit
JSON. No network request is made by this command.

```json
{
  "schema_version": 1,
  "input": "gaia_fast.fits.gz",
  "output": "catalogue.npz",
  "n_samples": 128,
  "seed": 20261019,
  "frame": {
    "galcen_distance_pc": 8122.0,
    "z_sun_pc": 20.8,
    "galcen_v_sun_kms": [12.9, 245.6, 7.78]
  },
  "priors": {
    "distance_length_pc": 1350.0,
    "missing_rv_mean_kms": 0.0,
    "missing_rv_sigma_kms": 300.0,
    "parallax_zero_point_mas": -0.017,
    "shared_zero_point_sigma_mas": 0.01
  }
}
```

These values are explicit exploratory assumptions, not universally appropriate
Gaia priors. The distance model is an exponentially decreasing space-density
prior, `p(r) ∝ r² exp(-r/L)`, multiplied by the parallax likelihood and inverted
numerically. It supports negative parallaxes. The domain is 0.1 pc through
max(100 kpc, 50 L); posterior mass at its boundaries is checked. The source
selection is **not** fitted as a population model.

The full five-parameter Gaia correlation matrix is retained. Conditional on
distance and calibration, angular position and proper motion are sampled from
the conditional multivariate Gaussian; right-ascension errors are tangent-plane
`alpha*cos(delta)` errors. The same zero-point draw is used across all stars in
each joint sample. This is an externally specified calibration uncertainty,
not a hierarchical inference of calibration from the selected catalogue.
The global offset does not replace source-dependent Gaia DR3 calibration.

A radial velocity is treated as usable when it has at least five transits,
0 < reported error <= 50 km/s and |RV| <= 700 km/s. Otherwise the explicit
**heliocentric** Gaussian prior is used and the ledger says `explicit_prior`.
That prior must be varied; it is not a measured velocity, halo-population model,
or a claim that the star's RV is zero. No missing astrometric error or correlation
is filled with zero: affected rows are excluded and their IDs/reasons retained.
The adapter requires epoch J2016.0. Gaia source IDs remain integers/strings,
never floating-point IDs. CSV/FITS/VOTable inputs are supported by Astropy;
VOTable archive results must have successful QUERY_STATUS and no OVERFLOW.

## Search and archive every traceback

```json
{
  "schema_version": 2,
  "epoch_label": "Gaia DR3 J2016.0",
  "catalogue": "catalogue.npz",
  "tracebacks_output": "tracebacks",
  "potential": {
    "components": [
      {"type": "plummer", "mass_msun": 5000000000, "scale_pc": 500},
      {"type": "miyamoto_nagai", "mass_msun": 68000000000, "a_pc": 3000, "b_pc": 280},
      {"type": "log_halo", "speed_kms": 180, "core_pc": 12000}
    ]
  },
  "search": {
    "lookback_max_myr": 10,
    "time_step_myr": 0.1,
    "radius_pc": 10,
    "group_size": 3,
    "min_support": 0.25,
    "max_step_myr": 0.01,
    "n_controls": 19,
    "seed": 20261019,
    "max_group_evaluations": 1000000,
    "max_pair_checks": 100000000
  }
}
```

Run `constellation find_ejector search.json -o result.json`. Paths resolve relative
to the manifest. Output destinations must be new. The NPZ contains:

- `states_pc_kms`: shape `(samples, stars, 6)`, positions in pc and velocities
  in km/s, in one Galactocentric Cartesian frame and common epoch.
- `source_ids`: unique string array, one per star.
- Optional `control_strata`: string array, one per star.

Joint draw index is preserved across stars. An upstream model can encode common
calibration or solar-frame uncertainty this way. Potential uncertainty requires
separate whole-catalogue runs; it is not averaged into a single orbit.

`tracebacks/states.npy` is a memory-mappable `(time, sample, star, 6)` archive.
`lookback_myr.npy` and `source_ids.npy` index it. `complete.json` is written only
when the observed search, numerical checks, and every requested control finish.
An interrupted/failed directory without this file is **incomplete**, even if its
preallocated trajectory file has the expected size. Disk usage is approximately
`8 * times * samples * stars * 6` bytes. Orbit/search memory is independent of
the number of time steps; dense pair lists and qualifying results still cost
memory. The integrator's per-batch resource bounds remain in effect.

Optional `counterparts` names another NPZ with matching joint draw count and
distinct IDs. Supplied counterparts are integrated to each candidate's peak
epoch, and conditional separation percentiles/proximity fractions are reported.
Absence of a supplied counterpart is **not evidence of a dark source**. This
interface does not establish a counterpart catalogue's depth or completeness.

## What the continuous search guarantees

Within one interpolation segment, a star follows `x(u)=a+u*b`, 0 <= u <= 1.
Its path lies inside a midpoint sphere with radius half its displacement. A
KD-tree returns a conservative superset of pairs whose spheres could approach
within 2R. Each pair is then tested analytically for closest approach. Only
pairs meeting the minimum number of joint draws form edges in the sparse graph.
Every requested-size clique is enumerated.

For a group, its arithmetic centroid is also linear in u. Each member's
squared centroid distance is a convex quadratic. Solving its radius inequality,
then intersecting across members, gives one closed interval per draw. Sweeping
all interval endpoints gives every interval with sufficient **simultaneous**
sample support. Pair encounters at unrelated times or support attained in
different draws at different times do not pass this final test.

The gate is necessary: a valid centroid-radius event implies every pair is
within 2R in the same qualifying draws. It therefore cannot discard a qualifying
group in exact arithmetic. The implementation uses float64, **not certified
interval arithmetic**; numerical boundary cases remain floating-point results.

Within each support region, the peak uses a convex quadratic compactness
criterion. Ranking is support plus a compactness tie-break smaller than one
sample of support, as in schema 1. All disjoint qualifying intervals are retained.
The dossier includes conditional origin quantiles, relative RMS velocity,
and whether the peak lies on a search boundary. A cold group or a peak at the
present epoch is not evidence of an explosive shared origin. Only the specified
group size (3--6) is enumerated; overlapping groups are not independent events.

## Numerical and statistical limits

Fine trajectories are used for the search. A separate coarse integration is
advanced over the full history without resetting onto the fine path. Maximum
coarse/fine position discrepancy and fine-orbit midpoint deviation from each
linear chord must each be <= 0.1R, including controls. Midpoint checking is a
**diagnostic**, not a rigorous bound on every exact orbit between checkpoints.
Repeat with smaller output and integration steps before interpreting candidates.

Velocity-cloud permutations within supplied strata rerun the entire group/time
search. The reported rank fraction is `(1 + control maxima >= observed score) /
(B+1)`. Its validity requires exchangeable velocity clouds under the chosen
null. Permutation breaks position/velocity covariance and does not reproduce
the Galactic selection function; this is a diagnostic conditional randomization,
**not an astrophysical false-discovery probability**. If several scenarios or
radii are inspected, an individual run's rank fraction does not correct that
additional tuning. No control means no significance value.

A finite draw count cannot certify an unsampled posterior. The real-data
screening run's small draw count must not be interpreted as calibrated
common-origin probability. Gaussian astrometry, the distance/RV assumptions,
source-dependent systematics, binary motion, nonstationary Galactic forces,
stellar ages, and source visibility all need further evidence for discovery.

### A control that retains radial-velocity selection

Schema 2 also accepts `search.control_method = "tangent_rotation"` and an explicit
`search.observer_state_pc_kms` six-vector in the same frame. For each star, it
rotates heliocentric transverse velocity around the line of sight while holding
position, radial velocity, and total heliocentric speed fixed in every draw.
One angle is shared across the star's cloud. This retains the radial expansion
pattern that whole-velocity permutations erase. It is useful for detecting
apparent origins near the observer caused by selecting extreme radial velocities.

`search_wide_control_rank_fraction` reports the rank against this specified
control. The legacy `search_wide_velocity_permutation_p` field is null for
`tangent_rotation`. Tangential-angle exchangeability is itself an assumption:
the Milky Way has anisotropic velocities and survey selection. Agreement or
disagreement between these controls is diagnostic, not a calibrated discovery
probability. The campaign also inspects spectral S/N and external identifiers.
A small formal Gaia RV uncertainty alone is not validation of an extreme RV.

Dossiers flag present-day pairs whose sampled separation is <1 pc and relative
speed <5 km/s. These are **close co-moving pairs**, not automatically bound
binaries. Counting two components of one physical system as two independent
ejection paths can make chance encounters look more informative than they are.

For known or hypothesized co-moving systems, the NPZ may include a
`rotation_groups` string array. Members with the same label receive the same
rotation angle. This retains their shared angular motion approximately (exactly
for identical sightlines) instead of destroying a binary's shared velocity in
every control. The grouping is an explicit conditional-null assumption. The
real-data campaign compares grouped and ungrouped controls for its close pair.

A completed schema-2 archive can also be rescanned at a smaller radius using
`python -m research.ejector_catalogue.tighten_radius ARCHIVE R OUTPUT.json`.
Any smaller-radius event must already belong to the complete parent search's
larger-radius group set. The helper therefore searches the union of those
members while retaining completeness for the **whole parent catalogue's same
sampled paths**. It does not reuse the larger-radius significance calculation,
and reports whether the original numerical diagnostics suffice at the new
radius. The original archive must remain unchanged.
