# Fit common origins and repeated stellar launches

`constellation ejector_fit manifest.json -o fit.json` finds **compatible fitted
encounters** in observed parallax, proper motion and radial velocity. It addresses
a weakness of a small Monte Carlo search: a physically compatible compact
encounter may occupy too little uncertainty volume to be sampled. A low sampled
support is not an exclusion of such an encounter.

This is a multistart constrained optimizer. It is not an exhaustive search,
global-optimum certificate, calibrated discovery test, or completed
`find_ejector` objective. Similar encounter-fitting methods already exist in
[Bhat, Irrgang & Heber (2022)](https://arxiv.org/abs/2204.01594).

## Input

A complete example is
[the historical Orion manifest](../research/ejector_fit/data/hip1997_triple.json).
The [real-data campaign](../research/ejector_fit/README.md) also fits one shared
cluster orbit to encounters at several different epochs.

Required top-level fields:

- `schema_version: 1`, `epoch_label`: all objects must use a common epoch.
- `frame`: explicit Galactocentric distance, solar height and velocity, as in
  the other stellar tools. Axes use the explicit Astropy orientation.
- `potential`: the supported smooth Galactic components; `components: []`
  selects free motion.
- `objects`: unique `id`, `ra_deg`, `dec_deg`, four-component `mean`, and
  positive-definite `covariance` for each object. Mean order is **parallax
  (mas), pmra including cos(dec) (mas/yr), pmdec (mas/yr), RV (km/s)**.
  Covariance has the corresponding product units. `notes` is optional.
- `encounters`: unique `id`, at least two distinct `members`, positive
  `lookback_bounds_myr`, and nonnegative `radius_pc`.
- `fit`: all settings below are required explicitly.

Angular positions are fixed. The four-observable covariance supports within-star
correlations, but not angular uncertainty, cross-star calibration covariance,
or uncertain frame/potential parameters. Missing RVs are not supported: supply
actual measurements and appropriate systemic velocities for binaries. An
invented Gaussian RV is not a measurement. Positive observed parallax above
the declared lower bound is required; this fitter is not a low-S/N distance
posterior sampler.

`radius_pc: 0` imposes equality of historical positions. A positive radius
requires every member to lie within that radius of its arithmetic centroid.
**For a two-object encounter, separation is at most twice this radius.** It is
not automatically the physical radius of an identified source. For instance,
use `radius_pc: 3.5` to require star–cluster separation at most 7 pc.

Objects appearing in multiple encounters have **one fitted observation vector**
and contribute their likelihood penalty only once. Each encounter has its own
lookback time. This permits repeated launches from a moving source without
letting pairwise fits silently choose contradictory source states. It is an
extension of the original simultaneous-convergence objective, not a replacement
definition of its completion.

## Method and settings

For object i, write fitted observables as `m_i + L_i z_i`, where `L_i` is the
Cholesky factor of its supplied covariance. Minimize `sum_i ||z_i||²` subject to
the encounter constraints and parameter bounds. This is a Gaussian
**profile likelihood** calculation in measured observables, with no distance
prior. The cost is not a posterior probability or false-association rate.

`starts_myr` is a matrix: one row per independent start, one column per encounter.
Each start begins at the observed mean, with the supplied ages. All start
outcomes are reported, including failures. Limits are 32 objects, 16 encounters,
and 128 starts; this command is intended for candidate follow-up.

`max_sigma` bounds each **whitened Cholesky coordinate**, not each marginal
measurement shift when covariances are correlated. `min_parallax_mas` bounds
the fitted true parallax. `max_iterations` caps each SLSQP solve. Active bounds
are reported by zero-based parameter index: four whitened parameters per object
in input order, then one age per encounter. A bound-active solution deserves
sensitivity testing, not an exclusion claim beyond that bound.

`max_step_myr` sets leapfrog resolution. Each encounter uses a fixed number of
steps across its allowed age range, keeping the constraint function smooth as
age changes. The Jacobian advances finite-difference perturbations in a batch.
`constraint_tolerance_pc` must be met by both the fitted encounter residual and
the difference from an integration with twice as many steps. Both diagnostics
are reported. These are numerical checks, not rigorous error enclosures.

## Interpretation

`feasible_local_fit` means at least one optimizer run converged and passed
numerical checks; `best_fit` is the lowest cost among those runs. It need not be
the global minimum. `no_converged_feasible_fit` is **inconclusive**: the optimizer
could have failed or the allowed domain could be too small. It does not prove
that no physical encounter exists. An independently integrated trajectory is
used to validate the committed research results.

A low cost demonstrates measurement compatibility under the chosen assumptions.
Propagation treats objects as test particles in the supplied smooth Galactic
potential; it does not simulate binary scattering or forces inside a cluster.
It does not distinguish ejection from passage, identify an ejection mechanism,
account for selection among many hypotheses, or establish that an unseen source
is dark. We deliberately return `discovery_probability: null` and
`full_objective_complete: false`.
