# `find_ejector`: uncertain stellar trajectories and common-origin candidates

Schema 2 now supports larger catalogues, continuous interpolation encounters, Gaia input preparation and complete traceback archives. See [the catalogue-search guide](find-ejector-catalogue.md) and [real-data campaign](../research/ejector_catalogue/README.md). The documentation below describes the retained schema-1 interface.

The tool searches for simultaneous spatial convergence of several stars in a **supplied finite catalogue**. It propagates complete 6D uncertainty under an explicit, smooth Galactic potential, searches every eligible group of a chosen size at every grid time, and compares each candidate with supplied source/cluster trajectories. It also repeats the entire search on velocity-permuted catalogues to measure how unusual the result is under that particular null.

A convergence is kinematic evidence to investigate. It can reflect a common birthplace, chance coincidence, a poor model, or other causes. It does not establish an ejection mechanism, identify an intermediate-mass black hole, or prove a source is dark.

## Input example

Generate a complete example with `magictools ejector_demo ejector-demo-output`. A minimal schema is:

```json
{
  "schema_version": 1,
  "epoch_label": "All states at the same declared reference epoch",
  "potential": {
    "components": [
      {"type": "plummer", "mass_msun": 5000000000, "scale_pc": 300},
      {"type": "miyamoto_nagai", "mass_msun": 68000000000, "a_pc": 3000, "b_pc": 280},
      {"type": "log_halo", "speed_kms": 220, "core_pc": 12000}
    ]
  },
  "stars": [
    {"id": "a", "state_pc_kms": [-7800, 200, 40, 100, 220, 0]},
    {"id": "b", "state_pc_kms": [-8000, 400, 40, 0, 320, 0]},
    {"id": "c", "state_pc_kms": [-8200, 200, 40, -100, 220, 0]}
  ],
  "search": {
    "lookback_max_myr": 5,
    "time_step_myr": 0.01,
    "radius_pc": 5,
    "group_size": 3,
    "min_support": 0.2,
    "n_samples": 128,
    "max_step_myr": 0.01,
    "n_controls": 63,
    "seed": 2026
  }
}
```

These potential values and stellar states are illustrative, not a fitted Milky Way model or observed catalogue. `potential.components: []` explicitly selects ballistic/free motion. No gravitational model is silently inferred.

### State and uncertainty formats

`state_pc_kms` is `[x,y,z,vx,vy,vz]` in Galactocentric Cartesian coordinates, parsecs and km/s, with Astropy's axis convention. All states must refer to the same reference epoch; `epoch_label` records that convention but does not perform epoch conversion. Internal dynamics use pc and Myr, with an explicit km/s conversion and Astropy's gravitational constant. The code does not decide which stars are fast enough; supply the scientifically selected catalogue.

Each star uses exactly one of these forms:

- `state_pc_kms`, optionally with a symmetric positive-semidefinite 6×6 `covariance` in the corresponding mixed units. The sampler retains position–velocity correlations. Omitted covariance means an exact state, not an estimated measurement uncertainty.
- `samples_pc_kms`: equally weighted complete six-component posterior samples. Rows are resampled intact, retaining within-star dependence.
- `icrs_samples`: equally weighted posterior rows `[ra_deg, dec_deg, distance_pc, pm_ra_cosdec_masyr, pm_dec_masyr, radial_velocity_kms]`. These require explicit `galactocentric_frame` parameters:

```json
"galactocentric_frame": {
  "galcen_distance_pc": 8122,
  "z_sun_pc": 20.8,
  "galcen_v_sun_kms": [12.9, 245.6, 7.78]
}
```

The conversion uses the [Astropy Galactocentric frame](https://docs.astropy.org/en/stable/coordinates/galactocentric.html), explicitly fixing the center orientation to ICRS RA 266.4051 degrees, Dec −28.936175 degrees, and roll zero. Solar frame values above are examples, not uncertainty-marginalized constants. Position/velocity samples are in the same convention whether supplied directly or converted.

Distance must be positive posterior distance, not an unexamined inverse of noisy parallax. A missing radial velocity needs an externally justified posterior; it is **not** filled with zero. The input format assumes independence between stars. Shared astrometric systematics, correlated cluster members, solar-frame uncertainty, and Galactic-potential uncertainty are not sampled automatically.

### Potential and integrator

Supported components can be summed in any number:

```
Plummer:       Phi = -G M / sqrt(r^2 + b^2)
Miyamoto-Nagai:Phi = -G M / sqrt(R^2 + (a + sqrt(z^2 + b^2))^2)
Log halo:      Phi = v_h^2 / 2 * log(1 + r^2 / r_c^2).
```

The accelerations are analytic negative gradients. These are standard smooth components, also represented in established dynamics libraries such as [galpy](https://docs.galpy.org/en/latest/reference/potential.html). The logarithmic halo is not a finite-mass outer-Galaxy model. There is no time-dependent bar, spiral structure, substructure, LMC, mutual stellar gravity, dynamical friction, close encounter, or ejection physics. Choose and validate the potential for the particular distance/time regime.

Trajectories use vectorized kick–drift–kick leapfrog, stepping backward from the declared present. Every output time is an integration endpoint. The observed catalogue is integrated twice, with the second maximum step halved. If the maximum position difference exceeds 10% of the convergence radius, the tool fails and asks for a smaller step. It also reports energy drift. This convergence check measures numerical error within the model; it does not establish that the potential is physically correct. Counterpart trajectories use the smaller step.

### Search and its finite guarantee

For every joint posterior draw at every specified time, an event occurs when **every member is within `radius_pc` of the group's arithmetic centroid**. `support` is the fraction of draws satisfying that geometric event. It is not the posterior probability of a common ejection mechanism.

All groups of exactly `group_size` (3–6) are eligible. Pair screening uses the necessary condition that every member pair is separated by at most twice the radius in at least `min_support` of the same draws. If a group meets the full event criterion in the required fraction of draws, it necessarily passes every pair screen. The code then enumerates fixed-size cliques and tests the full simultaneous event; a chain of pair encounters is not accepted as a group.

Thus the screening is complete **for qualifying groups in this finite catalogue, sampled posterior, and time grid**. It makes no claim about unsampled uncertainties, unobserved stars, or convergence between grid times. A hard evaluation cap raises an error rather than silently returning an incomplete search.

Each group retains its highest-scoring time. The score is

```
support + exp(-ensemble_mean_squared_radius / (2 radius^2)) / (2 (n_samples + 1)).
```

The second term breaks equal-support ties by compactness and is smaller than one sample's support increment. It is a ranking statistic, not a probability. All control searches use exactly the same statistic and settings. Overlapping reported groups are not automatically merged into a larger common-origin claim.

### Search-wide controls

Each control permutes complete velocity clouds among stars, while leaving their position clouds fixed, then reruns the integration and **entire group/time search**. Optional per-star `control_stratum` strings restrict permutations to declared exchangeable subsets. If every subset is a singleton and controls were requested, the tool rejects the meaningless control experiment. Permutations may include fixed points and repeats, as required by the randomization scheme.

For an observed candidate score `s` and `B` control search maxima, the reported value is

```
(1 + count(control_max >= s)) / (B + 1).
```

This includes selection across all groups and times **in this run**, rather than comparing a handpicked group with a weaker control test. It remains conditional on velocity-cloud exchangeability within the specified strata. Real velocity fields, survey selection, and position–velocity covariance can violate that null. Scrambling intentionally breaks the original position–velocity association, including its measurement correlation. This is a control-model limitation, not automatically an astrophysical false-positive probability. Tuning the radius, group size, sample selection, potential, or time range across multiple runs adds choices not included here.

At 63 controls, the minimum reported p-value is 1/64; zero exceedances do not imply zero chance. Use more controls for finer resolution. With `n_controls: 0`, the report explicitly provides no significance estimate.

### Counterparts and output dossiers

Optional `counterparts` use the same state/sample formats, separate IDs, and an optional `label`. These could be known clusters or possible source objects with supplied trajectories. They are not searched as stellar group members. At the selected event time, the tool computes their distance to the sampled group centroid, conditional on group convergence, and the fraction inside `association_radius_pc` (defaults to the convergence radius). These are geometric associations; an absent or distant catalogue counterpart is not evidence of a dark source.

Each candidate reports member IDs, lookback time, support and a pointwise Monte Carlo interval, origin-coordinate 16/50/84 percentiles conditional on convergence, member velocity summaries, the support time profile, and the fraction converging at any grid time. It also reports percentiles of each draw's minimum-RMS epoch; these are kinematic summaries, **not** a common-origin age posterior derived from an ejection likelihood. The pointwise support interval does not correct selection of the best group/time; the permutation control serves a separate purpose.

Search settings support 3–128 stars, 16–1,024 samples, at most 5,001 grid times over up to 100 Myr, and 0–999 controls. Trajectory storage is capped at 30 million scalar values; integration at 100,000 steps. These are software limits, not accuracy guarantees. `max_group_evaluations` defaults to 200,000 per search, and `max_results` defaults to 20. The complete search is performed before the reporting limit is applied.

The report records the actual time grid, settings, potential, provenance, control maxima, and numerical diagnostics. `time_grid_may_skip_convergences` flags when a conservative maximum-speed displacement over one time cell exceeds the radius. A false flag is not a continuous-time completeness proof: test finer grids, especially for narrow or boundary peaks.

## Validation

Tests compare each force with an independent finite-difference potential gradient; compare dynamics with high-accuracy DOP853; check free motion, energy conservation, and reversal; preserve correlated uncertainty samples; and check ICRS conversions. Search tests compare pair screening against exhaustive group enumeration, reject pairwise-only configurations, recover known members/age/origin, verify control counting and reproducibility, and reject silent truncation or ineffective control strata.

`ejector_demo` uses a separate finer-step forward integration to create four synthetic ejecta 2.4 Myr ago, adds four field stars and correlated errors, and searches their unknown membership. It recovers the injected group and compares it with an independently propagated synthetic cluster. This validates an internally specified experiment; real-data causal interpretation remains substantial scientific work.
