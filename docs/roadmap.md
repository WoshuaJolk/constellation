# From catalogue to working research tools

Assessment of the 23 catalogue proposals after implementing five local-data tools. These are engineering judgments, not proofs of feasibility or a literature survey establishing novelty.

| Idea | Assessment and next concrete step |
| --- | --- |
| `map_cover` | Local FITS coverage, numerical orbit propagation, light time, masks, family fractions, and Gaussian sensitivity implemented. Add survey indexing, real efficiency curves, and observatory-state adapters. |
| `force_track` | Core and linear-trail integration implemented. Add empirical PSFs, image stamps, instrument adapters, and independent astrometric checks. |
| `weigh_stamp` | Seven explicit model classes and prior-marginalized evidence implemented. Validate against real artifacts and injections, expand instrument models, and add multi-epoch tests. A complete artifact taxonomy is not available here. |
| `judge_line` | Package reproducible diagnostic tests, but calibrate their joint decision on held-out controls; correlated tests cannot simply have their p-values multiplied. |
| `learn_null` | Discrete inference, next-exposure ranking, and `map_cover` planning exports implemented. Add orbit/brightness sampling and calibrated real-survey selection functions. |
| `link_world` | A bounded linker is feasible using existing methods. Global data access, hyperbolic completeness, continuous operation, and chance-link calibration need separate substantial work. |
| `dig_floor` | Finite-trajectory joint likelihood implemented. Add realistic orbit banks, sparse indexing, acceleration, and injection-calibrated candidate vetting. |
| `map_complete` | Start with one survey's actual pointings and full discovery rules, including required linked nights. A union of surveys requires overlaps and dependence, not addition of completeness fractions. |
| `inject_all` | Build on validated per-survey simulators and use population likelihoods. The current seeded demo is only an implementation test, not this population experiment. |
| `fuse_astrom` | Start with a bounded crossmatched field, common reference frame, epoch-dependent systematics, and covariance model; an all-sky catalogue is a data-intensive project. |
| `find_ejector` | A bounded covariance-aware traceback search is implementable with supplied astrometry, radial velocities, and Galactic potential. Missing velocities and encounter significance require careful modeling. |
| `map_glow` | Research prototype possible on a small image set. Demonstrate separation from static sky, flat-field errors, and correlated noise before mapping power to population mass. |
| `weigh_tno` | First calculate expected perturbation amplitudes and parameter identifiability for actual astrometric baselines. Precision and degeneracies may prevent useful mass estimates. |
| `map_sedna` | Wrap a specified formation population and survey selection simulator; do not invent a universal inner-Oort prior. |
| `read_belt` | An emulator plus likelihood over specified migration simulations is plausible. Selection effects and non-unique histories must be retained in the posterior. |
| `read_oort` | Combine original comet orbits with stellar encounter posteriors and a dynamical response model. Direction matching alone cannot establish causation. |
| `twin_system` | Couple explicit formation simulations with survey simulators. An end-to-end fit remains a large, model-dependent inference project. |
| `run_hunt` | Orchestration becomes useful after inputs, calibrated acceptance rules, and orbit refitting are available. Do not automate acceptance of an unvalidated high-S/N peak. |
| `fold_kepler` | A general sub-trajectory-cost transform is a mathematical research problem. The current implementation makes no such complexity claim. |
| `match_all` | Small finite assignment problems can be solved exactly, but global physical completeness/false-link guarantees require explicit assumptions and new proofs. |
| `prove_floor` | Fixed-template Gaussian variance is available from the implemented information calculation. Unknown trajectories, nuisance parameters, artifacts, and multiple testing prevent calling it a universal detection bound. |
| `fit_sky` | Begin with a small patch and a tractable joint generative model. An all-photon, all-instrument posterior is beyond this implementation. |
| `invert_chaos` | Explore posterior identifiability over finite times and stated priors. Lost historical information cannot be restored by a generic inverse solver. |

## Recommended next increment

Choose one public, calibrated exposure set and a known moving source with reliable external ephemerides. Ingest variance/mask products, establish photometric units and PSF quality, and run position-held-fixed recovery. Inject sources over a grid of brightness and rate, repeat the real candidate-selection procedure on controls, then feed those measured efficiencies into `learn_null`. That produces a defensible survey-specific result before expanding to additional archives.
