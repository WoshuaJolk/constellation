# From catalogue to working research tools

Assessment of the 23 catalogue proposals after implementing twelve bounded tools and a public ZTF adapter. These are engineering judgments, not proofs of feasibility or a literature survey establishing novelty.

| Idea | Assessment and next concrete step |
| --- | --- |
| `map_cover` | ZTF public queries, selected cutout preparation, local FITS coverage, orbit propagation, light time, observatory parallax, masks and family fractions implemented. Add automatic orbit-enclosing query regions, further surveys and calibrated efficiency curves. |
| `force_track` | Gaussian and native-pixel empirical PSFs, linear trails and ZTF prepared inputs implemented. Add spatial PSF variation, oversampling, image stamps and independent astrometric checks. |
| `weigh_stamp` | Gaussian/measured PSFs, empirical artifact templates and finite-class evidence implemented. Still needs held-out real artifact validation, spatial PSFs, blends and calibrated acceptance. |
| `judge_line` | Reproducible jackknife, temporal, stationary/reversed-track and residual diagnostics implemented. Still needs orbit refits and joint-decision calibration on held-out controls; correlated tests cannot simply have their p-values multiplied. |
| `learn_null` | Joint orbit/brightness marginalization, Beta calibration uncertainty, shared-efficiency null integration, next-exposure ranking and coverage planning exports implemented. Still needs orbit fitting/ranging and independently validated real survey calibration. |
| `link_world` | A bounded linker is feasible using existing methods. Global data access, hyperbolic completeness, continuous operation, and chance-link calibration need separate substantial work. |
| `dig_floor` | Finite-trajectory joint likelihood and injection recovery on real backgrounds implemented. Add realistic orbit banks, sparse indexing, acceleration, source shot noise, and population-calibrated vetting. |
| `map_complete` | Exact enumeration for a finite population, exposure probabilities and cadence rules implemented, including shared-exposure survey unions. Needs real calibrated selection inputs, cross-exposure dependence and actual linkage acceptance. |
| `inject_all` | Finite-population selection Monte Carlo, Monte Carlo error intervals and conditional Poisson count likelihoods implemented. Full pixel/dynamics population simulation remains larger work. |
| `fuse_astrom` | Start with a bounded crossmatched field, common reference frame, epoch-dependent systematics, and covariance model; an all-sky catalogue is a data-intensive project. |
| `find_ejector` | Bounded 6D uncertainty propagation, Galactic dynamics, complete fixed-size group/time search, supplied-counterpart comparison, and velocity-permutation controls implemented. Missing velocities, selection, potential uncertainty, and causal interpretation remain substantial work. |
| `map_glow` | Research prototype possible on a small image set. Demonstrate separation from static sky, flat-field errors, and correlated noise before mapping power to population mass. |
| `weigh_tno` | First calculate expected perturbation amplitudes and parameter identifiability for actual astrometric baselines. Precision and degeneracies may prevent useful mass estimates. |
| `map_sedna` | Wrap a specified formation population and survey selection simulator; do not invent a universal inner-Oort prior. |
| `read_belt` | An emulator plus likelihood over specified migration simulations is plausible. Selection effects and non-unique histories must be retained in the posterior. |
| `read_oort` | Combine original comet orbits with stellar encounter posteriors and a dynamical response model. Direction matching alone cannot establish causation. |
| `twin_system` | Couple explicit formation simulations with survey simulators. An end-to-end fit remains a large, model-dependent inference project. |
| `run_hunt` | Orchestration becomes useful after inputs, calibrated acceptance rules, and orbit refitting are available. Do not automate acceptance of an unvalidated high-S/N peak. |
| `fold_kepler` | Restricted finite-bank transform implemented: FFT sufficient statistics, shared orbit-cell subpaths, Bernstein bounds and precise refinement. A 64,000-orbit benchmark evaluates about 3.6× faster after compilation, with identical direct-search results. Compilation makes a single-use run slower. General continuous-orbit transforms, formally rounded bounds and real-survey validation remain open. |
| `match_all` | Exact integer set packing for up to 128 supplied candidates implemented with independently checkable branch/clique certificates and honest bounded status. Physical candidate completeness, score calibration and archive-scale guarantees remain open. |
| `prove_floor` | Finite-family Gaussian power bounds, a necessary-flux lower bound, and a calibrated prior-optimal mixture-LR statistic implemented, including correlated noise and projected backgrounds. Continuous orbit families, unknown noise/artifact laws, and global archive limits remain open. |
| `fit_sky` | Begin with a small patch and a tractable joint generative model. An all-photon, all-instrument posterior is beyond this implementation. |
| `invert_chaos` | Explore posterior identifiability over finite times and stated priors. Lost historical information cannot be restored by a generic inverse solver. |

## Recommended next increment

Choose one public, calibrated exposure set and a known moving source with reliable external ephemerides. Ingest variance/mask products, establish photometric units and PSF quality, and run position-held-fixed recovery. Inject sources over a grid of brightness and rate, repeat the real candidate-selection procedure on controls, then feed those measured efficiencies into `learn_null`. That produces a defensible survey-specific result before expanding to additional archives.

## Current execution order

The first pass through priorities 1–9 is implemented with the scopes above:
`map_cover`, `force_track`, `judge_line`, `dig_floor`, `learn_null`, `weigh_stamp`,
`map_complete`, `inject_all`, `match_all`. The live ZTF experiment, exact selection
examples, and assignment certificates are reproducible from `examples/`.

The remaining queue, in the requested order, is:

10. `link_world`: bounded physical orbit-candidate generation and linkage controls.
11. `fold_kepler`: initial restricted transform implemented at the user's request; extend reuse, physical models and verified arithmetic (see [guide](fold-kepler.md)).
12. `prove_floor`: continuous-template and uncertain-noise extensions.
13. `run_hunt`: orchestrate validated searches and audit records.
14. `fuse_astrom`: one-field multi-epoch astrometry with covariance/systematics.
15. `find_ejector`: real catalogues and uncertain Galactic potentials.
16. `map_sedna`: explicit formation populations through calibrated selection.
17. `read_belt`: migration-simulation likelihoods and degeneracy checks.
18. `map_glow`: controlled motion-compensated power experiments.
19. `read_oort`: joint comet/stellar encounter response models.
20. `weigh_tno`: perturbation identifiability on real astrometric baselines.
21. `twin_system`: combine independently validated formation and selection modules.
22. `fit_sky`: a small, tractable joint generative image model.
23. `invert_chaos`: quantify historical identifiability before attempting reconstruction.
