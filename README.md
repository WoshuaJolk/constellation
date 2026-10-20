# Constellation astronomy

**For agents:** [`constellation`](docs/agent-cli.md) offers JSON discovery,
schemas, runnable examples, input validation, execution and checkpoint inspection
for Kepler and FitSky. Install with `pip install '.[research,agent]'`, then run
`constellation list`. Scientific limitations remain explicit in every run.

Working implementations of twelve ideas in the [Magic tools catalogue](https://magictools-theta.vercel.app/). Version 0.6 adds **`fold_kepler`: shared orbit sums with bounded screening and precise candidate refinement**. A synthetic 64,000-orbit benchmark gives about 3.6× faster evaluation after compilation, with identical detections/top candidates to direct evaluation. Compilation makes a single-use run slower; the benefit is repeated searches with a reusable bank. See the [derivation, benchmark and limits](docs/fold-kepler.md). Earlier versions provide real ZTF inputs, measured PSFs, vetting, injections, null inference, finite survey selection, and verifiable finite assignments. Each has an explicit scope; all-survey completeness and validated discovery probabilities remain future work.

The subsequent [full-objective research attempts](research/kepler_transform/README.md)
test global compression, hierarchical phase factorization, and rotational
harmonics with a directional light-time extension. **The full objective was
not achieved.** Derivations, failures and reproducible experiments are preserved
separately from the production tools; no breakthrough is claimed.

The [second Kepler construction campaign](research/kepler_transform/construction_results/README.md)
adds a localized phase-corrected NUFFT and a six-dimensional apparent-position
atlas with validated continuous geometry bounds. A million-query bounded test
runs **2.10× faster including geometry certification** than exact propagation
plus per-frame NUFFTs. A stronger comparison using the same approximate geometry
finds accumulation gains at longer cadences, but a loss at eight exposures.
That campaign's broad-domain accuracy and resource tests failed; it is a research backend,
**not completion of the general all-orbit transform**.

The subsequent [observer-chart campaign](research/kepler_transform/chart_results/README.md)
adds adaptive continuous coverage, a higher-order construction that resolves
the widest declared stress domain, a finite-detector likelihood adapter tested
on real ZTF pixels, and an installed experimental command:
`magictools-fold-chart`. Its benchmarks include a compiled direct-orbit solver
and an optimized same-geometry comparison. See the
[library, derivation and manifest guide](research/kepler_transform/chart_construction.md).
Full fast numerical certification and economical general outer-coordinate
synthesis remain open; no new mathematical breakthrough is claimed.

The [mathematics-first global campaign](research/kepler_transform/global_results/README.md)
constructs a continuous six-coordinate response field before choosing query
orbits. It derives the hierarchy's error and full construction cost, and
certifies its exact-arithmetic kernel error below `1e-8` across all three
declared physical domains using complex-time interval bounds. It also proves
why a shared regularized-clock FFT shortcut and an unrestricted inverse fail.
The representation still has a six-dimensional resolution cost: **this is not
a sublinear-in-trajectories breakthrough or completion of the catalogue goal**.

The [full-objective `prove_floor` research](research/detection_limit/README.md)
adds continuous-family detection certificates, least-favorable-prior searches,
and interval orbital image models. It closes the power gap below one percentage
point in explicit controls and a narrow multimodal orbit family. Broad continuous
orbit coverage still fails the target; **the full detection-limit objective is
not achieved**. The research includes derivations of the conditional bounds,
reproducible failures, and explicit Monte Carlo confidence/arithmetic limits.

The [real-data `find_ejector` campaign](research/ejector_catalogue/README.md)
adds a sparse continuous-time search, correlated Gaia likelihood sampling,
explicit missing-RV priors, shared calibration draws, and disk-backed tracebacks.
It searches an uncapped **38,889-source all-sky Gaia DR3 selection** and audits
actual web-repo data. See the campaign for results and remaining full-objective
gaps; a kinematic convergence is not an ejection discovery.

The [observed-system follow-up](research/ejector_fit/README.md) adds
[`ejector_fit`](docs/ejector-fit.md): constrained fits in measured astrometry and
radial velocity, including several launch times from one shared moving source.
It tests historical and updated Orion measurements and explicitly checks why
a low Monte Carlo hit rate cannot exclude a compatible common origin.

The [catalogue decision campaign](research/ejector_search/README.md) adds
[`ejector_search` and `ejector_bound`](docs/ejector-search.md): all-group fits,
whole-search controls, a 7,167-object published source catalogue, and validated
continuous-time interval decisions. It excludes all 816 triples in a real
18-star sample within a declared cost/model domain, certifies the historical
Orion encounter, and identifies the sole injected group and its source among
1,330 recovery hypotheses. These are completed bounded decision problems;
the original all-survey discovery objective remains open.

| Tool | Available now | Still needed for the catalogue's full ambition |
| --- | --- | --- |
| `fold_kepler` | FFT likelihood maps, shared partial orbit sums, Bernstein score bounds, exact-position refinement and direct-baseline validation. Elliptic Kepler/light-time generator or supplied pixel trajectories. | A general continuous Kepler transform, profitable one-shot compilation, physical PSF/trailing/nuisance extensions and real-survey validation. No new mathematical breakthrough claimed. |
| `prove_floor` | Analytic upper bounds on any detector's prior-average power for a finite Gaussian template bank, a necessary-flux lower bound, and a prior-optimal mixture likelihood-ratio statistic with conservative calibration. Supports correlated noise and background projection. | Unknown/continuous orbit families, realistic nuisance and artifact distributions, survey-wide limits. Uses established theory; does not claim a new universal theorem. |
| `find_ejector` | Sparse continuous searches, correlated Gaia inputs, explicit RV/distance priors, traceback archives, all-group measurement fits, whole-search controls, published cluster matching, and Arb continuous-flow exclusion/existence certificates in a bounded fixed model. | All-survey completeness, population-calibrated missing velocities and selection, uncertainty in Galactic histories/frame, complete source visibility and physical confirmation of an ejector. |
| `map_cover` | ZTF public exposure queries and selected cutouts; orbit propagation, light time, observatory parallax, FITS coverage and masks, family fractions, trails and estimated S/N. | Complete orbit-family archive indexing, orbit fitting/ranging, calibrated depth curves and further surveys. |
| `force_track` | Gaussian or native-pixel empirical PSF fits with optional trails, signed flux and uncertainty, background, residuals and usable template fraction; ZTF input preparation. | Spatial/oversampled PSFs, WCS refinement, instrument-specific uncertainty and color calibration. |
| `judge_line` | Leave-one-out and temporal-half fits, residual and constant-flux checks, stationary/reversed controls on common exposures, explicit diagnostic flags. | Orbit refits, astrometric confirmation, held-out artifact calibration and a validated combined acceptance rule. |
| `weigh_stamp` | Measured-PSF moving/stationary/dipole models, additional empirical artifact templates, and finite-class Bayesian comparisons on common pixels. | Held-out real-artifact validation, spatial PSF models, blends and calibrated acceptance probabilities. |
| `dig_floor` | Joint likelihood over a finite trajectory bank with empirical or Gaussian PSFs and known scales; Gaussian trial correction; `inject_tracks` reruns the search on signal injections into real or synthetic backgrounds. | Orbit-bank generation, GPUs, correlated-noise models, full-archive indexing and population-calibrated selection. |
| `learn_null` | Bayesian updates and exposure ranking with orbit/brightness marginalization, finite-calibration uncertainty, and correct integration of reused efficiencies. | Orbital ranging, independently calibrated real-survey selection and continuous population priors. |
| `map_complete` | Exact finite-population cadence selection and survey-union probabilities, retaining shared-exposure dependence. | Validated detection probabilities, correlated exposure models, actual orbit-linking acceptance and complete survey histories. |
| `inject_all` | Reproducible population selection Monte Carlo, error bounds and conditional Poisson count likelihoods. | Full pixel/dynamics population injections, selection-calibration uncertainty and orbital-property likelihoods. |
| `match_all` | Exact integer weighted assignment of supplied candidates with checkable branch/clique certificates; honest bounds when work is capped. | Candidate generation, physical completeness and false-link calibration at archive scale. |

These are Python library functions and command-line tools. Inference runs offline; the optional ZTF adapter and Gaia campaign fetcher download public data without API keys. Local images are never uploaded. Archive requests send only the supplied sky region, dates, and selected product identifiers.

## Install and try

Python 3.10 or newer:

```bash
git clone https://github.com/WoshuaJolk/magictools.git
cd magictools
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
magictools demo demo-output
magictools orbit_demo orbit-demo-output
magictools floor_demo floor-demo-output
magictools ejector_demo ejector-demo-output
magictools fold_demo fold-demo-output
python -m pytest -q
# Optional live archive experiment; downloads real ZTF images:
python examples/ztf_workflow.py new-ztf-output
```

The original `demo` creates 16 FITS images with different seeing, noise, photometric scales, and WCS pointings, plus predictions for an injected moving source and two control trajectories. The photometry, joint detection, and non-detection tools run, and their inputs, outputs, injection truth, and summary are saved. A fixed seed makes the experiment reproducible; an existing nonempty directory is never overwritten.

For the default seed `2026`, the injected reference flux is 85, the recovered flux is about **100.08 ± 10.42**, the strongest single-frame S/N is **4.22**, and the joint S/N is **9.60**. The correct trajectory ranks first. This is a synthetic recovery test, not an observation of a real object or a survey completeness measurement.

The new `orbit_demo` starts from a JPL Horizons Ceres state, creates four orbit hypotheses and eight **synthetic** images with a detector gap, exports coverage and planning inputs, and recovers the injected trajectory at **10.54σ**, with flux **96.77 ± 9.18** for an injection of 100. Separate source and cosmic-ray controls prefer the expected image models. The example is not an observation of Ceres.

`floor_demo` uses four trajectory templates in six frames with correlated noise and unknown frame backgrounds. At a 1% false-alarm limit, the faint-source experiment has an **analytic upper bound of 17.3% average detection power**; the implemented detector achieves about **13.3%**. The brighter experiment reaches about **98.6%**. These are model-conditional statements, not survey limits. See [the derivation and assumptions](docs/prove-floor.md).

`ejector_demo` hides four synthetic stars with a shared origin among four field stars, then searches without knowing the member IDs. It recovers the injected **2.4 Myr** convergence and compares it with a moving synthetic cluster. It includes correlated phase-space errors, 63 full-search controls, and a smaller-step integration check. Its control p-value is conditional on velocity exchangeability and has resolution 1/64. See [the stellar-search guide](docs/find-ejector.md).

Run the tools separately:

```bash
magictools force_track demo-output/images.json -o forced.json
magictools dig_floor demo-output/images.json --alpha 0.01 -o joint.json
magictools learn_null demo-output/nulls.json -o posterior.json

magictools map_cover orbit-demo-output/orbits.json -o coverage.json \
  --photometry-output tracks.json --planning-output plan.json
magictools dig_floor tracks.json -o orbit-search.json
magictools learn_null plan.json -o next-exposure.json
magictools weigh_stamp orbit-demo-output/moving-source-input.json -o stamp-evidence.json
magictools prove_floor floor-demo-output/faint-input.json -o detection-bound.json
magictools find_ejector ejector-demo-output/input.json -o origins.json
magictools learn_null examples/null-calibration.json -o calibrated-nulls.json
magictools map_complete examples/selection.json -o exact-selection.json
magictools inject_all examples/selection.json -o simulated-selection.json
magictools match_all examples/matching.json -o assignments.json
magictools verify_match assignments.json -o verified.json
magictools fold_kepler fold-demo-output/input.json -o folded-search.json
```

Omit `-o` for JSON on stdout. Output files must be new. Errors return exit code 2. `python -m magictools` is equivalent to `magictools`. Reports contain the software versions and SHA-256 hashes of every input file, including the manifest.

## Using your own data

See [input and output reference](docs/formats.md) for complete manifests. Supply calibrated images, **variance** (not standard deviation or inverse variance), the Gaussian PSF sigma in pixels, and positions at the actual exposure times. Positions may be zero-based pixel centers or ICRS RA/Dec in degrees. Paths are relative to the manifest, so commands work from any directory.

`prove_floor` instead accepts pixel templates and a known covariance, or their noise-weighted Gram matrix. `find_ejector` accepts present-day Galactocentric 6D states/covariances or complete posterior samples, including an optional ICRS conversion. Missing radial velocities and uncertain distances require supplied posteriors; neither is silently invented.

Alternatively, provide barycentric Cartesian orbit samples in **ICRS axes, AU and AU/day, at an explicit MJD TDB epoch**, and let `map_cover` generate the positions. The [orbit coverage and stamp comparison guide](docs/ambitious-tools.md) specifies the new formats, observer conventions, model priors, and limitations. The dynamics default uses Astropy's offline approximate ephemerides; optionally supply a local JPL SPK kernel after installing `.[ephemeris]`.

`dig_floor` assumes a single reference flux with known positive per-exposure `flux_scale`: predicted image flux = reference flux × scale. Match filters or supply a defensible color model. Exposure time, throughput, magnitude zero points, and unit conversion are **not** inferred from headers. A variable source needs a different model.

Use injection-measured `detection_probability` in `learn_null` when available. That probability must include usable coverage and the actual detection procedure. For the Gaussian option, `coverage` means the probability the hypothesis is on usable detector area; it is **not** `force_track`'s `psf_fraction`. A non-detection is an observed failure to pass a specified detection test. A missing image, invalid fit, or off-chip prediction is not itself a measured upper limit.

## Scientific scope

- The likelihood assumes a correct Gaussian or supplied empirical PSF, optionally integrated over a constant-rate linear trail; a constant background per stamp; known, independent Gaussian pixel noise; and independent exposures. ZTF preparation supplies an estimated noise model, so its Gaussian p-values need empirical calibration. Shared difference-image templates and resampling can violate independence. Correlated noise, blends, curved trails, PSF mismatch, and WCS errors need additional modeling.
- Forced positions and the orbit family must be specified independently of the fitted pixel noise. `map_cover` propagates supplied states; it does not fit them to detections, search every possible orbit, or account for adaptive searches performed outside this run. It defaults to a geocentric observer; supply an observatory/spacecraft barycentric state when parallax matters.
- `weigh_stamp` reports model-conditional posterior weights, Bayes factors, and profile likelihood ratios. These depend on the stated template bank and amplitude/model priors; they are not calibrated probabilities that an object is real. A zero-length trail is explicitly indistinguishable from the stationary-point template in a single stamp.
- Each per-frame flux remains signed. The joint fit includes negative values and reports a signed S/N. The one-sided Gaussian p-value tests a positive source. Bonferroni uses **every supplied trajectory**, including candidates with no usable data; it controls only the stated Gaussian null model and does not establish an astrophysical discovery.
- The reported `five_sigma_flux_sensitivity` is five times the fitted flux uncertainty for a fixed trajectory. It is neither an injection-derived survey limit nor a posterior upper bound, and it does not fulfill the catalogue's proposed `prove_floor` theorem.
- Non-detections must be conditionally independent given each hypothesis. Do not reuse the same pixels as both a flux likelihood in the prior and a non-detection likelihood. Discrete weights cannot recover missing hypotheses; a concentrated posterior or low effective sample size calls for a better sampled family.
- Invalid/missing predictions are listed in `unavailable`; no fabricated zero-flux measurements are inserted. A trajectory with no usable fits has `joint: null`. Check these fields before interpreting the ranking. Duplicate image-HDU references are rejected; duplicated data stored in different files or HDUs remain the caller's responsibility.

The [methods](docs/methods.md) explain the equations and validation. The [roadmap](docs/roadmap.md) assesses the remaining catalogue ideas and the practical next steps.

## Python API

```python
from magictools import forced_photometry, combine_measurements, update_nulls, rank_exposures
from magictools.nulls import gaussian_log_null

# image is a 2D NumPy array; variance has image-units squared.
measurement = forced_photometry(image, variance, x=42.3, y=17.8, psf_sigma=1.4)
joint = combine_measurements([measurement])

log_null = gaussian_log_null([[10, 2, 10]], threshold_snr=5, coverage=[[1, 1, 0]])
posterior = update_nulls([1, 1, 1], log_null)
ranked = rank_exposures(posterior['posterior_weights'], [[0.5, 0.95, 0.05]])
```

Runtime dependencies: NumPy, SciPy, Astropy. Tests use pytest. The repository includes GitHub Actions checks on Python 3.10 and 3.12.

Independent validation includes analytic bound/unbound two-body trajectories, light-time/parallax cases, and frozen JPL Horizons Ceres vectors. In the ten-day Ceres check, the propagated position differs by about **0.18 km** and the geocentric direction by **0.034–0.038 arcseconds**. These results validate that particular case, not arbitrary close encounters, long arcs, or astrometric precision across the sky. All tests run offline.

## FitSky full-objective research

The [FitSky campaign](research/fit_sky/README.md) implements joint catalogue
inference with source interactions, continuous source/count uncertainty, shared
instrument nuisance, replayable posterior bounds, and global statistical
integration. A confused-source experiment meets its declared evidence and
posterior accuracy targets while retaining ambiguity between one and two
sources. A subsequent continuous-template bound also brings the original
moving-source, two-image, and three-image ZTF experiments within their declared
accuracy targets, with posterior catalogue draws and a bounded held-out
predictive calculation. Broad mixed scenes remain unresolved: **the full FitSky objective is
not achieved**. The original rubric,
failed attempts, source snapshots, and independent controls are retained.

The optional research command is `magictools-fit-sky`; see its
[model and interface guide](docs/fit-sky.md). This work adds no modifications to
the existing Kepler implementation.

`magictools-fit-sky-stream` adds resumable, approximate inference across
exposure sequences, with unknown source counts, shared calibration, conditional
flux updates and split/merge moves. The
[30-exposure mixed-scene follow-up](research/fit_sky/sequential_results/README.md)
retains the original broad priors. Its global accuracy bounds remain unresolved;
handling more exposures is not a full-solution claim.
