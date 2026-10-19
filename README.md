# Magictools astronomy

Working implementations of five ideas in the [Magic tools catalogue](https://magictools-theta.vercel.app/), using local calibrated FITS images. Version 0.2 adds numerical orbit-family propagation, detector coverage, motion-trail fitting, and Bayesian comparisons against explicit artifact models. These are bounded research tools; an all-survey archive and instrument-calibrated discovery pipeline remain future work.

| Tool | Available now | Still needed for the catalogue's full ambition |
| --- | --- | --- |
| `map_cover` | Numerically propagate barycentric orbit samples under the Sun, eight planets, and Moon; compute light-time-corrected directions, local FITS coverage, mask gaps, family fractions, motion trails, and expected S/N. Export inputs for photometry and observation planning. | Archive-wide indexing, orbit fitting/ranging, calibrated depth curves, survey-specific WCS checks. |
| `force_track` | Forced PSF flux, uncertainty, S/N, local background, goodness of fit, and usable PSF fraction along each trajectory, with optional linear exposure trails. Pixel or ICRS coordinates; FITS science, variance, mask extensions. | Archive access, empirical PSFs, WCS refinement, stamp dossiers, instrument-specific reductions. |
| `weigh_stamp` | Compare a trailed source against a stationary PSF, extended source, dipole residual, cosmic-ray spike, narrow streak, and background using explicit priors and marginalized likelihoods. | Complete artifact families, empirical instrument models, blends/chip-edge modeling, injection calibration and multi-epoch vetting. |
| `dig_floor` | Joint pixel-likelihood equivalent over a finite bank of trajectories, with different seeing, noise, and known photometric scales; Gaussian trial-count correction. | Orbit-bank generation, GPUs, correlated-noise models, full-archive indexing, survey injection calibration. |
| `learn_null` | Bayesian reweighting of supplied hypotheses from non-detections and ranking of future exposures by expected information gain. Accepts measured efficiencies, an explicit Gaussian model, or planning inputs exported by `map_cover`. | Orbital ranging, population/brightness priors, real survey calibration. |

These are Python library functions and command-line tools. No API keys, network services, or browser are needed at runtime. Real input files are never uploaded.

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
python -m pytest -q
```

The original `demo` creates 16 FITS images with different seeing, noise, photometric scales, and WCS pointings, plus predictions for an injected moving source and two control trajectories. The photometry, joint detection, and non-detection tools run, and their inputs, outputs, injection truth, and summary are saved. A fixed seed makes the experiment reproducible; an existing nonempty directory is never overwritten.

For the default seed `2026`, the injected reference flux is 85, the recovered flux is about **100.08 ± 10.42**, the strongest single-frame S/N is **4.22**, and the joint S/N is **9.60**. The correct trajectory ranks first. This is a synthetic recovery test, not an observation of a real object or a survey completeness measurement.

The new `orbit_demo` starts from a JPL Horizons Ceres state, creates four orbit hypotheses and eight **synthetic** images with a detector gap, exports coverage and planning inputs, and recovers the injected trajectory at **10.54σ**, with flux **96.77 ± 9.18** for an injection of 100. Separate source and cosmic-ray controls prefer the expected image models. The example is not an observation of Ceres.

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
```

Omit `-o` for JSON on stdout. Output files must be new. Errors return exit code 2. `python -m magictools` is equivalent to `magictools`. Reports contain the software versions and SHA-256 hashes of every input file, including the manifest.

## Using your own data

See [input and output reference](docs/formats.md) for complete manifests. Supply calibrated images, **variance** (not standard deviation or inverse variance), the Gaussian PSF sigma in pixels, and positions at the actual exposure times. Positions may be zero-based pixel centers or ICRS RA/Dec in degrees. Paths are relative to the manifest, so commands work from any directory.

Alternatively, provide barycentric Cartesian orbit samples in **ICRS axes, AU and AU/day, at an explicit MJD TDB epoch**, and let `map_cover` generate the positions. The [orbit coverage and stamp comparison guide](docs/ambitious-tools.md) specifies the new formats, observer conventions, model priors, and limitations. The dynamics default uses Astropy's offline approximate ephemerides; optionally supply a local JPL SPK kernel after installing `.[ephemeris]`.

`dig_floor` assumes a single reference flux with known positive per-exposure `flux_scale`: predicted image flux = reference flux × scale. Match filters or supply a defensible color model. Exposure time, throughput, magnitude zero points, and unit conversion are **not** inferred from headers. A variable source needs a different model.

Use injection-measured `detection_probability` in `learn_null` when available. That probability must include usable coverage and the actual detection procedure. For the Gaussian option, `coverage` means the probability the hypothesis is on usable detector area; it is **not** `force_track`'s `psf_fraction`. A non-detection is an observed failure to pass a specified detection test. A missing image, invalid fit, or off-chip prediction is not itself a measured upper limit.

## Scientific scope

- The likelihood assumes a correct, pixel-integrated circular Gaussian PSF, optionally integrated over a constant-rate linear trail; a constant background per stamp; known, independent Gaussian pixel noise; and independent exposures. Shared difference-image templates and resampling can violate independence. Correlated noise, blends, curved trails, PSF mismatch, and WCS errors need additional modeling.
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
