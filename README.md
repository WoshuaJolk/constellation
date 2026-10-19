# Magictools astronomy

Working statistical cores for three ideas in the [Magic tools catalogue](https://magictools-theta.vercel.app/). This repository starts with local calibrated FITS images and explicitly supplied trajectory predictions. It does **not** claim to have built an all-survey archive, a new orbital transform, or a discovery pipeline with validated real-world false-positive rates.

| Tool | Available now | Still needed for the catalogue's full ambition |
| --- | --- | --- |
| `force_track` | Forced PSF flux, uncertainty, S/N, local background, goodness of fit, and usable PSF fraction along each supplied trajectory. Pixel or ICRS coordinates; FITS science, variance, mask extensions. | Archive access, calibrated empirical/trailing PSFs, WCS refinement, stamp dossiers, instrument-specific reductions. |
| `dig_floor` | Joint pixel-likelihood equivalent over a finite bank of trajectories, with different seeing, noise, and known photometric scales; Gaussian trial-count correction. | Orbit-bank generation, GPUs, correlated-noise models, full-archive indexing, survey injection calibration. |
| `learn_null` | Bayesian reweighting of supplied hypotheses from non-detections and ranking of future exposures by expected information gain. Accepts measured detection efficiencies or an explicit Gaussian model. | Orbital ranging/propagation, archive footprint integration, population/brightness priors, real survey calibration. |

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
python -m pytest -q
```

The demo creates 16 FITS images with different seeing, noise, photometric scales, and WCS pointings, plus predictions for an injected moving source and two control trajectories. All three tools run, and their inputs, outputs, injection truth, and summary are saved. A fixed seed makes the experiment reproducible; an existing nonempty directory is never overwritten.

For the default seed `2026`, the injected reference flux is 85, the recovered flux is about **100.08 ± 10.42**, the strongest single-frame S/N is **4.22**, and the joint S/N is **9.60**. The correct trajectory ranks first. This is a synthetic recovery test, not an observation of a real object or a survey completeness measurement.

Run the tools separately:

```bash
magictools force_track demo-output/images.json -o forced.json
magictools dig_floor demo-output/images.json --alpha 0.01 -o joint.json
magictools learn_null demo-output/nulls.json -o posterior.json
```

Omit `-o` for JSON on stdout. Output files must be new. Errors return exit code 2. `python -m magictools` is equivalent to `magictools`. Reports contain the software versions and SHA-256 hashes of every input file, including the manifest.

## Using your own data

See [input and output reference](docs/formats.md) for complete manifests. Supply calibrated images, **variance** (not standard deviation or inverse variance), the Gaussian PSF sigma in pixels, and positions at the actual exposure times. Positions may be zero-based pixel centers or ICRS RA/Dec in degrees. Paths are relative to the manifest, so commands work from any directory.

`dig_floor` assumes a single reference flux with known positive per-exposure `flux_scale`: predicted image flux = reference flux × scale. Match filters or supply a defensible color model. Exposure time, throughput, magnitude zero points, and unit conversion are **not** inferred from headers. A variable source needs a different model.

Use injection-measured `detection_probability` in `learn_null` when available. That probability must include usable coverage and the actual detection procedure. For the Gaussian option, `coverage` means the probability the hypothesis is on usable detector area; it is **not** `force_track`'s `psf_fraction`. A non-detection is an observed failure to pass a specified detection test. A missing image, invalid fit, or off-chip prediction is not itself a measured upper limit.

## Scientific scope

- The likelihood assumes a correct, pixel-integrated circular Gaussian PSF; a constant background per stamp; known, independent Gaussian pixel noise; and independent exposures. Shared difference-image templates and resampling can violate independence. Correlated noise, blends, cosmic rays, trails, and WCS errors need additional models or masking.
- Forced positions and the trajectory bank must be specified independently of the fitted pixel noise. The tool does not fit or propagate an orbit, search every possible trajectory, or account for adaptive searches performed outside this run.
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
