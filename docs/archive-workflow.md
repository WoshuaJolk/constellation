# Real archive images, candidate diagnostics, and recovery experiments

Version 0.4 connects the first four priorities: ZTF archive inputs for `map_cover`,
empirical PSFs for `force_track`, a `judge_line` dossier, and injections that rerun
`dig_floor` on real backgrounds. It does not assert all-survey coverage or certify
discoveries. All network access is in the explicit `ztf_search` / `ztf_fetch`
commands; the inference commands still run offline.

## Reproduce the live example

```bash
python examples/ztf_workflow.py /path/to/new-ztf-experiment
```

This downloads three historical public ZTF r-band exposures near RA 180°, Dec 30°
on 2019-01-28 (if the archive inventory remains unchanged). It propagates three
**synthetic** orbital hypotheses onto these actual images, runs coverage,
photometry and candidate diagnostics, then adds synthetic sources at 15 sky
offsets at five flux levels. It saves all inputs, raw products, prepared FITS
files, checksums, URLs, and JSON reports. Allow a few minutes and several MB.
An interrupted fetch leaves its downloaded files for inspection; rerun with a new
directory. No files are overwritten. Network tests are not part of offline CI.

The October 19, 2026 live run retrieved PIDs `757381141115`, `757388711115`,
`757389683415`. All three hypotheses lay on usable pixels in all three cutouts.
The uninjected primary track had joint S/N **−1.22**, correctly failing the
Gaussian threshold. Injection reference fluxes 0, 10, 30, 100, 300 produced correct
track recovery counts **0, 2, 15, 15, 15 out of 15**. One shifted search already
contained a detection before injection; at flux 30 the count of *new* recoveries
was therefore **14/15**, not 15/15. These are results on these placements, not a
survey completeness curve or a real-object discovery. High flux tests omit added
source shot noise. No binomial confidence interval is asserted for overlapping
placements on the same images.

## Search the public exposure index

`magictools ztf_search query.json -o index.json` accepts:

```json
{
  "schema_version": 1,
  "regions": [{"ra_deg": 180, "dec_deg": 30, "width_deg": 0.01}],
  "start_mjd_utc": 58511.38,
  "end_mjd_utc": 58511.41,
  "filtercode": "zr",
  "max_exposures": 20,
  "require_zero_infobits": true
}
```

The boxes are in degrees and use IBE's `OVERLAPS` footprint intersection. The
UTC interval is half-open and filters the index's observation-start `obsjd`.
Supply boxes covering the full orbital uncertainty region and time span you
intend to search. This adapter does not prove that boxes enclose a continuous
orbit family between sampling times. Queries support one filter (`zg`, `zr`, or
`zi`), at most 100 boxes and 366 days, with 1–10,000 unique exposure products.
Exceeding the cap is an error, never silent truncation. The response must declare
`QUERY_STATUS=OK` and a consistent row count. Products are deduplicated by PID;
all returned metadata, including corners, seeing, quality flags and catalogue
depth, are retained. Catalogue depth is not a measured efficiency for your search.

## Fetch selected products

`magictools ztf_fetch fetch.json -o inventory.json` accepts:

```json
{
  "schema_version": 1,
  "index": "index.json",
  "directory": "new-download-directory",
  "reference_mag": 25,
  "max_total_bytes": 134217728,
  "selections": [{
    "pid": "757381141115",
    "ra_deg": 180, "dec_deg": 30, "size_arcsec": 180
  }]
}
```

Select 1–100 distinct product IDs, one filter, with cutout sides 20–1200 arcsec.
The download budget is enforced while reading (default 128 MiB, maximum 1 GiB).
Transient HTTP failures receive two bounded retries. The destination directory
must be new. The prepared multi-extension FITS contains science (`SCI`), estimated
variance (`VAR`), conservative invalid mask (`MASK`), and measured PSF (`PSF`).
Raw science, mask and quadrant-center PSF products are also retained. Product ID,
filter, time, science/mask shape and WCS alignment are checked.

Calibration choices are recorded in the inventory:

- **Time:** actual shutter-open/close midpoint when available, converted UTC →
  TDB; observation-start plus half duration otherwise. Shutter opening may lag
  `OBSJD` by up to five seconds; larger discrepancies fail validation.
- **Location:** geodetic longitude/latitude/altitude from the science header;
  `map_cover` adds the IERS-rotated site displacement to Earth's barycentric
  position using offline IERS tables. Meter-scale relativistic GCRS/ICRS site
  corrections and atmospheric effects are omitted.
- **Flux:** `flux_scale = 10**(0.4*(MAGZP-reference_mag))`, so reference magnitude
  is `reference_mag - 2.5*log10(reference_flux)` in the adopted native ZTF system.
  No color correction or zero-point uncertainty is fitted. The finite empirical
  PSF and its interpolation may introduce further photometric bias.
- **Noise:** median sky and `1.4826*MAD` from unmasked cutout pixels; variance is
  sky RMS squared plus `max(image-sky,0)/GAIN`. This is an estimated local model,
  **not an archive variance product**. Crowding, background gradients, flat-field
  noise and correlations can invalidate its Gaussian p-values.
- **Mask:** all nonzero survey mask bits, nonfinite pixels and saturation are
  excluded. This is deliberately conservative and may remove usable pixels.
- **PSF:** native-pixel survey PSF at quadrant center, retaining negative noisy
  wings. Spatial PSF variation is not inferred from the cutout. Header `SEEING`
  is FWHM **in pixels**; metadata-table seeing has different documented units.

To use the inventory, replace `exposures` in a `map_cover` manifest with
`"inventory": "inventory.json"`. All orbit fields and options are unchanged.
Coverage refers to the **downloaded cutout**, not the full quadrant footprint.
Exported photometry and planning manifests work as before. The inventory also
provides `photometry_exposures` for independently supplied trajectories.

## Empirical PSFs with force_track and dig_floor

Any exposure can specify `psf_image` and optional `psf_hdu` in addition to its
existing fields. The complete odd-sized 3–255-pixel kernel must be finite with
positive sum and is normalized once; cropped or masked stamps are never
renormalized. Its central array pixel is the origin. Bilinear native-pixel shifts
with zero padding preserve flux and allow linear trails up to 64 pixels. This is
a sampled empirical template, not an oversampled optical PSF reconstruction;
interpolation broadens it. Kernel-estimation uncertainty is not in the reported
flux error. `psf_fraction` is signed template mass for kernels with negative wings.
The supplied `psf_sigma_pixels` remains required for the Gaussian fallback.

## judge_line

```json
{
  "schema_version": 1,
  "photometry_manifest": "tracks.json",
  "candidate_id": "candidate-a",
  "times_mjd_tdb": {"exposure-a": 60000, "exposure-b": 60000.01},
  "alpha": 0.01,
  "jackknife_min_snr": 3
}
```

Run `magictools judge_line input.json -o dossier.json`. Supply a midpoint for
every exposure. Celestial WCS is required to compare the same sky position
across images. The dossier includes the original finite-bank trial correction,
signed per-frame fits, Gaussian residual probabilities, constant-flux test,
chronological halves, leave-one-exposure-out fits, three stationary sky controls
at the sampled beginning/middle/end, and a reversed track about the sampled
middle position. Each control comparison recombines the candidate using only
**common usable exposures**. Differences in S/N are diagnostics, not Bayes
factors. These finite controls do not optimize all possible stationary blends.

Flags identify low significance, fewer than three frames, leave-one-out fragility,
controls matching/exceeding the candidate, incomplete control coverage, or no
temporal baseline. They are not a calibrated acceptance rule; the combined
false-positive probability is explicitly `null`. Orbit refitting, astrometric
residual tests and held-out real artifact calibration remain future work.

## Injection recovery through the actual finite-bank search

```json
{
  "schema_version": 1,
  "photometry_manifest": "tracks.json",
  "injection_trajectory_id": "candidate-a",
  "reference_fluxes": [0, 10, 30, 100],
  "offsets_arcsec": [[0, 0], [20, 0], [0, 20]],
  "alpha": 0.01,
  "injection_psf_sigma_factor": 1
}
```

Run `magictools inject_tracks input.json -o recovery.json`. Each east/north offset
shifts the whole supplied bank on the sky; exactly one source is added to the
unchanged background at a time. All candidates are fitted, preserving masks,
scales, PSFs, negative measurements and the full trial count. Correct recovery
requires the injected ID to rank first and pass the Gaussian family threshold;
numerical ties are not successful identifications. Off-chip injections count as
misses. Baseline searches run before injection, and `new_recovery` excludes
placements already triggering the search. Neither selection silently drops
difficult placements. Summary flux residuals include all usable fits, avoiding
conditioning them on detection.

Signals use the fitted PSF by default, a useful plumbing test but optimistic about
model mismatch. For Gaussian models, set `injection_psf_sigma_factor` between 0.5
and 2 to test width mismatch without changing the fitting PSF. Empirical-kernel
width perturbations are rejected. Added source shot noise is not simulated;
restrict scientific efficiency interpretation to background-dominated sources.
Offsets preserve local pixel trail vectors and are intended for small fields;
large distortions across the shifted field require fresh propagated tracks.
At most 200,000 frame fits are allowed per request.

The summary describes these explicit backgrounds and placements. It is not
`map_complete` for an underlying population and cannot be fed to `learn_null` as
an efficiency for another selection rule, brightness, rate, or coverage region.
Use separately validated calibration samples and keep evaluated pixels out of
their own selection calibration.

## References

- [IRSA ZTF API](https://irsa.ipac.caltech.edu/docs/program_interface/ztf_api.html)
- [ZTF product paths and metadata](https://irsa.ipac.caltech.edu/docs/program_interface/ztf_metadata.html)
- [ZTF data documentation](https://irsa.ipac.caltech.edu/Missions/ztf.html)
