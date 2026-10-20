"""FITS interface and restricted elliptic Kepler bank for the folding engine."""

from pathlib import Path
from time import perf_counter

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

from .folding import FilterFrame, FoldPlan, search_fold
from .io import _fields, _fits_plane, _named_rows, _provenance, read_json
from .orbits import GM, C_AU_DAY
from .photometry import gaussian_psf


ELEMENTS = ("a_au", "e", "inclination_deg", "node_deg", "periapsis_deg", "mean_anomaly_deg")


def kepler_positions(elements, offsets_days):
    """Heliocentric two-body positions; reference plane is the ICRS equator.

    elements: (N,6), a/e/i/node/argument-of-periapsis/mean-anomaly-at-epoch.
    offsets: (T,) or (N,T), TDB days. Ellipses only (0 <= e <= .95).
    No planetary perturbations, GR, ecliptic rotation, or inferred elements.
    """
    elements = np.asarray(elements, dtype=float)
    if (elements.ndim != 2 or elements.shape[1] != 6 or len(elements) == 0
            or not np.all(np.isfinite(elements)) or np.any(elements[:, 0] < .05)
            or np.any(elements[:, 0] > 1000) or np.any(elements[:, 1] < 0)
            or np.any(elements[:, 1] > .95) or np.any(elements[:, 2] < 0)
            or np.any(elements[:, 2] > 180) or np.any(np.abs(elements[:, 3:]) > 360)):
        raise ValueError("Elliptic elements require .05<=a<=1000 AU, 0<=e<=.95, 0<=i<=180 and |angles|<=360 deg")
    offsets = np.asarray(offsets_days, dtype=float)
    if offsets.ndim == 1:
        offsets = np.broadcast_to(offsets, (len(elements), len(offsets)))
    if (offsets.ndim != 2 or offsets.shape[0] != len(elements) or offsets.size == 0
            or not np.all(np.isfinite(offsets)) or np.max(np.abs(offsets)) > 36525):
        raise ValueError("Kepler offsets must be finite (times,) or (orbits,times), within 100 years")
    a, e = elements[:, 0, None], elements[:, 1, None]
    inc, node, peri, anomaly = np.radians(elements[:, 2:]).T
    mean = (anomaly[:, None] + np.sqrt(GM["sun"]/a**3)*offsets + np.pi) % (2*np.pi) - np.pi
    eccentric = mean + .85*e*np.sign(np.sin(mean))
    for _ in range(50):
        correction = (eccentric-e*np.sin(eccentric)-mean)/(1-e*np.cos(eccentric))
        eccentric -= correction
        if np.max(np.abs(correction)) < 2e-14:
            break
    else:
        raise ValueError("Kepler equation did not converge")
    x, y = a*(np.cos(eccentric)-e), a*np.sqrt(1-e**2)*np.sin(eccentric)
    cn, sn, cp, sp, ci, si = np.cos(node), np.sin(node), np.cos(peri), np.sin(peri), np.cos(inc), np.sin(inc)
    p = np.column_stack((cn*cp-sn*sp*ci, sn*cp+cn*sp*ci, sp*si))
    q = np.column_stack((-cn*sp-sn*cp*ci, -sn*sp+cn*cp*ci, cp*si))
    return x[..., None]*p[:, None, :] + y[..., None]*q[:, None, :]


def project_kepler(elements, epoch, times, observers, wcs_list):
    """Light-time corrected directions; supplied heliocentric observer at reception."""
    times, observers = np.asarray(times, dtype=float), np.asarray(observers, dtype=float)
    if (times.ndim != 1 or not len(times) or observers.shape != (len(times), 3)
            or len(wcs_list) != len(times) or not np.all(np.isfinite(observers))
            or not np.all(np.isfinite(times)) or not np.isfinite(epoch)):
        raise ValueError("Kepler projection needs finite times, epoch, and one heliocentric observer/WCS per exposure")
    output = np.empty((len(elements), len(times), 2))
    offsets = times-epoch
    for start in range(0, len(elements), 2048):
        batch = elements[start:start+2048]
        emitted = np.broadcast_to(offsets, (len(batch), len(times))).copy()
        for _ in range(30):
            delta = kepler_positions(batch, emitted)-observers[None, :, :]
            distance = np.linalg.norm(delta, axis=2)
            if np.any(distance < 1e-8):
                raise ValueError("Object and observer coincide within 1e-8 AU")
            updated = offsets[None, :] - distance/C_AU_DAY
            if np.max(np.abs(updated-emitted)) < 1e-11:
                emitted = updated
                break
            emitted = updated
        else:
            raise ValueError("Kepler light-time iteration did not converge")
        delta = kepler_positions(batch, emitted)-observers[None, :, :]
        ra = np.degrees(np.arctan2(delta[..., 1], delta[..., 0])) % 360
        dec = np.degrees(np.arctan2(delta[..., 2], np.hypot(delta[..., 0], delta[..., 1])))
        for j, wcs in enumerate(wcs_list):
            if not wcs.has_celestial:
                raise ValueError("Kepler orbit inputs require celestial image WCS")
            output[start:start+len(batch), j] = np.column_stack(wcs.world_to_pixel_values(ra[:, j], dec[:, j]))
    # A TAN projection can fail behind the tangent plane: reject rather than
    # inventing coordinates or claiming those members were searched.
    if not np.all(np.isfinite(output)):
        raise ValueError("An orbit lies outside the finite WCS projection; use a suitable patch/bank")
    return output


def run_fold(manifest_path):
    path = Path(manifest_path).resolve()
    value = read_json(path)
    _fields(value, ["schema_version", "exposures"],
            ["trajectories", "orbits", "epoch_mjd_tdb", "cell_subdivisions", "alpha", "top_k", "validate_direct", "benchmark_repeats"])
    if value["schema_version"] != 1 or ("trajectories" in value) == ("orbits" in value):
        raise ValueError("schema_version 1 requires exactly trajectories or orbits")
    _named_rows(value["exposures"], "exposures")
    rows = value.get("orbits", value.get("trajectories"))
    _named_rows(rows, "bank")
    if len(rows) > 100000 or len(value["exposures"]) > 256 or len(rows)*len(value["exposures"]) > 4_000_000:
        raise ValueError("Fold bank exceeds 100000 tracks, 256 frames or 4 million track/frame pairs")
    if not isinstance(value.get("validate_direct", False), bool):
        raise ValueError("validate_direct must be a boolean")
    repeats = value.get("benchmark_repeats", 0)
    if isinstance(repeats, bool) or not isinstance(repeats, int) or not 0 <= repeats <= 9:
        raise ValueError("benchmark_repeats must be an integer in 0..9")
    if "trajectories" in value and "epoch_mjd_tdb" in value:
        raise ValueError("epoch_mjd_tdb belongs to orbit inputs, not supplied pixel trajectories")
    total_start = perf_counter()
    files, sources, frames, wcs_list, times, observers = [path], set(), [], [], [], []
    total_pixels = 0
    for exposure in value["exposures"]:
        _fields(exposure, ["id", "image", "variance", "background"],
                ["image_hdu", "variance_hdu", "mask", "mask_hdu", "psf_sigma_pixels", "psf_image", "psf_hdu",
                 "flux_scale", "mjd_tdb", "observer_heliocentric_au"])
        if ("psf_image" in exposure) == ("psf_sigma_pixels" in exposure):
            raise ValueError("Supply exactly one of psf_image or psf_sigma_pixels")
        image_path = (path.parent/exposure["image"]).resolve()
        with fits.open(image_path, memmap=False) as hdus:
            index = hdus.index_of(exposure.get("image_hdu", 0))
            if (image_path, index) in sources:
                raise ValueError("The same image HDU cannot count twice")
            sources.add((image_path, index))
            image = np.asarray(hdus[index].data, dtype=float)
            wcs_list.append(WCS(hdus[index].header).celestial)
        files.append(image_path)
        total_pixels += image.size
        if total_pixels > 2_000_000:
            raise ValueError("Fold FFT maps are limited to 2 million total input image pixels")
        variance, mask = exposure["variance"], None
        if isinstance(variance, str):
            vp = (path.parent/variance).resolve()
            files.append(vp)
            variance = _fits_plane(vp, exposure.get("variance_hdu", 0))
            if variance.shape != image.shape:
                raise ValueError("Variance image shape must match science image")
        elif isinstance(variance, bool) or not np.isfinite(variance) or variance <= 0:
            raise ValueError("Scalar variance must be finite and positive")
        if "mask" in exposure:
            mp = (path.parent/exposure["mask"]).resolve()
            files.append(mp)
            mask = _fits_plane(mp, exposure.get("mask_hdu", 0))
        if "psf_image" in exposure:
            pp = (path.parent/exposure["psf_image"]).resolve()
            files.append(pp)
            kernel = _fits_plane(pp, exposure.get("psf_hdu", 0))
        else:
            sigma = exposure["psf_sigma_pixels"]
            if not np.isfinite(sigma) or not .2 <= sigma <= 10:
                raise ValueError("Gaussian PSF sigma must be in [.2,10] pixels")
            radius = int(np.ceil(5*sigma))
            yy, xx = np.mgrid[-radius:radius+1, -radius:radius+1]
            kernel = gaussian_psf(xx, yy, 0, 0, sigma)
        frames.append(FilterFrame(image, variance, kernel, exposure["background"], mask, exposure.get("flux_scale", 1.)))
        if "orbits" in value:
            times.append(exposure["mjd_tdb"])
            observers.append(exposure["observer_heliocentric_au"])
    preprocessing = perf_counter()-total_start
    start = perf_counter()
    if "orbits" in value:
        for row in rows:
            _fields(row, ["id", *ELEMENTS])
        elements = np.asarray([[row[key] for key in ELEMENTS] for row in rows])
        positions = project_kepler(elements, value["epoch_mjd_tdb"], times, observers, wcs_list)
        dynamics = "elliptic two-body Sun; heliocentric ICRS-equatorial elements; reception observer, iterated light time"
    else:
        positions = []
        for row in rows:
            _fields(row, ["id", "xy"])
            positions.append(row["xy"])
        positions = np.asarray(positions, dtype=float)
        dynamics = "supplied pixel trajectories; no propagation or physical accuracy inferred"
    propagation = perf_counter()-start
    start = perf_counter()
    plan = FoldPlan(positions, [f.shape for f in frames], value.get("cell_subdivisions", 1))
    compilation = perf_counter()-start
    result = search_fold(frames, plan, [row["id"] for row in rows], value.get("alpha", .01),
                         value.get("top_k", min(10, len(rows))), value.get("validate_direct", False))
    result.update(schema_version=1, tool="fold_kepler", dynamics=dynamics)
    result["work"].update(fft_and_input_seconds=preprocessing, orbit_projection_seconds=propagation,
                           plan_compilation_seconds=compilation, cell_subdivisions=plan.subdivisions)
    if "validation" in result:
        common = preprocessing+propagation
        result["validation"]["single_use_total_speedup"] = ((common+result["validation"]["direct_evaluation_seconds"])
              / (common+compilation+result["work"]["fold_evaluation_seconds"]))
    if repeats:
        # Reuse the compiled plan and FFT maps, warm up both methods above,
        # then validate every timed run. Report sample vectors, not just a ratio.
        folded, direct = [], []
        for _ in range(repeats):
            trial = search_fold(frames, plan, [row["id"] for row in rows], value.get("alpha", .01),
                                value.get("top_k", min(10, len(rows))), True)
            folded.append(trial["work"]["fold_evaluation_seconds"])
            direct.append(trial["validation"]["direct_evaluation_seconds"])
        fmed, dmed = float(np.median(folded)), float(np.median(direct))
        saving = dmed-fmed
        result["benchmark"] = {"repeats": repeats, "fold_seconds": folded, "direct_seconds": direct,
                               "median_evaluation_speedup": dmed/fmed,
                               "single_use_total_speedup": (preprocessing+propagation+dmed)/(preprocessing+propagation+compilation+fmed),
                               "plan_amortization_evaluations": int(np.floor(compilation/saving))+1 if saving > 0 else None,
                               "interpretation": "same FFT maps for both methods; compilation and orbit projection excluded from evaluation speedup; includes exact refinement"}
    result["provenance"] = _provenance(files)
    return result
