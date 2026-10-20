"""Replay real ZTF pixels without substituting the benchmark observer model."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from ..detector_fields import direct_pixel_statistics
from ..factorization import ARCSEC
from ..observer_chart import ObserverChart
from ..real_exposure_experiment import directions, tangent_wcs
from ..temporal.transform import FourierEvaluator
from ..temporal.benchmark import source_hashes
from ..universal import apparent
from .fields import CommonDetector, trim_fields, sum_bounds


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run(directory):
    directory.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[3]
    inputs = root / "research/kepler_transform/chart_results/real_inputs"
    metadata = json.loads((inputs / "observations.json").read_text())
    images, variances, masks, worlds = [], [], [], []
    hashes = {}
    for name in metadata["files"]:
        path = inputs / name
        hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        with fits.open(path) as hdus:
            images.append(hdus["SCI"].data.astype(float))
            variances.append(hdus["VAR"].data.astype(float))
            masks.append(hdus["MASK"].data != 0)
            worlds.append(WCS(hdus["SCI"].header))
    images, variances, masks = map(np.asarray, (images, variances, masks))
    normal, east, north, _ = tangent_wcs(worlds[0], 4., 4.)
    def project(vectors):
        return np.stack((vectors @ east, vectors @ north), axis=-1) / (vectors @ normal)[..., None] / ARCSEC
    matrices, offsets = [], []
    for wcs in worlds:
        points = np.array([[4., 4.], [4.5, 4.], [3.5, 4.], [4., 4.5], [4., 3.5]])
        sky = wcs.all_pix2world(points, 0)
        xy = project(directions(sky[:, 0], sky[:, 1]))
        offsets.append(xy[0])
        matrices.append(np.column_stack((xy[1] - xy[2], xy[3] - xy[4])))
    started = perf_counter()
    detector = CommonDetector(variances, metadata["sigma_pixels"], matrices, offsets=offsets,
                              masks=masks, flux_scales=metadata["flux_scales"],
                              source_half_width=4., spectral_sigma=8.)
    full_n, full_i, report = detector.prepare(images)
    n, i, trim = trim_fields(full_n, full_i, [1e-8, 1e-10])
    preparation = perf_counter() - started
    # Use the actual previous real-data orbit states, actual reception times,
    # and recorded Earth/site ephemerides, independently of the certified demo.
    previous = root / "research/kepler_transform/chart_results/real/queries-and-scores.npz"
    with np.load(previous) as archive:
        indices = np.linspace(0, len(archive["parameters"]) - 1, 256, dtype=int)
        parameters = archive["parameters"][indices]
    chart = ObserverChart(metadata["observers"][0], metadata["observer_velocities"][0],
                          normal, east, north, retarded_epoch=True)
    times = np.array(metadata["mjd_tdb"]) - metadata["mjd_tdb"][0]
    emitted, _ = apparent(chart.states(parameters), times, np.asarray(metadata["observers"]))
    delta = emitted[..., :3] - np.asarray(metadata["observers"])
    positions = project(delta)
    started = perf_counter()
    evaluator = FourierEvaluator(positions, n.shape[1:], [detector.spacing] * 2, backend="nufft")
    actual_n, actual_i = evaluator.apply(n), evaluator.apply(i)
    fourier_seconds = perf_counter() - started
    started = perf_counter()
    direct_n, direct_i = detector.direct(images, positions)
    direct_seconds = perf_counter() - started
    ra = np.rad2deg(np.arctan2(delta[..., 1], delta[..., 0])) % 360
    dec = np.rad2deg(np.arctan2(delta[..., 2], np.hypot(delta[..., 0], delta[..., 1])))
    actual_pixels = np.empty_like(positions)
    for j, wcs in enumerate(worlds):
        x, y = wcs.all_world2pix(ra[:, j], dec[:, j], 0)
        actual_pixels[:, j] = np.column_stack((x - 4, y - 4))
    tpv_n, tpv_i = direct_pixel_statistics(images, variances, metadata["sigma_pixels"], actual_pixels,
                                          masks=masks, profile_background=True,
                                          flux_scales=metadata["flux_scales"])
    nbound = sum_bounds([x["numerator_model_error_bound"] for x in report["exposures"]] +
                        [trim["discarded_numerator_l1_bound"]])
    ibound = sum_bounds([x["information_model_error_bound"] for x in report["exposures"]] +
                        [trim["discarded_information_l1_bound"]])
    nerror, ierror = float(max(abs(actual_n - direct_n))), float(max(abs(actual_i - direct_i)))
    if nerror > nbound + 1e-10 or ierror > ibound + 1e-11:
        raise ValueError("Real-data affine detector comparison exceeds its model budget")
    if min(actual_i.real.min(), direct_i.min(), tpv_i.min()) <= 0:
        raise ValueError("Unresolved nonpositive information in real-data replay")
    snr, direct_snr, tpv_snr = actual_n.real / np.sqrt(actual_i.real), direct_n / np.sqrt(direct_i), tpv_n / np.sqrt(tpv_i)
    result = {
        "real_exposures": len(images), "pixels_per_exposure": images.shape[1]**2,
        "queries": len(parameters), "span_minutes": float(np.ptp(times) * 1440),
        "source_sha256": source_hashes(),
        "input_sha256": {**hashes, str(previous.relative_to(root)): hashlib.sha256(previous.read_bytes()).hexdigest(),
                         str((inputs / "observations.json").relative_to(root)):
                         hashlib.sha256((inputs / "observations.json").read_bytes()).hexdigest()},
        "detector": report, "trim": trim,
        "maximum_numerator_error_affine_pixels": nerror, "maximum_information_error_affine_pixels": ierror,
        "numerator_analytic_model_budget": nbound, "information_analytic_model_budget": ibound,
        "maximum_snr_error_affine_pixels": float(max(abs(snr - direct_snr))),
        "maximum_snr_error_actual_tpv_pixels": float(max(abs(snr - tpv_snr))),
        "maximum_sampled_affine_tpv_position_error_pixels": float(np.max(abs(detector.pixel_positions(positions) - actual_pixels))),
        "same_best_query_affine": bool(np.argmax(snr) == np.argmax(direct_snr)),
        "same_best_query_tpv": bool(np.argmax(snr) == np.argmax(tpv_snr)),
        "preparation_seconds": preparation, "fourier_query_seconds": fourier_seconds,
        "direct_pixel_seconds_same_positions": direct_seconds,
        "time_compression_applied_to_real_stack": False,
        "reason": "Three real exposures do not benefit from compression; the analytic time proof uses a different observer model.",
        "wcs_residual_is_sampled_not_certified": True,
        "psf": "pixel-integrated Gaussian approximation; empirical survey PSF retained in original FITS only",
        "noise": "supplied estimated diagonal variances, not a calibrated survey covariance",
        "full_objective_complete": False,
    }
    np.savez_compressed(directory / "scores.npz", positions_arcsec=positions, numerator=actual_n,
                        information=actual_i, affine_numerator=direct_n, affine_information=direct_i,
                        tpv_numerator=tpv_n, tpv_information=tpv_i)
    result["artifact_sha256"] = {"scores.npz": hashlib.sha256((directory / "scores.npz").read_bytes()).hexdigest()}
    save(directory / "results.json", result)
    print(json.dumps({k: result[k] for k in ["maximum_snr_error_affine_pixels",
          "maximum_snr_error_actual_tpv_pixels", "preparation_seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    run(parser.parse_args().directory)
