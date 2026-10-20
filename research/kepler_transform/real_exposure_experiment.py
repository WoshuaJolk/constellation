"""Exercise the chart engine on real ZTF pixels with explicit model limitations.

Preparation uses an existing ztf_fetch inventory; replay uses archived cropped
FITS files and recorded observer positions, with no remote dependency.
Gaussian PSFs and estimated variances are declared approximations. The actual
TPV WCS is an independent comparison to the local affine forward model.
"""

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation, get_body_barycentric_posvel
from astropy.io import fits
from astropy.time import Time
from astropy.utils import iers
from astropy.wcs import WCS

from magictools.orbits import GM
from .chart_cover import ChartCover, physical_kernel
from .detector_fields import finite_pixel_fields, direct_pixel_statistics
from .factorization import ARCSEC
from .kernel import KeplerKernel
from .observer_chart import ObserverChart
from .universal import apparent


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def directions(ra, dec):
    ra, dec = np.deg2rad(ra), np.deg2rad(dec)
    return np.stack((np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)), axis=-1)


def tangent_wcs(wcs, x, y):
    ra, dec = wcs.all_pix2world(x, y, 0)
    normal = directions(ra, dec)
    ra, dec = np.deg2rad(ra), np.deg2rad(dec)
    east = np.array([-np.sin(ra), np.cos(ra), 0.])
    north = np.cross(normal, east)
    sample = np.array([[x+.5, y], [x-.5, y], [x, y+.5], [x, y-.5]])
    sky = wcs.all_pix2world(sample, 0)
    vector = directions(sky[:, 0], sky[:, 1])
    tangent = np.column_stack((vector @ east, vector @ north))/(vector @ normal)[:, None]
    jacobian = np.column_stack((tangent[0]-tangent[1], tangent[2]-tangent[3]))
    return normal, east, north, np.linalg.inv(jacobian)


def prepare(inventory_path, directory):
    directory.mkdir(parents=True, exist_ok=False)
    inventory_path = Path(inventory_path)
    inventory = json.loads(inventory_path.read_text())
    exposures = inventory["exposures"]
    if len(exposures) < 2:
        raise ValueError("At least two actual exposures required")
    times = np.array([row["mjd_tdb"] for row in exposures])
    astropy_times = Time(times, format="mjd", scale="tdb")
    earth_p, earth_v = get_body_barycentric_posvel("earth", astropy_times, ephemeris="builtin")
    sun_p, sun_v = get_body_barycentric_posvel("sun", astropy_times, ephemeris="builtin")
    observers, velocities, provenance = [], [], []
    for index, exposure in enumerate(exposures):
        source = Path(exposure["image"])
        if not source.is_absolute():
            source = inventory_path.parent/source
        with fits.open(source) as hdus:
            header = hdus["SCI"].header.copy()
            wcs = WCS(header)
            x, y = wcs.all_world2pix(180., 30., 0)
            x, y = int(round(float(x))), int(round(float(y)))
            extent = (slice(y-4, y+5), slice(x-4, x+5))
            header["CRPIX1"] -= x-4
            header["CRPIX2"] -= y-4
            pieces = [fits.PrimaryHDU()]
            for name in ["SCI", "VAR", "MASK"]:
                data = np.asarray(hdus[name].data[extent]).copy()
                if data.shape != (9, 9):
                    raise ValueError("Requested real-data crop is outside image")
                pieces.append(fits.ImageHDU(data, header=header if name == "SCI" else None, name=name))
            # Retain the supplied empirical PSF for provenance; this experiment
            # explicitly uses a Gaussian approximation, not this empirical PSF.
            pieces.append(fits.ImageHDU(hdus["PSF"].data.copy(), name="SURVEYPSF"))
            fits.HDUList(pieces).writeto(directory/f"frame-{index}.fits")
        site = exposure["observatory"]
        location = EarthLocation.from_geodetic(site["longitude_deg"]*u.deg,
                                               site["latitude_deg"]*u.deg, site["height_m"]*u.m)
        with iers.conf.set_temp("auto_download", False):
            p, v = location.get_gcrs_posvel(astropy_times[index])
        observers.append((earth_p.xyz[:, index]-sun_p.xyz[:, index]+p.xyz).to_value(u.au))
        velocities.append((earth_v.xyz[:, index]-sun_v.xyz[:, index]+v.xyz).to_value(u.au/u.day))
        provenance.append({"id": exposure["id"], "prepared_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                           "crop_center_original_pixel": [x, y], "exposure_seconds": exposure["exposure_seconds"]})
    manifest = {"files": [f"frame-{j}.fits" for j in range(len(exposures))],
                "mjd_tdb": times.tolist(), "observers": np.asarray(observers).tolist(),
                "observer_velocities": np.asarray(velocities).tolist(),
                "sigma_pixels": [[r["psf_sigma_pixels"]]*2 for r in exposures],
                "flux_scales": [r["flux_scale"] for r in exposures],
                "provenance": provenance, "public_products": inventory["products"],
                "original_calibration": inventory["calibration"],
                "ephemeris": "Astropy builtin Earth-Sun plus observatory GCRS displacement; recorded here",
                "real_survey_data": True}
    save(directory/"observations.json", manifest)
    return directory/"observations.json"


def run(observations, output):
    observations, output = Path(observations), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads(observations.read_text())
    images, variances, masks, wcs_objects, bases, matrices = [], [], [], [], [], []
    for path in metadata["files"]:
        with fits.open(observations.parent/path) as hdus:
            images.append(hdus["SCI"].data.astype(float))
            variances.append(hdus["VAR"].data.astype(float))
            masks.append(hdus["MASK"].data != 0)
            wcs = WCS(hdus["SCI"].header)
            wcs_objects.append(wcs)
            normal, east, north, affine = tangent_wcs(wcs, 4., 4.)
            bases.append([normal, east, north])
            matrices.append(affine)
    images, variances, masks = np.array(images), np.array(variances), np.array(masks)
    if not np.all(np.isfinite(images)) or not np.all(np.isfinite(variances)) or np.any(variances <= 0):
        raise ValueError("Finite real cutouts and positive declared variance required")
    bases, matrices = np.array(bases), np.array(matrices)
    times = np.asarray(metadata["mjd_tdb"])-metadata["mjd_tdb"][0]
    observers = np.asarray(metadata["observers"])
    chart = ObserverChart(observers[0], metadata["observer_velocities"][0],
                          *bases[0], retarded_epoch=True)
    rho = 1/40.
    r = observers[0]+bases[0, 0]/rho
    tangent = np.cross([0., 0., 1.], r)
    v = tangent/np.linalg.norm(tangent)*np.sqrt(GM["sun"]/np.linalg.norm(r))
    relative = v-chart.velocity
    center = np.array([rho, rho*(relative @ bases[0, 0]), rho*(relative @ bases[0, 2]),
                       0., 0., rho*(relative @ bases[0, 1])])
    widths = np.array([.00625, .0001, ARCSEC, ARCSEC, ARCSEC, ARCSEC])
    start = perf_counter()
    visits, frequencies, numerator, information, detector = finite_pixel_fields(
        images, variances, metadata["sigma_pixels"], 1., masks=masks,
        source_half_width_pixels=4., profile_background=True, flux_scales=metadata["flux_scales"])
    # k_pixel . (A * tangent) = (k_pixel * A) . tangent.
    frequencies = np.einsum("mi,mij->mj", frequencies, matrices[visits])
    reference = chart.states(center[None])[0]
    geometry = KeplerKernel(reference[None], times, observers, visits, frequencies, reference)
    geometry.normal, geometry.east, geometry.north = bases[:, 0], bases[:, 1], bases[:, 2]
    preparation_seconds = perf_counter()-start
    cover = ChartCover(chart, center, widths, geometry, position_tolerance=.0005*ARCSEC, max_leaves=8)
    if not cover.report()["all_cells_resolved"]:
        raise ValueError("Real-image geometry domain unresolved")
    # Certify source positions stay inside the finite detector approximation's
    # declared domain, for every point in each continuous 6D cell.
    from flint import arb, ctx
    from research.detection_limit.interval_orbits import upper_float
    from .local_transform import POWERS
    position_bounds = []
    with ctx.workprec(128):
        for leaf in cover.leaves:
            atlas, certificate = cover.atlases[leaf["path"]], cover.certificates[leaf["path"]]
            for t in range(len(times)):
                bounds = []
                for axis in range(2):
                    bound = arb(0)
                    for powers in atlas.indices:
                        if powers[3:] in POWERS:
                            bound += abs(sum((arb(float(matrices[t, axis, j]))*
                                              arb(float(atlas.coefficients[powers+(t, j)]))
                                              for j in range(2)), arb(0)))
                    bound += sum((abs(arb(float(matrices[t, axis, j])))*
                                  arb(float(certificate["position_error_radians"][t][j]))
                                  for j in range(2)), arb(0))
                    bounds.append(upper_float(bound))
                position_bounds.append(bounds)
    if np.max(position_bounds) > 4:
        raise ValueError("Certified trajectories leave declared finite detector domain")
    rng = np.random.default_rng(20261024)
    inner = rng.uniform(-.95, .95, (2048, 3))
    normalized = np.concatenate([np.column_stack((np.tile(outer, (len(inner), 1)), inner))
                                 for outer in product([-.7, .7], repeat=3)])
    parameters = center+normalized*widths
    start = perf_counter()
    fast_n, report_n = cover.apply(numerator, parameters, route="slices")
    fast_i, report_i = cover.apply(information, parameters, route="slices")
    fast_seconds = perf_counter()-start
    start = perf_counter()
    states = chart.states(parameters)
    exact = physical_kernel(geometry, states)
    gnomonic = exact.projected(np.arange(len(states)), np.arange(len(times)))
    affine_positions = np.einsum("tij,ntj->nti", matrices, gnomonic)
    n_affine, i_affine = direct_pixel_statistics(images, variances, metadata["sigma_pixels"],
        affine_positions, masks=masks, profile_background=True, flux_scales=metadata["flux_scales"])
    direct_seconds = perf_counter()-start
    emitted, _ = apparent(states, times, observers)
    delta = emitted[..., :3]-observers
    ra = np.rad2deg(np.arctan2(delta[..., 1], delta[..., 0])) % 360
    dec = np.rad2deg(np.arctan2(delta[..., 2], np.hypot(delta[..., 0], delta[..., 1])))
    actual_positions = np.empty_like(affine_positions)
    for t, wcs in enumerate(wcs_objects):
        x, y = wcs.all_world2pix(ra[:, t], dec[:, t], 0)
        actual_positions[:, t] = np.column_stack((x-4, y-4))
    n_wcs, i_wcs = direct_pixel_statistics(images, variances, metadata["sigma_pixels"],
        actual_positions, masks=masks, profile_background=True, flux_scales=metadata["flux_scales"])
    if np.any(fast_i.real <= 0) or np.any(i_affine <= 0) or np.any(i_wcs <= 0):
        raise ValueError("Nonpositive information: cannot report a source score")
    snr, affine_snr, actual_snr = fast_n.real/np.sqrt(fast_i.real), n_affine/np.sqrt(i_affine), n_wcs/np.sqrt(i_wcs)
    result = {"real_survey_data": True, "real_exposures": len(times), "queries": len(parameters),
              "observation_span_minutes": float(np.ptp(times)*1440),
              "pixels_per_exposure": 81, "fourier_modes": len(visits),
              "detector_model": detector, "cover": cover.report(),
              "continuous_source_position_absolute_bound_pixels": position_bounds,
              "field_preparation_seconds": preparation_seconds,
              "fast_both_statistics_seconds": fast_seconds,
              "physical_geometry_plus_direct_pixels_seconds": direct_seconds,
              "total_fast_seconds": preparation_seconds+cover.construction_seconds+fast_seconds,
              "speedup_against_direct_pixels": direct_seconds/(preparation_seconds+cover.construction_seconds+fast_seconds),
              "maximum_numerator_error_same_affine_model": float(max(abs(fast_n.real-n_affine))),
              "maximum_information_error_same_affine_model": float(max(abs(fast_i.real-i_affine))),
              "maximum_snr_error_same_affine_model": float(max(abs(snr-affine_snr))),
              "maximum_snr_error_actual_tpv_wcs": float(max(abs(snr-actual_snr))),
              "maximum_sampled_affine_wcs_position_error_pixels": float(np.max(abs(actual_positions-affine_positions))),
              "same_best_query_as_direct_affine": bool(np.argmax(snr) == np.argmax(affine_snr)),
              "same_best_query_as_direct_tpv": bool(np.argmax(snr) == np.argmax(actual_snr)),
              "numerator_application": report_n, "information_application": report_i,
              "discovery_claimed": False, "full_objective_complete": False,
              "limitations": ["Gaussian approximation replaces supplied empirical survey PSF",
                              "Variance is the prior adapter's estimate, not a survey covariance product",
                              "Local affine WCS: TPV disagreement measured at queries, not globally certified",
                              "Exposure trailing and PSF spatial variation omitted",
                              "No calibrated false-positive or completeness claim from this small stack"]}
    np.savez_compressed(output/"queries-and-scores.npz", parameters=parameters, numerator=fast_n,
                        information=fast_i, direct_affine_numerator=n_affine,
                        direct_affine_information=i_affine, direct_tpv_numerator=n_wcs,
                        direct_tpv_information=i_wcs)
    save(output/"certificates.json", cover.certificates)
    save(output/"result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--inventory", type=Path)
    group.add_argument("--observations", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    observations = args.observations
    if args.inventory:
        observations = prepare(args.inventory, args.output.with_name(args.output.name+"-inputs"))
    result = run(observations, args.output)
    print(json.dumps({key: result[key] for key in ["queries", "real_exposures",
                     "maximum_snr_error_same_affine_model", "maximum_snr_error_actual_tpv_wcs",
                     "speedup_against_direct_pixels"]}, indent=2))


if __name__ == "__main__":
    main()
