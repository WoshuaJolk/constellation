"""Reproduce segment-composition attempts and a whole-domain inverse bound."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from flint import arb, arb_mat, ctx
from scipy.optimize import least_squares

from magictools.orbits import GM
from research.detection_limit.interval_orbits import lower_float, upper_float
from .certify_global import partitioned_gradients
from .chart_experiments import setup
from .compiled_reference import projected_chart
from .complex_source_certificate import global_speed_certificate, root_and_gradients
from .factorization import ARCSEC, observer_path
from .segment_composition import ReceptionChart, conditional_tree_counts, finite_difference_vector
from .universal import apparent


def reception_chart(time):
    rate = np.sqrt(GM["sun"])
    observer = observer_path(np.array([time]))[0]
    velocity = np.array([-rate*np.sin(rate*time), rate*np.cos(rate*time), 0.])
    return ReceptionChart(observer, velocity, [1., 0., 0.], [0., 1., 0.], [0., 0., 1.])


def geometry(times, axes):
    times = np.asarray(times, float)
    return SimpleNamespace(times=times, observers=observer_path(times),
                           normal=np.tile(axes[0], (len(times), 1)),
                           east=np.tile(axes[1], (len(times), 1)),
                           north=np.tile(axes[2], (len(times), 1)))


def positions_from_states(states, times):
    emitted, _ = apparent(states, times, observer_path(times))
    delta = emitted[..., :3]-observer_path(times)
    if np.any(delta[..., 0] <= 0):
        raise ValueError("Projection horizon crossed")
    return delta[..., 1:]/delta[..., :1]


def composition_experiment(case):
    old, center, widths, g, _ = setup(case)
    rng = np.random.default_rng(20261025)
    original = center+rng.uniform(-.9, .9, (64, 6))*widths
    states = old.states(original)
    root, left, right = [reception_chart(t) for t in [0., -1., 1.]]
    q = root.parameters(states)
    recovered = root.states(q)
    q_left = root.rebase(q, -1., left)
    direct = root.rebase(q, 1., right)
    composed = left.rebase(q_left, 2., right)
    scale = np.maximum(abs(direct), np.array([.01, .001, ARCSEC, ARCSEC, ARCSEC, ARCSEC]))
    h = 1e-3
    actual_rate = (positions_from_states(states, np.array([-h, h]))[:, 1]
                   - positions_from_states(states, np.array([-h, h]))[:, 0])/(2*h)
    desired_rate = q[:, [5, 2]]
    energies = np.sum(states[:, 3:]**2, axis=1)/2-GM["sun"]/np.linalg.norm(states[:, :3], axis=1)
    # Scalar response identity, including the different reception epochs.
    times = np.linspace(-2., 2., 9)
    frequencies = rng.uniform(-1, 1, (len(times), 2))/ARCSEC
    data = rng.normal(size=len(times))+1j*rng.normal(size=len(times))
    all_positions = positions_from_states(states, times)
    reference = np.exp(1j*np.einsum("nti,ti->nt", all_positions, frequencies)) @ data
    value = np.zeros(len(states), complex)
    for selection, epoch, chart in [(np.arange(4), -1., left), (np.arange(4, 9), 1., right)]:
        local_q = root.rebase(q, epoch, chart)
        local_state = chart.states(local_q)
        emitted, _ = apparent(local_state, times[selection]-epoch, observer_path(times[selection]))
        delta = emitted[..., :3]-observer_path(times[selection])
        pos = delta[..., 1:]/delta[..., :1]
        value += np.exp(1j*np.einsum("nti,ti->nt", pos, frequencies[selection])) @ data[selection]
    return {"case": case, "states": len(states),
            "original_domain_center": center.tolist(), "original_domain_half_widths": widths.tolist(),
            "state_roundtrip_max_absolute": float(np.max(abs(states-recovered))),
            "rebase_composition_scaled_error": float(np.max(abs(direct-composed)/scale)),
            "observed_rate_finite_difference_error_radians_day": float(np.max(abs(actual_rate-desired_rate))),
            "segment_score_identity_max_error": float(np.max(abs(reference-value))),
            "bound_states": int(np.sum(energies < 0)), "unbound_states": int(np.sum(energies >= 0)),
            "identity_only_not_a_fast_transform": True}


def hidden_coordinate_experiment(case):
    old, center, widths, _, _ = setup(case)
    chart = reception_chart(0.)
    q = chart.parameters(old.states(center[None]))[0]
    variants = np.tile(q, (3, 1))
    variants[1, 0] += .2*widths[0]
    variants[2, 1] += .2*widths[1]
    states = chart.states(variants)
    rows = []
    for h in [1., .5, .25, .125, .0625]:
        positions = positions_from_states(states, np.array([h]))[:, 0]
        difference = np.linalg.norm(positions[1:]-positions[0], axis=1)/ARCSEC
        rows.append({"offset_days": h, "angular_difference_arcsec": difference.tolist(),
                     "difference_over_h_squared": (difference/h**2).tolist()})
    return {"case": case, "same_initial_observed_position_and_rate": True,
            "tested_hidden_axes": ["inverse_range", "fractional_range_rate"],
            "measurements": rows, "uniform_derivative_bound_certified": False}


def endpoint_experiment(case):
    """Find two states inside the ORIGINAL box with the same endpoint directions."""
    chart, center, widths, g, _ = setup(case)
    times = np.array([g.times[0], 0., g.times[-1]])
    geom = geometry(times, chart.axes)
    normalized_reference = np.array([0., 0., .35, 0., .25, .2])
    reference = center+widths*normalized_reference
    target = projected_chart(chart, reference[None], geom)[0]
    inner = np.array([2, 3, 4, 5])
    witnesses = []
    for delta in [.025, .05, .1, .2, .4, .6]:
        proposed = normalized_reference.copy()
        proposed[:2] += [delta, .7*delta]

        def residual(z):
            normalized = proposed.copy()
            normalized[inner] = z
            positions = projected_chart(chart, (center+widths*normalized)[None], geom)[0]
            return ((positions[[0, 2]]-target[[0, 2]])/ARCSEC).ravel()

        fit = least_squares(residual, normalized_reference[inner], bounds=(-.99, .99),
                            jac="3-point", xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=200)
        proposed[inner] = fit.x
        parameter = center+widths*proposed
        positions = projected_chart(chart, parameter[None], geom)[0]
        error = float(np.max(abs(positions[[0, 2]]-target[[0, 2]]))/ARCSEC)
        if error < 1e-7:
            witnesses.append({"reference_normalized": normalized_reference.tolist(),
                              "alternative_normalized": proposed.tolist(),
                              "reference_parameters": reference.tolist(),
                              "alternative_parameters": parameter.tolist(),
                              "endpoint_error_arcsec": error,
                              "midpoint_separation_arcsec": float(np.linalg.norm(positions[1]-target[1])/ARCSEC)})
    if not witnesses:
        raise RuntimeError(f"No endpoint ambiguity witness found for {case}")
    witness = max(witnesses, key=lambda row: row["midpoint_separation_arcsec"])
    # Independent all-conic propagator, not the compiled optimization oracle.
    physical = positions_from_states(chart.states(np.array([reference, witness["alternative_parameters"]])), times)
    witness.update(case=case, reception_times_days=times.tolist(),
                   independently_checked_endpoint_error_arcsec=float(np.max(abs(physical[0, [0, 2]]-physical[1, [0, 2]]))/ARCSEC),
                   independently_checked_midpoint_separation_arcsec=float(np.linalg.norm(physical[0, 1]-physical[1, 1])/ARCSEC),
                   both_inside_original_domain=True, numerical_witness_not_interval_existence_proof=True)
    return witness


def endpoint_rank_certificate(case):
    """Certify local endpoint ambiguity by the implicit-function theorem.

    Two endpoint directions have rank four. Adding a midpoint has rank six.
    Therefore a two-dimensional endpoint fiber exists locally, and the midpoint
    varies on it. This certifies existence, not the separately fitted finite
    numerical witness. The global speed proof identifies the physical branch
    of the narrow local root disks, which need not contain the reference epoch.
    """
    chart, center, widths, g, _ = setup(case)
    point = center+widths*np.array([0., 0., .35, 0., .25, .2])
    with ctx.workprec(128):
        physical = global_speed_certificate(chart, point, np.zeros(6))
        rows, records = [], []
        for time in [g.times[0], g.times[-1], 0.]:
            gradients, record = root_and_gradients(
                chart, point, np.zeros(6), float(time), 1e-12,
                rho=1., derivative_scales=widths)
            if not record["positive_real_retarded_root_derivative"]:
                raise ValueError("Local root is not simple")
            rows.extend([[z.real/arb(ARCSEC) for z in axis] for axis in gradients])
            records.append(record)
        minor = arb_mat([[row[j] for j in [2, 3, 4, 5]] for row in rows[:4]]).det()
        full = arb_mat(rows).det()
        if minor.contains(0) or full.contains(0):
            raise ValueError("Cannot certify the endpoint/midpoint ranks")
        interval = lambda value: [lower_float(value), upper_float(value)]
        return {"case": case, "physical_parameters": point.tolist(),
                "normalized_derivative_scales": widths.tolist(),
                "reception_times_days_row_order": [float(g.times[0]), float(g.times[-1]), 0.],
                "output_units": "tangent-plane arcseconds",
                "jacobian_intervals": [[interval(x) for x in row] for row in rows],
                "endpoint_minor_columns": [2, 3, 4, 5],
                "endpoint_minor_determinant_interval": interval(minor),
                "three_epoch_determinant_interval": interval(full),
                "endpoint_rank": 4, "three_epoch_rank": 6,
                "local_endpoint_fiber_dimension": 2,
                "midpoint_varies_on_endpoint_fiber_certified": True,
                "branch_identification": "global noncollision/subluminal proof and unique real retarded root",
                "global_physical_certificate": physical, "local_root_certificates": records,
                "does_not_rule_out_six_state_segment_algorithms": True,
                "independent_specialist_review_completed": False}


def inverse_certificate(case="near", *, order=32, total_span=1.):
    """Uniform near-null direction on an unchanged six-dimensional orbit box.

    Rebuild the complex-time enclosure; do not trust an archived numeric bound.
    The data vector has one nonzero spatial frequency at each DISTINCT epoch.
    Its phases cancel the central orbit exactly in the mathematical statement.
    """
    chart, center, widths, g, _ = setup(case)
    span = float(max(abs(g.times)))
    ellipse_center, ellipse_half, rho = -span/2, span/2, 2.
    if (not isinstance(order, int) or isinstance(order, bool) or not 1 <= order <= 64
            or not np.isfinite(total_span) or total_span <= 0
            or total_span/2 >= ellipse_half*(rho-1/rho)/2):
        raise ValueError("Invalid order or exposure interval outside the certified ellipse disk")
    started = perf_counter()
    with ctx.workprec(192):
        gradients, roots = partitioned_gradients(chart, center, widths, ellipse_center,
                                                 ellipse_half, rho, partition_depth=6)
        # Choose one physical frequency, below the previous campaign's maximum.
        frequency = np.array([.7, .4])/ARCSEC
        log_modulus = sum((abs(value.imag)*abs(arb(float(k)))
                           for axis, k in zip(gradients, frequency) for value in axis), arb(0))
        radius = arb(ellipse_half)*(arb(rho)-1/arb(rho))/2-arb(float(total_span))/2
        step = arb(float(total_span))/order
        normalization = arb(math.comb(2*order, order)).sqrt()
        bound = log_modulus.exp()*arb.fac_ui(order)*(step/radius)**order/normalization
        bound_upper = upper_float(bound)
        real_bound_upper = upper_float(arb(2).sqrt()*arb(bound_upper))
        result = {"case": case, "order": order, "distinct_exposures": order+1,
                  "exposure_span_days": total_span, "ellipse_center_days": ellipse_center,
                  "ellipse_half_width_days": ellipse_half, "ellipse_parameter": rho,
                  "cauchy_radius_days_lower": lower_float(radius),
                  "log_kernel_modulus_upper": upper_float(log_modulus),
                  "uniform_l2_input_to_sup_response_upper": bound_upper,
                  "inverse_lipschitz_lower_bound": lower_float(1/arb(bound_upper)),
                  "real_conjugate_symmetric_input_response_upper": real_bound_upper,
                  "real_input_inverse_lipschitz_lower_bound": lower_float(1/arb(real_bound_upper)),
                  "normalized_input_l2": 1, "zero_spatial_frequencies_used": False,
                  "frequency_radians_inverse": frequency.tolist(),
                  "root_center": center.tolist(), "root_widths": widths.tolist(),
                  "whole_original_domain": True, "root_enclosures": roots,
                  "certificate_seconds": perf_counter()-started,
                  "bound_is_exact_arithmetic": True,
                  "ordinary_float_input_vector_certified": False,
                  "novelty_claimed": False}
    # This check exposes the double-precision floor; it cannot verify a 1e-20 bound.
    times = ellipse_center+np.linspace(-total_span/2, total_span/2, order+1)
    rng = np.random.default_rng(20261026)
    parameters = center+rng.uniform(-1, 1, (256, 6))*widths
    geom = geometry(times, chart.axes)
    positions = projected_chart(chart, parameters, geom)
    central = projected_chart(chart, center[None], geom)[0]
    weights = finite_difference_vector(order)
    residual_phase = np.einsum("nti,i->nt", positions-central, frequency)
    response = np.exp(1j*residual_phase) @ weights
    result.update(float_sample_max_response=float(np.max(abs(response))),
                  float_evaluation_is_not_a_verification_of_smaller_exact_bound=True)
    return result


def real_pixel_composition():
    """Check the exact leaf identity against archived ZTF pixels and full WCS.

    This is deliberately a direct identity check, not a claimed fast pipeline.
    No observer ephemeris is interpolated between the recorded exposure epochs.
    """
    from astropy.io import fits
    from astropy.wcs import WCS
    from .detector_fields import direct_pixel_statistics
    from .observer_chart import ObserverChart
    from .real_exposure_experiment import tangent_wcs

    directory = Path(__file__).parent/"chart_results"
    manifest_path = directory/"real_inputs/observations.json"
    report_path = directory/"real/result.json"
    metadata = json.loads(manifest_path.read_text())
    previous = json.loads(report_path.read_text())["cover"]
    old = ObserverChart(**previous["chart"])
    images, variances, masks, charts, wcs_objects = [], [], [], [], []
    inputs = [manifest_path, report_path]
    for index, filename in enumerate(metadata["files"]):
        path = manifest_path.parent/filename
        inputs.append(path)
        with fits.open(path) as hdus:
            images.append(hdus["SCI"].data.astype(float))
            variances.append(hdus["VAR"].data.astype(float))
            masks.append(hdus["MASK"].data != 0)
            wcs = WCS(hdus["SCI"].header)
            normal, east, north, _ = tangent_wcs(wcs, 4., 4.)
        charts.append(ReceptionChart(metadata["observers"][index],
                                     metadata["observer_velocities"][index], normal, east, north))
        wcs_objects.append(wcs)
    images, variances, masks = np.array(images), np.array(variances), np.array(masks)
    times = np.asarray(metadata["mjd_tdb"])-metadata["mjd_tdb"][0]
    rng = np.random.default_rng(20261027)
    parameters = np.asarray(previous["center"])+rng.uniform(-.95, .95, (64, 6))*previous["widths"]
    states = old.states(parameters)
    q = charts[0].parameters(states)
    emitted, _ = apparent(states, times, np.asarray(metadata["observers"]))
    direct_delta = emitted[..., :3]-np.asarray(metadata["observers"])
    direct_positions, composed_positions = [], []

    def pixels(delta, wcs):
        ra = np.rad2deg(np.arctan2(delta[:, 1], delta[:, 0])) % 360
        dec = np.rad2deg(np.arctan2(delta[:, 2], np.hypot(delta[:, 0], delta[:, 1])))
        x, y = wcs.all_world2pix(ra, dec, 0)
        return np.column_stack((x-4., y-4.))

    for index, (time, chart, wcs) in enumerate(zip(times, charts, wcs_objects)):
        local = charts[0].rebase(q, time, chart)
        direction = chart.axes[0]+local[:, 3, None]*chart.axes[1]+local[:, 4, None]*chart.axes[2]
        direct_positions.append(pixels(direct_delta[:, index], wcs))
        composed_positions.append(pixels(direction, wcs))
    direct_positions = np.stack(direct_positions, axis=1)
    composed_positions = np.stack(composed_positions, axis=1)
    options = dict(masks=masks, profile_background=True, flux_scales=metadata["flux_scales"])
    direct = direct_pixel_statistics(images, variances, metadata["sigma_pixels"], direct_positions, **options)
    composed = direct_pixel_statistics(images, variances, metadata["sigma_pixels"], composed_positions, **options)
    shifted = direct_pixel_statistics(images+np.array([7., -11., 13.])[:, None, None],
                                     variances, metadata["sigma_pixels"], direct_positions, **options)
    return {"real_survey_data": True, "exposures": len(times), "queries": len(states),
            "pixels_per_exposure": int(np.prod(images.shape[1:])),
            "uses_full_recorded_TPV_WCS": True, "observer_interpolation_used": False,
            "observation_span_minutes": float(np.ptp(times)*1440),
            "position_identity_max_pixels": float(np.max(abs(direct_positions-composed_positions))),
            "numerator_identity_max_absolute": float(np.max(abs(direct[0]-composed[0]))),
            "information_identity_max_absolute": float(np.max(abs(direct[1]-composed[1]))),
            "profiled_background_shift_numerator_max_absolute": float(np.max(abs(direct[0]-shifted[0]))),
            "input_sha256": {str(path.relative_to(directory.parent.parent.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in inputs},
            "same_declared_gaussian_PSF_and_variance_approximations_as_original_experiment": True,
            "fast_transform_or_discovery_claimed": False}


def run(output):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    results = {"schema_version": 1, "full_objective_complete": False,
               "new_fast_transform_claimed": False, "real_survey_data": False}
    for name, action in [("composition", composition_experiment),
                         ("hidden_coordinate_scaling", hidden_coordinate_experiment),
                         ("endpoint_ambiguity", endpoint_experiment),
                         ("endpoint_rank_certificates", endpoint_rank_certificate)]:
        results[name] = []
        for case in ["near", "distant", "stress"]:
            value = action(case)
            results[name].append(value)
            print(name, case, "done", flush=True)
    results["inverse_certificates"] = []
    for order, span in [(16, .125), (32, 1.)]:
        results["inverse_certificates"].append(inverse_certificate(order=order, total_span=span))
        print("inverse", order, span, "done", flush=True)
    results["real_pixel_identity"] = real_pixel_composition()
    print("real ZTF pixel identity done", flush=True)
    results["conditional_counts"] = [
        conditional_tree_counts(b, depth)
        for b in [4, 16, 64] for depth in [4, 8, 12, 16]]
    directory = Path(__file__).parent
    sources = list(directory.glob("*.py")) + [Path(__file__).parents[2]/"src/magictools/orbits.py"]
    results["source_sha256"] = {str(p.relative_to(Path(__file__).parents[2])): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sources}
    (output/"results.json").write_text(json.dumps(results, indent=2, allow_nan=False)+"\n")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
