"""Constructive domains for one-dimensional factorizations of Kepler flow.

The certificate uses interval derivatives over *whole* boxes, not sampled
Jacobian nonsingularity. It supplies a common domain on which every partial
coordinate inverse exists, is unique, and stays inside the child field.
The ordinary floating-point cascade implementation remains uncertified.
"""

from time import perf_counter

import numpy as np
from flint import acb, arb

from research.detection_limit.interval_orbits import upper_float
from .reception_flow import _union, box_balls, enclose
from .variational_flow import flow_jacobian


def _upper_vector(values):
    return np.array([upper_float(x) for x in values])


def _absolute_matrix(matrix):
    return [[abs(matrix[i, j]).upper() for j in range(6)] for i in range(6)]


def _majorize(error, matrix, radii):
    return [error[i]+sum(matrix[i][j]*arb(float(radii[j])) for j in range(6))
            for i in range(6)]


def _weighted_norm(matrix, radii):
    return max(sum(matrix[i][j]*arb(float(radii[j])) for j in range(6))
               /arb(float(radii[i])) for i in range(6)).upper()


def contraction_weights(matrix):
    """Propose weights in floats, then certify their norm in ball arithmetic."""
    dimension = len(matrix)
    upper = np.array([[upper_float(x) for x in row] for row in matrix])
    try:
        weights = np.linalg.solve(np.eye(dimension)-upper, np.ones(dimension))
    except np.linalg.LinAlgError as error:
        raise ValueError("Contraction preconditioner is singular") from error
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("Positive contraction weights not found")
    bound = max(sum(matrix[i][j]*float(weights[j]) for j in range(dimension))
                /float(weights[i]) for i in range(dimension)).upper()
    return weights, bound


def _state_box(center, widths, radii, imaginary=None):
    return [(acb(arb(float(c), (arb(float(w))*float(a)).upper()),
                 arb(0, (arb(float(w))*float(v)).upper()))
             if imaginary is not None else arb(float(c), (arb(float(w))*float(a)).upper()))
            for c, w, a, v in zip(center, widths, radii,
                                  np.zeros(6) if imaginary is None else imaginary)]


def certify_domain(observer, center, widths, start, stop, *, max_step=.03125,
                   max_iterations=24, inflation=1.08, complex_ellipse=None,
                   max_range_pieces=16):
    """Construct a padded child box and certify all 64 coordinate hybrids.

    With normalized z, R(c+Wz)=c_child+W(z+d(z)). On a larger box |z_i|<=a_i,
    |d|<=delta and |Dd|<=E are enclosed. The verified inequalities
        1+2*delta < a,  ||diag(a)^-1 E diag(a)||_infty < 1
    make x_unknown=y_unknown-d_unknown(x) a contraction for every hybrid of
    parent [-1,1] and child [-1-delta,1+delta] coordinates. No energy-sign
    restriction or sampling is used.

    If complex_ellipse>1, additionally certify inverse holomorphy over complex
    rectangles containing the Bernstein ellipses of *every* hybrid grid. This
    supplies an interpolation prerequisite, not a complete error certificate.
    """
    begin = perf_counter()
    # Validate dimensions, widths and finiteness using the shared constructor.
    box_balls(center, widths)
    center, widths = np.asarray(center, float), np.asarray(widths, float)
    if not np.isfinite(inflation) or inflation <= 1 or max_iterations < 1:
        raise ValueError("Inflation must exceed one and iterations must be positive")
    if not isinstance(max_range_pieces, int) or max_range_pieces < 1:
        raise ValueError("A positive integer range subdivision budget is required")
    if complex_ellipse is not None and (
            not np.isfinite(complex_ellipse) or complex_ellipse <= 1):
        raise ValueError("Complex Bernstein ellipse must exceed one")
    point, _, point_work = enclose(list(map(arb, center)), start, stop, observer,
                                  max_step=max_step)
    child_center = np.array([float(x.mid()) for x in point])
    origin_error = [abs((x-arb(float(c)))/float(w)).upper()
                    for x, c, w in zip(point, child_center, widths)]
    radii = np.full(6, 1.02)
    work = {"point_flow_steps": point_work["steps"], "derivative_flow_steps": 0,
            "derivative_evaluations": 0, "domain_trials": 0,
            "derivative_flow_calls": 0, "refused_derivative_flow_calls": 0,
            "largest_range_partition": 1}

    def attempt(a, imaginary=None):
        initial = _state_box(center, widths, a, imaginary)
        inverse_range = initial[0].real if imaginary is not None else initial[0]
        if not inverse_range > 0:
            raise ValueError("Enclosing inverse range crosses zero; subdivide the parent box")
        pieces = 1
        while True:
            increment = None
            try:
                for part in range(pieces):
                    values = initial.copy()
                    if pieces > 1:
                        # Cover the *whole* enclosing range interval, including
                        # overhang. No orbit cell is dropped upon subdivision.
                        source = initial[0].real if imaginary is not None else initial[0]
                        real = source.mid()+source.rad()*arb(
                            -1+(2*part+1)/pieces, 1/pieces)
                        values[0] = acb(real, initial[0].imag) if imaginary is not None else real
                    work["derivative_flow_calls"] += 1
                    _, local, cost = flow_jacobian(
                        values, start, stop, observer, widths, max_step=max_step)
                    work["derivative_flow_steps"] += cost["state_steps"]
                    work["derivative_evaluations"] += cost["jacobian_evaluations"]
                    if increment is None:
                        increment = local
                    else:
                        for i in range(6):
                            for j in range(6):
                                increment[i, j] = _union(increment[i, j], local[i, j])
                break
            except ValueError:
                work["refused_derivative_flow_calls"] += 1
                if pieces*2 > max_range_pieces:
                    raise
                pieces *= 2
        work["largest_range_partition"] = max(work["largest_range_partition"], pieces)
        work["domain_trials"] += 1
        matrix = _absolute_matrix(increment)
        delta = _majorize(origin_error, matrix, a)
        return matrix, delta, increment

    for _ in range(max_iterations):
        matrix, delta, _ = attempt(radii)
        required = [1+2*d for d in delta]
        contraction = _weighted_norm(matrix, radii)
        if (all(arb(float(a)) > r for a, r in zip(radii, required))
                and contraction < 1):
            break
        radii = 1+inflation*(_upper_vector(required)-1)+1e-10
    else:
        raise ValueError("Whole-domain padding iteration needs a shorter segment or spatial subdivision")
    # Outward float widths used by numerical consumers need their own proof.
    child_widths = _upper_vector([arb(float(w))*(1+d) for w, d in zip(widths, delta)])
    b = [arb(float(cw))/float(w) for cw, w in zip(child_widths, widths)]
    # Numerical outward conversion enlarged b; verify the actual returned box.
    assert all(arb(float(a)) > bb+d for a, bb, d in zip(radii, b, delta))
    report = {
        "parent_center": center.tolist(), "parent_widths": widths.tolist(),
        "child_center": child_center.tolist(), "child_widths": child_widths.tolist(),
        "start": float(start), "stop": float(stop),
        "normalized_enclosing_radii": radii.tolist(),
        "normalized_displacement_bounds": _upper_vector(delta).tolist(),
        "normalized_center_error_bounds": _upper_vector(origin_error).tolist(),
        "derivative_increment_absolute_bounds":
            [[upper_float(x) for x in row] for row in matrix],
        "weighted_contraction_bound": upper_float(contraction),
        "all_coordinate_hybrids_certified": True,
        "all_real_intermediate_images_in_child_box": True,
        "uniqueness_scope": "enclosing normalized box; not arbitrary distant roots",
        "no_energy_sign_restriction": True,
        "complex_inverse_certified": False,
        "floating_point_evaluation_certified": False,
        "full_objective_complete": False,
    }
    if complex_ellipse is not None:
        ellipse = arb(float(complex_ellipse))
        semi_major = (ellipse+1/ellipse)/2
        semi_minor = (ellipse-1/ellipse)/2
        base = [semi_major*bb for bb in b]
        imag_base = [semi_minor*bb for bb in b]
        complex_radii = _upper_vector(base)+inflation*(radii-1)+1e-8
        imaginary_radii = _upper_vector(imag_base)*inflation+1e-8
        for _ in range(max_iterations):
            # d(x+i*y)=d(x)+integral_0^1 Dd(x+i*s*y) i*y ds.
            # Preserve the real axis: an absolute complex-radius bound would
            # gratuitously extend imaginary range by a full real box width.
            _, real_delta, _ = attempt(complex_radii)
            _, _, cincrement = attempt(complex_radii, imaginary_radii)
            er = [[abs(cincrement[i, j].real).upper() for j in range(6)] for i in range(6)]
            ei = [[abs(cincrement[i, j].imag).upper() for j in range(6)] for i in range(6)]
            real_required = [
                base[i]+real_delta[i]+sum(ei[i][j]*float(imaginary_radii[j]) for j in range(6))
                for i in range(6)]
            imag_required = [
                imag_base[i]+sum(er[i][j]*float(imaginary_radii[j]) for j in range(6))
                for i in range(6)]
            # Real representation of the complex derivative; weighted norm
            # proves contraction on the entire product of real/imag intervals.
            block = [er[i]+ei[i] for i in range(6)]+[ei[i]+er[i] for i in range(6)]
            cweights, ccontraction = contraction_weights(block)
            if (all(arb(float(a)) > r for a, r in zip(complex_radii, real_required))
                    and all(arb(float(a)) > r for a, r in zip(imaginary_radii, imag_required))
                    and ccontraction < 1):
                break
            complex_radii = _upper_vector(base)+inflation*(
                _upper_vector(real_required)-_upper_vector(base))+1e-8
            imaginary_radii = _upper_vector(imag_base)+inflation*(
                _upper_vector(imag_required)-_upper_vector(imag_base))+1e-8
        else:
            raise ValueError("Complex inverse cover needs a shorter segment or subdivision")
        report.update({
            "complex_inverse_certified": True, "complex_ellipse": complex_ellipse,
            "complex_normalized_enclosing_radii": complex_radii.tolist(),
            "complex_normalized_imaginary_radii": imaginary_radii.tolist(),
            "complex_weighted_contraction_bound": upper_float(ccontraction),
            "complex_contraction_weights": cweights.tolist(),
        })
    report.update(work)
    report["certificate_seconds"] = perf_counter()-begin
    return report


def certify_atlas(observer, center, widths, start, stop, *, axes=(0, 1),
                  max_leaves=64, **options):
    """Cover the entire original box by certified transfer patches.

    Failed patches are bisected with outward-rounded covers, never omitted.
    The partition is geometric; no conic-energy class or observation is
    filtered. A budget failure returns no partial success certificate.
    """
    box_balls(center, widths)
    if (not axes or len(set(axes)) != len(axes) or any(a not in range(6) for a in axes)
            or not isinstance(max_leaves, int) or max_leaves < 1):
        raise ValueError("Distinct subdivision axes and a positive leaf budget required")
    start_time = perf_counter()
    queue = [(np.asarray(center, float), np.asarray(widths, float), 0)]
    accepted, failed = [], []
    while queue:
        c, w, depth = queue.pop()
        trial_start = perf_counter()
        try:
            proof = certify_domain(observer, c, w, start, stop, **options)
            accepted.append(proof)
        except ValueError as error:
            failed.append({"depth": depth, "reason": str(error),
                           "seconds": perf_counter()-trial_start})
            if len(accepted)+len(queue)+2 > max_leaves:
                raise ValueError("Full-coverage atlas exceeds its leaf budget; no partial cover returned") from error
            axis = axes[depth % len(axes)]
            half = arb(float(w[axis]))/2
            for sign in [-1, 1]:
                exact_center = arb(float(c[axis]))+sign*half
                nc, nw = c.copy(), w.copy()
                nc[axis] = float(exact_center.mid())
                nw[axis] = upper_float(half+abs(exact_center-arb(float(nc[axis]))))
                queue.append((nc, nw, depth+1))
    return {
        "original_center": np.asarray(center).tolist(),
        "original_widths": np.asarray(widths).tolist(),
        "start": float(start), "stop": float(stop),
        "leaves": accepted, "leaf_count": len(accepted), "refused_parent_trials": failed,
        "total_seconds_including_refusals": perf_counter()-start_time,
        "whole_original_box_covered_by_outward_bisections": True,
        "all_coordinate_hybrids_certified_on_each_patch": True,
        "complex_inverse_certified_on_each_patch":
            all(p["complex_inverse_certified"] for p in accepted),
        "full_objective_complete": False,
    }


def solve_hybrid(report, hybrid, fixed, rebase, *, tolerance=1e-11, max_iterations=100):
    """Numerical constructive fixed-point solve using the certified box.

    Returns physical parent coordinates. The certificate concerns the exact
    map, not roundoff in the supplied rebase callback. Checked residuals below
    are floating-point diagnostics; they must not be called validated errors.
    """
    fixed = tuple(fixed)
    if len(set(fixed)) != len(fixed) or any(a not in range(6) for a in fixed):
        raise ValueError("Fixed axes must be distinct coordinate indices")
    if not np.isfinite(tolerance) or tolerance <= 0 or max_iterations < 1:
        raise ValueError("Positive tolerance and iteration budget required")
    hybrid = np.asarray(hybrid, float)
    if hybrid.ndim != 2 or hybrid.shape[1] != 6 or not np.all(np.isfinite(hybrid)):
        raise ValueError("Finite (N,6) hybrid coordinates required")
    unknown = [a for a in range(6) if a not in fixed]
    c, w = np.array(report["parent_center"]), np.array(report["parent_widths"])
    cc, cw = np.array(report["child_center"]), np.array(report["child_widths"])
    radii = np.array(report["normalized_enclosing_radii"])
    targets = (hybrid-cc)/w
    x = np.zeros_like(hybrid)
    x[:, fixed] = (hybrid[:, fixed]-c[list(fixed)])/w[list(fixed)]
    if (np.any(abs(x[:, fixed]) > 1)
            or np.any(abs(targets[:, unknown]) > (cw/w)[unknown])):
        raise ValueError("Hybrid query leaves the certified domain")
    x[:, unknown] = targets[:, unknown]
    for iteration in range(max_iterations):
        mapped = (rebase(c+x*w)-cc)/w
        residual = mapped[:, unknown]-targets[:, unknown]
        maximum = float(np.max(abs(residual))) if residual.size else 0.
        if maximum <= tolerance:
            return c+x*w, {"iterations": iteration+1, "scaled_residual": maximum,
                           "floating_point_error_certified": False}
        x[:, unknown] -= residual
        if np.any(abs(x) > radii):
            raise ValueError("Numerical iteration left its certified enclosing box")
    raise ValueError("Numerical hybrid solve exceeded its iteration budget")
