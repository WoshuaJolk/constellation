"""Validated likelihood envelopes retaining linear flux correlations.

A numerical quadratic-program solution only proposes a dual vector. Its bound
is independently evaluated in Arb and transfers to every continuous template
in the cell using the reverse triangle inequality.
"""

import numpy as np
from scipy.optimize import minimize
from flint import arb

from .arithmetic import square
from .integrate import _box


def covariance_quadratic(scene, vector):
    manifest = scene.manifest
    if "covariance" in manifest:
        c = manifest["covariance"]
        result = sum((a*arb(float(c[i][j]))*b for i, a in enumerate(vector)
                      for j, b in enumerate(vector)), arb(0))
    else:
        result = sum((square(a*arb(s)) for a, s in zip(vector, scene.sigmas)), arb(0))
    if "linear_nuisance" in manifest:
        model = manifest["linear_nuisance"]
        b, s = model["design"], model["covariance"]
        projected = [sum((arb(float(row[j]))*a for row, a in zip(b, vector)), arb(0)) for j in range(len(s))]
        result += sum((a*arb(float(s[i][j]))*v for i, a in enumerate(projected)
                       for j, v in enumerate(projected)), arb(0))
    return result


def bound(scene, root, path):
    """Return an Arb upper bound on the scaled Gaussian likelihood."""
    if scene.noise != "gaussian":
        return arb(1), {"status": "unsupported_poisson"}
    box = _box(root, path)
    whole = [lo.union(hi) for lo, hi in box]
    center = [(lo+hi)/2 for lo, hi in box]
    zero = {i: arb(0) for i in range(len(root["classes"]))}
    base = scene.means(root, center, source_fluxes=zero)
    background = scene.means(root, whole, source_fluxes=zero)
    r = [arb(y)-b for y, b in zip(scene.data, base)]
    wr = scene._whiten(r)
    templates, flux_bounds, errors = [], [], []
    background_error = scene._norm(scene._whiten([b-a for a, b in zip(base, background)])).upper().nonnegative_part().sqrt()
    for i in range(len(root["classes"])):
        point = scene.means(root, center, source_only=i, source_fluxes={i: arb(1)})
        interval = scene.means(root, whole, source_only=i, source_fluxes={i: arb(1)})
        templates.append(point)
        error = scene._norm(scene._whiten([b-a for a, b in zip(point, interval)])).upper().nonnegative_part().sqrt()
        key = f"s{i}.flux"
        limits = box[root["names"].index(key)] if key in root["names"] else [arb(root["fixed"][key])]*2
        flux_bounds.append(limits)
        errors.append(limits[1]*error)
    whitened = [scene._whiten(p) for p in templates]
    if templates:
        gram = np.array([[float(scene._inner(a, b)) for b in whitened] for a in whitened])
        score = np.array([float(scene._inner(p, wr)) for p in whitened])
        limits = [(float(lo), float(hi)) for lo, hi in flux_bounds]
        fit = minimize(lambda f: .5*f@gram@f-score@f, np.mean(limits, axis=1),
                       jac=lambda f: gram@f-score, bounds=limits, method="L-BFGS-B")
        flux = np.array(fit.x)
    else:
        flux = np.empty(0)
    residual = np.array(list(map(float, r)))
    for f, p in zip(flux, templates):
        residual -= f*np.array(list(map(float, p)))
    # This float vector is merely a proposal. Any finite vector is dual-feasible.
    if scene.cholesky is not None:
        chol = np.array([[float(x) for x in row] for row in scene.cholesky])
        proposed = np.linalg.solve(chol.T, np.linalg.solve(chol, residual))
    else:
        sigma = np.array(scene.sigmas)
        w = residual/sigma
        if scene.low_rank is not None:
            t, h = (np.array([[float(x) for x in row] for row in mat]) for mat in scene.low_rank)
            w -= t@np.linalg.solve(h.T, np.linalg.solve(h, t.T@w))
        proposed = w/sigma
    if not np.all(np.isfinite(proposed)):
        return arb(1), {"status": "nonfinite_proposal"}
    dual = list(map(lambda x: arb(float(x)), proposed))
    lower_squared = 2*sum((a*b for a, b in zip(dual, r)), arb(0))-covariance_quadratic(scene, dual)
    for p, (lo, hi) in zip(templates, flux_bounds):
        coefficient = sum((a*b for a, b in zip(dual, p)), arb(0))
        support = max((coefficient*lo).upper(), (coefficient*hi).upper())
        lower_squared -= 2*support
    distance = max(arb(0), lower_squared.lower()).sqrt()
    error = (background_error+sum(errors, arb(0))).upper()
    residual_lower = max(arb(0), (distance-error).lower())
    upper = (-square(residual_lower)/2).exp().upper()
    return min(arb(1), upper), {
        "status": "validated_dual_transfer",
        "reference_distance_lower": float(distance.lower()),
        "template_perturbation_upper": float(error),
        "physical_distance_lower": float(residual_lower),
        "optimizer_is_only_a_proposal": True,
    }
