"""Global Gaussian likelihood ceilings from continuous template support bounds.

For every vector z, residual energy is at least
    2 z.T y - z.T C z - 2 sup_theta z.T mean(theta).
An inexpensive convex relaxation proposes z. Arb then verifies the support
bound over the *continuous* source domain, independently of the optimizer.
Per-exposure relaxations discard physical constraints only in the direction
that enlarges the possible mean set. They cannot exclude a catalogue.
"""

import heapq
from itertools import product
from math import isfinite
from time import perf_counter

import numpy as np
from flint import arb, ctx
from scipy.linalg import solve_triangular, orth
from scipy.special import ndtr

from .arithmetic import lower, upper, square
from .envelope import covariance_quadratic
from .integrate import _box, _canonical
import hashlib
import copy


def geometry(scene, cls, exposure):
    """Relax source coordinates to a detector-space x,y,sigma_x,sigma_y box."""
    source = scene.types[cls]
    if source["kind"] == "kepler":
        raise ValueError("Continuous orbital detector-domain bounds are not implemented here")
    p = lambda k, default=0: arb(source["parameters"][k][0]).union(
        arb(source["parameters"][k][1])) if k in source["parameters"] else arb(default)
    camera = exposure.get("instrument", "default")
    n = lambda k: arb(scene.nuisance[camera+"_"+k][0]).union(
        arb(scene.nuisance[camera+"_"+k][1])) if camera+"_"+k in scene.nuisance else arb(0)
    t = arb(float(exposure["time"]))
    x = p("x")+p("vx")*t+p("parallax")*arb(float(exposure.get("parallax_x", 0)))
    y = p("y")+p("vy")*t+p("parallax")*arb(float(exposure.get("parallax_y", 0)))
    a, b = exposure.get("affine", [[1, 0], [0, 1]]), exposure.get("offset", [0, 0])
    x, y = (arb(float(a[0][0]))*x+arb(float(a[0][1]))*y+arb(float(b[0]))+n("dx"),
            arb(float(a[1][0]))*x+arb(float(a[1][1]))*y+arb(float(b[1]))+n("dy"))
    psf = arb(float(exposure["psf_sigma"]))*n("log_psf").exp()
    sx = (square(psf)+square(p("sigma_x"))).sqrt() if source["kind"] == "galaxy" else psf
    sy = (square(psf)+square(p("sigma_y"))).sqrt() if source["kind"] == "galaxy" else psf
    amplitude = p("flux")*arb(float(exposure.get("flux_scale", 1)))*(
        n("log_gain")+p("color")*arb(float(exposure.get("band_coordinate", 0)))+p("variability")*t).exp()
    if exposure.get("flux_scale", 1) < 0 or not sx > 0 or not sy > 0:
        raise ValueError("Nonnegative amplitude and strictly positive widths required")
    amplitude = amplitude.nonnegative_part()
    # Outward binary endpoints let the certificate reconstruct the same cover.
    def endpoints(v):
        f = float(v.mid())
        return [f, f] if v == arb(f) else [lower(v), upper(v)]
    return {"names": ["x", "y", "sigma_x", "sigma_y"],
            "ranges": [endpoints(v) for v in (x, y, sx, sy)]}, amplitude.upper()


def _profiles(coordinates, position, width):
    """Pixel probability and its second position/width derivatives."""
    result = []
    norm = (2*arb.pi()).sqrt()
    for c in coordinates:
        a, b = (arb(c)-arb(".5")-position)/width, (arb(c)+arb(".5")-position)/width
        pa, pb = (-square(a)/2).exp()/norm, (-square(b)/2).exp()/norm
        value = ((b/arb(2).sqrt()).erf()-(a/arb(2).sqrt()).erf())/2
        dx2 = (a*pa-b*pb)/square(width)
        ds2 = ((a**3-2*a)*pa-(b**3-2*b)*pb)/square(width)
        result.append((value.nonnegative_part(), dx2, ds2))
    return result


def _response(exposure, active, dual, coordinates, derivatives=False):
    shape = np.asarray(exposure["data"]).shape
    xs = sorted({float(i-(shape[1]-1)/2) for j, i in active})
    ys = sorted({float(j-(shape[0]-1)/2) for j, i in active})
    x, y, sx, sy = coordinates
    px, py = dict(zip(xs, _profiles(xs, x, sx))), dict(zip(ys, _profiles(ys, y, sy)))
    result = [arb(0) for _ in range(5)]
    for (j, i), z in zip(active, dual):
        a, b = px[float(i-(shape[1]-1)/2)], py[float(j-(shape[0]-1)/2)]
        result[0] += z*a[0]*b[0]
        if derivatives:
            result[1] += z*a[1]*b[0]
            result[2] += z*a[0]*b[1]
            result[3] += z*a[2]*b[0]
            result[4] += z*a[0]*b[2]
    return result


def _cell(exposure, active, dual, domain, path):
    box = _box(domain, path)
    whole = [a.union(b) for a, b in box]
    values = _response(exposure, active, dual, whole, True)
    axes = [i for i, (a, b) in enumerate(box) if a != b]
    # Successive linear interpolation in each coordinate is a positive
    # contraction. Its supremum error is bounded by sum(h_i^2 M_ii / 8).
    corners = []
    for bits in product((0, 1), repeat=len(axes)):
        p = [a for a, b in box]
        for i, bit in zip(axes, bits):
            p[i] = box[i][bit]
        corners.append(_response(exposure, active, dual, p)[0])
    errors = [square(b-a)*v.abs_upper()/8 for (a, b), v in zip(box, values[1:])]
    bound = min(values[0].upper(),
                (max(v.upper() for v in corners)+sum(errors, arb(0))).upper())
    sample = max(v.lower() for v in corners)
    axis = max(axes, key=lambda i: float(errors[i])) if axes else None
    return bound, sample, axis


def support(exposure, active, dual, domain, *, max_leaves=512, tolerance=1e-5):
    """Cover the full domain; return a rigorous supremum upper bound."""
    heap, records, serial = [], {}, 0
    best = arb("-inf")

    def add(path):
        nonlocal serial, best
        hi, sample, axis = _cell(exposure, active, dual, domain, path)
        best = max(best, sample)
        records[serial] = {"path": path, "upper": upper(hi), "axis": axis}
        heapq.heappush(heap, (-upper(hi), serial))
        serial += 1

    add([])
    while len(records) < max_leaves:
        top, index = heap[0]
        if -top <= max(0., float(best))+tolerance:
            break
        record = records[index]
        if record["axis"] is None:
            break
        heapq.heappop(heap)
        del records[index]
        for side in (0, 1):
            add(record["path"]+[[record["axis"], side]])
    return {"upper": max(0., -heap[0][0]), "sample_lower": float(best),
            "leaves": [{"path": r["path"], "upper": r["upper"]} for r in records.values()]}


def _covariance(scene):
    if "covariance" in scene.manifest:
        c = np.array(scene.manifest["covariance"], float)
    else:
        c = np.diag(np.array(scene.sigmas)**2)
    if "linear_nuisance" in scene.manifest:
        n = scene.manifest["linear_nuisance"]
        b = np.array(n["design"])
        c += b@np.array(n["covariance"])@b.T
    return c


def _simplex(v, cap):
    if cap == 0:
        return np.zeros_like(v)
    q = np.maximum(v, 0)
    if q.sum() <= cap:
        return q
    u = np.sort(v)[::-1]
    css = np.cumsum(u)-cap
    eligible = np.flatnonzero(u-css/np.arange(1, len(u)+1) > 0)
    k = eligible[-1]
    return np.maximum(v-css[k]/(k+1), 0)


def propose(scene, domains, amplitudes, *, grid=9, iterations=1000):
    """Capped nonnegative template mixtures; no optimization accuracy is assumed."""
    n = len(scene.data)
    matrix, slices, caps, baseline, nuisance = [], [], [], np.zeros(n), []
    offset, column = 0, 0
    prior = scene.manifest["count_prior"]
    maximum = prior.get("max_count", len(prior.get("probabilities", []))-1)
    for eindex, (e, active) in enumerate(zip(scene.exposures, scene.active)):
        shape = np.asarray(e["data"]).shape
        xx = np.array([i-(shape[1]-1)/2 for j, i in active])
        yy = np.array([j-(shape[0]-1)/2 for j, i in active])
        key = e.get("instrument", "default")+"_background"
        if key in scene.nuisance:
            baseline[offset:offset+len(active)] = np.mean(scene.nuisance[key])
            v = np.zeros(n)
            v[offset:offset+len(active)] = 1
            nuisance.append(v)  # relaxing shared backgrounds independently is safe
        else:
            baseline[offset:offset+len(active)] = e.get("background", 0)
        count = 0
        for cls in range(len(scene.types)):
            domain = domains[cls][eindex]["ranges"]
            axes = [np.linspace(a, b, grid if i < 2 else 3) if b-a > 1e-12 else [(a+b)/2]
                    for i, (a, b) in enumerate(domain)]
            points = np.array(list(product(*axes)))
            x, y, sx, sy = points.T
            a = (ndtr((xx[:, None]+.5-x)/sx)-ndtr((xx[:, None]-.5-x)/sx))*(
                ndtr((yy[:, None]+.5-y)/sy)-ndtr((yy[:, None]-.5-y)/sy))
            full = np.zeros((n, len(points)))
            full[offset:offset+len(active)] = a
            matrix.append(full)
            count += len(points)
        slices.append(slice(column, column+count))
        caps.append(maximum*max(float(amplitudes[c][eindex]) for c in range(len(scene.types))))
        column += count
        offset += len(active)
    a = np.concatenate(matrix, axis=1)
    chol = np.linalg.cholesky(_covariance(scene))
    a = solve_triangular(chol, a, lower=True)
    y = solve_triangular(chol, np.array(list(map(float, scene.data)))-baseline, lower=True)
    if nuisance:
        basis = orth(solve_triangular(chol, np.array(nuisance).T, lower=True))
        a -= basis@(basis.T@a)
        y -= basis@(basis.T@y)
    lipschitz = np.linalg.norm(a, 2)**2
    if lipschitz == 0:
        return solve_triangular(chol.T, y, lower=False).tolist()
    f, v, t = np.zeros(a.shape[1]), np.zeros(a.shape[1]), 1.
    for _ in range(iterations):
        candidate = v-a.T@(a@v-y)/lipschitz
        new = np.concatenate([_simplex(candidate[s], cap) for s, cap in zip(slices, caps)])
        tn = (1+np.sqrt(1+4*t*t))/2
        v, f, t = new+(t-1)/tn*(new-f), new, tn
    return solve_triangular(chol.T, y-a@f, lower=False).tolist()


def _terms(scene, dual, supports, amplitudes):
    background, offset = arb(0), 0
    for e, active in zip(scene.exposures, scene.active):
        total = sum(dual[offset:offset+len(active)], arb(0))
        key = e.get("instrument", "default")+"_background"
        if key in scene.nuisance:
            a, b = scene.nuisance[key]
            background += max((total*arb(a)).upper(), (total*arb(b)).upper())
        else:
            background += total*arb(float(e.get("background", 0)))
        offset += len(active)
    # y already has the integrated Gaussian nuisance prior mean removed.
    baseline = (2*sum((z*arb(y) for z, y in zip(dual, scene.data)), arb(0))
                -covariance_quadratic(scene, dual)-2*background).lower()
    classes = [sum((arb(r["upper"])*a for r, a in zip(row, amps)), arb(0)).upper()
               for row, amps in zip(supports, amplitudes)]
    return baseline, classes


def build(scene, *, max_leaves=512, tolerance=1e-5, grid=9, iterations=1000):
    started = perf_counter()
    if scene.noise != "gaussian":
        raise ValueError("Noise floor requires Gaussian measurements")
    if (not isinstance(max_leaves, int) or max_leaves < 1
            or not isfinite(tolerance) or tolerance <= 0 or grid < 2 or iterations < 1):
        raise ValueError("Positive support and proposal budgets required")
    with ctx.workprec(100):
        pairs = [[geometry(scene, c, e) for e in scene.exposures] for c in range(len(scene.types))]
        domains = [[a for a, b in row] for row in pairs]
        amplitudes = [[b for a, b in row] for row in pairs]
        proposed = propose(scene, domains, amplitudes, grid=grid, iterations=iterations)
        if not all(isfinite(x) for x in proposed):
            raise ArithmeticError("Nonfinite dual proposal")
        dual = [arb(x) for x in proposed]
        supports = []
        for row in domains:
            offset, results = 0, []
            for e, active, domain in zip(scene.exposures, scene.active, row):
                results.append(support(e, active, dual[offset:offset+len(active)], domain,
                                       max_leaves=max_leaves, tolerance=tolerance))
                offset += len(active)
            supports.append(results)
        baseline, classes = _terms(scene, dual, supports, amplitudes)
        return {"schema_version": 1, "method": "continuous_template_dual_noise_floor",
                "manifest_sha256": hashlib.sha256(_canonical(scene.manifest)).hexdigest(),
                "precision": 100, "dual": proposed, "supports": supports,
                "baseline_energy_lower": lower(baseline), "class_support_upper": [upper(v) for v in classes],
                "settings": {"max_leaves": max_leaves, "tolerance": tolerance,
                             "grid": grid, "iterations": iterations},
                "seconds": perf_counter()-started, "full_objective_complete": False}


def ceiling(certificate, root):
    """Pointwise scaled likelihood ceiling for an entire catalogue class root."""
    exponent = -arb(certificate["baseline_energy_lower"])/2+sum(
        (arb(certificate["class_support_upper"][c]) for c in root["classes"]), arb(0))
    return min(arb(1), exponent.exp().upper())


def tail_ceiling(scene, certificate):
    """Poisson exponential tilting bounds all unrepresented catalogue counts."""
    scene.roots()  # initializes the original prior tail
    if "poisson_mean" not in scene.manifest["count_prior"]:
        return arb(0)
    prior = scene.manifest["count_prior"]
    mean = arb(float(prior["poisson_mean"]))
    weights = [arb(float(t.get("weight", 1))) for t in scene.types]
    total = sum(weights, arb(0))
    mgf = sum((w*arb(h).exp() for w, h in zip(weights, certificate["class_support_upper"])), arb(0))/total
    tilted = mean*mgf
    probability, cumulative = (-tilted).exp(), (-tilted).exp()
    for k in range(1, prior["max_count"]+1):
        probability *= tilted/k
        cumulative += probability
    tail = (arb(1)-cumulative).nonnegative_part()
    bound = (-arb(certificate["baseline_energy_lower"])/2+mean*(mgf-1)).exp()*tail
    return min(scene.tail.upper(), bound.upper())


def verify(scene, certificate):
    """Replay support enclosures and independently check complete partitions."""
    if (certificate.get("schema_version") != 1 or certificate.get("method") != "continuous_template_dual_noise_floor"
            or certificate.get("manifest_sha256") != hashlib.sha256(_canonical(scene.manifest)).hexdigest()
            or certificate.get("full_objective_complete") is not False):
        raise ValueError("Noise-floor certificate identity mismatch")
    with ctx.workprec(certificate["precision"]):
        if len(certificate["dual"]) != len(scene.data) or not all(isfinite(x) for x in certificate["dual"]):
            raise ValueError("Invalid dual vector")
        dual = [arb(v) for v in certificate["dual"]]
        supports = certificate["supports"]
        if len(supports) != len(scene.types):
            raise ValueError("Missing source-class support")
        amplitudes = []
        for cls, row in enumerate(supports):
            if len(row) != len(scene.exposures):
                raise ValueError("Missing exposure support")
            offset, amps = 0, []
            for e, active, record in zip(scene.exposures, scene.active, row):
                domain, amp = geometry(scene, cls, e)
                amps.append(amp)
                tree = {}
                largest = arb("-inf")
                for leaf in record["leaves"]:
                    node = tree
                    for axis, side in leaf["path"]:
                        if not isinstance(axis, int) or not 0 <= axis < 4 or side not in (0, 1):
                            raise ValueError("Invalid support path")
                        if "leaf" in node or ("axis" in node and node["axis"] != axis):
                            raise ValueError("Overlapping support cover")
                        node["axis"] = axis
                        node = node.setdefault(side, {})
                    if node:
                        raise ValueError("Duplicate support leaf")
                    node["leaf"] = True
                    hi, _, _ = _cell(e, active, dual[offset:offset+len(active)], domain, leaf["path"])
                    if not isfinite(leaf["upper"]) or not arb(leaf["upper"]) >= hi:
                        raise ValueError("Support leaf fails replay")
                    largest = max(largest, arb(leaf["upper"]))
                stack = [tree]
                while stack:
                    node = stack.pop()
                    if "leaf" not in node:
                        if 0 not in node or 1 not in node:
                            raise ValueError("Incomplete support partition")
                        stack.extend([node[0], node[1]])
                if not isfinite(record["upper"]) or not arb(record["upper"]) >= max(arb(0), largest):
                    raise ValueError("Invalid aggregate support")
                offset += len(active)
            amplitudes.append(amps)
        baseline, classes = _terms(scene, dual, supports, amplitudes)
        if (not isfinite(certificate["baseline_energy_lower"])
                or not arb(certificate["baseline_energy_lower"]) <= baseline):
            raise ValueError("Invalid baseline residual bound")
        if len(classes) != len(certificate["class_support_upper"]) or any(
                not isfinite(saved) or not arb(saved) >= true
                for saved, true in zip(certificate["class_support_upper"], classes)):
            raise ValueError("Invalid class support aggregation")
    return {"valid": True, "pointwise": True, "full_objective_complete": False}


def conditional_scene(scene, root, path):
    """Reuse the observation covariance while restricting only a prior cell."""
    result = copy.copy(scene)
    box = _box(root, path)
    values = {**{key: [value, value] for key, value in root["fixed"].items()},
              **{key: [lower(a), upper(b)] for key, (a, b) in zip(root["names"], box)}}
    # Outward endpoints can extend a [0,...] flux/size interval below zero.
    # These constraints are already part of the original source prior.
    result.types = []
    for i, cls in enumerate(root["classes"]):
        source = copy.deepcopy(scene.types[cls])
        source["name"] = f"source{i}"
        source["parameters"] = {key: values[f"s{i}.{key}"] for key in source["parameters"]}
        for key in ("flux", "sigma_x", "sigma_y"):
            if key in source["parameters"]:
                source["parameters"][key][0] = max(0., source["parameters"][key][0])
        result.types.append(source)
    result.nuisance = {key: values["instrument."+key] for key in scene.nuisance}
    count = len(result.types)
    result.manifest = {**scene.manifest, "source_types": result.types, "nuisance": result.nuisance,
                       "count_prior": {"probabilities": [0.]*count+[1.]}}
    return result


def build_cover(scene, cover, *, max_leaves=16, tolerance=1e-5, grid=3, iterations=100):
    """A pointwise proof for every nonempty cell of an existing full partition."""
    from .integrate import verify as verify_partition
    started = perf_counter()
    if cover["manifest"] != scene.manifest:
        raise ValueError("Noise cover belongs to another scene")
    verify_partition(cover, replay=False)
    proofs = []
    with ctx.workprec(100):
        roots = scene.roots()
        for leaf in cover["certificate"]:
            root = roots[leaf["root"]]
            if not root["classes"]:
                proofs.append(None)
                continue
            local = conditional_scene(scene, root, leaf["path"])
            proof = build(local, max_leaves=max_leaves, tolerance=tolerance, grid=grid, iterations=iterations)
            proofs.append(proof)
    return {"schema_version": 1, "method": "conditional_continuous_noise_floors",
            "cover_sha256": hashlib.sha256(_canonical(cover)).hexdigest(), "proofs": proofs,
            "seconds": perf_counter()-started, "full_objective_complete": False}


def verify_cover(scene, cover, certificate):
    from .integrate import verify as verify_partition
    if (cover["manifest"] != scene.manifest or certificate.get("schema_version") != 1
            or certificate.get("method") != "conditional_continuous_noise_floors"
            or certificate.get("full_objective_complete") is not False
            or certificate.get("cover_sha256") != hashlib.sha256(_canonical(cover)).hexdigest()
            or len(certificate["proofs"]) != len(cover["certificate"])):
        raise ValueError("Conditional noise cover mismatch")
    verify_partition(cover, replay=False)
    with ctx.workprec(100):
        roots, ceilings = scene.roots(), []
        for leaf, proof in zip(cover["certificate"], certificate["proofs"]):
            root = roots[leaf["root"]]
            if not root["classes"]:
                if proof is not None:
                    raise ValueError("Unexpected empty-catalogue proof")
                ceilings.append(arb(1))
                continue
            local = conditional_scene(scene, root, leaf["path"])
            verify(local, proof)
            ceilings.append(ceiling(proof, {"classes": list(range(len(root["classes"])))}))
    return ceilings
