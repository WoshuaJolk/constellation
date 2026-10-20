"""Bounded-weight global posterior integration with conditional confidence.

This route does not use sampler convergence as evidence of mode coverage.
Every represented prior cell has positive proposal probability. Statistical
bounds assume exact model sampling/evaluation; the implementation uses float64
and is explicitly not a formal arithmetic certificate.
"""

from collections import defaultdict
from time import perf_counter

import numpy as np
from scipy.special import ndtr, xlogy
from flint import ctx

from .arithmetic import upper
from .integrate import _box, enclose, verify
from .scene import Scene


def root_for_classes(scene, classes):
    names, ranges, fixed = [], [], {}
    for prefix, params in [("instrument", scene.nuisance)]+[
            (f"s{i}", scene.types[t]["parameters"]) for i, t in enumerate(classes)]:
        for key, (lo, hi) in sorted(params.items()):
            name = prefix+"."+key
            if lo == hi:
                fixed[name] = float(lo)
            else:
                names.append(name)
                ranges.append([float(lo), float(hi)])
    return {"classes": list(classes), "names": names, "ranges": ranges, "fixed": fixed}


class BatchModel:
    """Vectorized independent float renderer, validated against the Arb model."""

    def __init__(self, scene):
        if any(t["kind"] == "kepler" for t in scene.types):
            raise ValueError("Vectorized statistical route does not implement orbital propagation")
        self.scene = scene
        self.data = np.array(list(map(float, scene.data)))
        self.sigma = np.array(scene.sigmas)
        self.chol = None if scene.cholesky is None else np.array([[float(x) for x in r] for r in scene.cholesky])
        self.low_rank = None
        if scene.low_rank is not None:
            self.low_rank = tuple(np.array([[float(x) for x in r] for r in a]) for a in scene.low_rank)

    def likelihood(self, root, coordinates):
        scene = self.scene
        n = len(coordinates)
        values = {**root["fixed"], **{name: coordinates[:, i] for i, name in enumerate(root["names"])}}
        pieces = []
        for e, active in zip(scene.exposures, scene.active):
            camera = e.get("instrument", "default")
            get = lambda k, default=0: values.get("instrument."+camera+"_"+k, default)
            background = np.broadcast_to(np.asarray(get("background", e.get("background", 0)), float), (n,))
            means = np.broadcast_to(background[:, None], (n, len(active))).copy()
            psf = e["psf_sigma"]*np.exp(get("log_psf"))
            gain = np.exp(get("log_gain"))*e.get("flux_scale", 1)
            shape = np.asarray(e["data"]).shape
            xx = np.array([i-(shape[1]-1)/2 for j, i in active])
            yy = np.array([j-(shape[0]-1)/2 for j, i in active])
            for index, cls in enumerate(root["classes"]):
                prefix = f"s{index}."
                p = lambda key, default=0: values.get(prefix+key, default)
                x = p("x")+p("vx")*e["time"]+p("parallax")*e.get("parallax_x", 0)
                y = p("y")+p("vy")*e["time"]+p("parallax")*e.get("parallax_y", 0)
                a, off = e.get("affine", [[1, 0], [0, 1]]), e.get("offset", [0, 0])
                x, y = a[0][0]*x+a[0][1]*y+off[0]+get("dx"), a[1][0]*x+a[1][1]*y+off[1]+get("dy")
                x, y = np.broadcast_to(x, (n,))[:, None], np.broadcast_to(y, (n,))[:, None]
                sx = np.hypot(psf, p("sigma_x")) if scene.types[cls]["kind"] == "galaxy" else psf
                sy = np.hypot(psf, p("sigma_y")) if scene.types[cls]["kind"] == "galaxy" else psf
                sx, sy = np.broadcast_to(sx, (n,))[:, None], np.broadcast_to(sy, (n,))[:, None]
                flux = p("flux")*gain*np.exp(p("color")*e.get("band_coordinate", 0)+p("variability")*e["time"])
                flux = np.broadcast_to(flux, (n,))[:, None]
                means += flux*(ndtr((xx+.5-x)/sx)-ndtr((xx-.5-x)/sx))*(
                    ndtr((yy+.5-y)/sy)-ndtr((yy-.5-y)/sy))
            pieces.append(means)
        means = np.concatenate(pieces, axis=1)
        if scene.noise == "gaussian":
            r = (self.data-means)/self.sigma if self.chol is None else np.linalg.solve(self.chol, (self.data-means).T).T
            energy = np.sum(r*r, axis=1)
            if self.low_rank is not None:
                t, chol = self.low_rank
                projected = np.linalg.solve(chol, (r@t).T).T
                energy -= np.sum(projected*projected, axis=1)
            if np.min(energy) < -1e-8:
                raise ArithmeticError("Negative Gaussian energy beyond rounding tolerance")
            result = np.exp(-np.maximum(0, energy)/2)
        else:
            if np.any(means < 0):
                raise ValueError("Negative Poisson intensity")
            y = self.data
            result = np.exp(np.sum(xlogy(y, means)-means-xlogy(y, y)+y, axis=1))
        if not np.all(np.isfinite(result)) or np.any(result > 1+1e-10):
            raise ArithmeticError("Invalid scaled likelihood")
        return np.minimum(result, 1)

    def event(self, root, coordinates, query):
        if not query.startswith("any:"):
            return np.full(len(coordinates), self.scene.query(root, query), bool)
        _, classname, parameter, lo, hi = query.split(":")
        result = np.zeros(len(coordinates), bool)
        for i, cls in enumerate(root["classes"]):
            if self.scene.types[cls]["name"] != classname:
                continue
            name = f"s{i}.{parameter}"
            x = coordinates[:, root["names"].index(name)] if name in root["names"] else root["fixed"][name]
            result |= (x >= float(lo)) & (x <= float(hi))
        return result


def _interval(total, squares, count, failure):
    mean = total/count
    variance = max(0., (squares-total*total/count)/(count-1))
    slack = np.sqrt(2*variance*np.log(2/failure)/count)+7*np.log(2/failure)/(3*(count-1))
    return max(0., mean-slack), min(1., mean+slack)


def infer(manifest, *, cover_report=None, samples=1_000_000, confidence=.99,
          queries=None, seed=20261021, batch_size=4096, tolerance=.01,
          use_dual_envelope=True):
    """Fixed-budget simultaneous confidence bounds on evidence and events.

    With no cover, draw the complete prior, including *unbounded* Poisson counts.
    With a cover, use every cell's pointwise envelope and retain its count tail.
    No posterior modes are removed by a candidate list or optimizer.
    """
    if (not isinstance(samples, int) or samples < 2 or not 0 < confidence < 1
            or not isinstance(batch_size, int) or batch_size < 1 or not 0 < tolerance < 1):
        raise ValueError("Valid sample count, confidence, batch and tolerance required")
    started = perf_counter()
    with ctx.workprec(100):
        scene = Scene(manifest)
        model = BatchModel(scene)
        queries = queries or ["has:"+t["name"] for t in scene.types]
        cells = None
        envelope, tail = 1., 0.
        if cover_report is not None:
            if cover_report["manifest"] != manifest:
                raise ValueError("Proposal cover belongs to another experiment")
            verify(cover_report, replay=False)
            roots = scene.roots()
            cells, masses = [], []
            for record in cover_report["certificate"]:
                root = roots[record["root"]]
                _, hi, _, _, _ = enclose(scene, root, record["path"], "range", 0)
                if use_dual_envelope:
                    from .envelope import bound
                    try:
                        tighter, _ = bound(scene, root, record["path"])
                        hi = min(hi, tighter)
                    except (ValueError, ZeroDivisionError):
                        pass  # retain the valid original pointwise range
                if hi == 0:
                    continue
                mass = float(root["mass"])/2**len(record["path"])
                if mass <= 0:
                    raise ArithmeticError("Proposal prior mass underflow; cannot silently omit cell")
                bounds = [[float(lo), float(hi)] for lo, hi in _box(root, record["path"])]
                cell_upper = min(1., upper(hi))
                cells.append((root, bounds, mass, cell_upper))
                masses.append(mass*cell_upper)
            proposal = np.asarray(masses)/sum(masses)
            if np.any(proposal <= 0):
                raise ArithmeticError("Nonpositive proposal probability")
            envelope = float(np.max(np.asarray(masses)/proposal))
            tail = max(0., upper(scene.tail))
        construction_seconds = perf_counter()-started
        rng = np.random.default_rng(seed)
        totals = np.zeros(1+2*len(queries))
        squares = np.zeros_like(totals)
        max_weight = 0.
        max_count_seen = 0
        for start in range(0, samples, batch_size):
            n = min(batch_size, samples-start)
            groups = defaultdict(list)
            if cells is not None:
                choices = rng.choice(len(cells), size=n, p=proposal)
                for row, choice in enumerate(choices):
                    groups[int(choice)].append(row)
            else:
                prior = manifest["count_prior"]
                if "poisson_mean" in prior:
                    counts = rng.poisson(prior["poisson_mean"], size=n)
                else:
                    p = np.asarray(prior["probabilities"], float)
                    counts = rng.choice(len(p), size=n, p=p/p.sum())
                q = np.array([s.get("weight", 1.) for s in scene.types], float)
                q /= q.sum()
                for row, count in enumerate(counts):
                    classes = tuple(sorted(rng.choice(len(q), size=count, p=q).tolist()))
                    groups[classes].append(row)
            for key, rows in groups.items():
                if cells is None:
                    root = root_for_classes(scene, key)
                    domain = root["ranges"]
                    factor = 1.
                else:
                    root, domain, mass, _ = cells[key]
                    factor = mass/proposal[key]/envelope
                max_count_seen = max(max_count_seen, len(root["classes"]))
                bounds = np.asarray(domain, float).reshape(-1, 2)
                coordinates = bounds[:, 0]+rng.random((len(rows), len(bounds)))*(bounds[:, 1]-bounds[:, 0])
                weight = factor*model.likelihood(root, coordinates)
                if np.max(weight) > 1+1e-9:
                    raise ArithmeticError("Importance envelope violated")
                weight = np.minimum(weight, 1.)
                max_weight = max(max_weight, float(np.max(weight)))
                columns = [weight]
                for query in queries:
                    event = model.event(root, coordinates, query)
                    columns.extend([weight*event, weight*~event])
                for j, column in enumerate(columns):
                    totals[j] += column.sum()
                    squares[j] += column@column
        # Two one-sided events per expectation, allocated before validation.
        failure = (1-confidence)/(2*len(totals))
        intervals = [tuple(envelope*x for x in _interval(a, b, samples, failure))
                     for a, b in zip(totals, squares)]
        zlo, zhi = intervals[0][0], intervals[0][1]+tail
        posterior, estimates = {}, {}
        for j, query in enumerate(queries):
            nl, nu = intervals[1+2*j]
            cl, cu = intervals[2+2*j]
            posterior[query] = [nl/(nl+cu+tail) if nl+cu+tail > 0 else 0.,
                                (nu+tail)/(nu+tail+cl) if nu+tail+cl > 0 else 1.]
            estimates[query] = float(totals[1+2*j]/totals[0]) if totals[0] > 0 else None
        gap = (zhi-zlo)/zlo if zlo > 0 else None
        met = gap is not None and gap <= tolerance and all(b-a <= tolerance for a, b in posterior.values())
        return {"method": "bounded_weight_importance" if cells is not None else "complete_prior_monte_carlo",
                "status": "confidence_target_met" if met else "unresolved",
                "scaled_evidence": [zlo, zhi], "relative_evidence_gap": gap,
                "queries": posterior, "point_estimates_not_certificates": estimates,
                "simultaneous_confidence": confidence, "samples": samples, "seed": seed,
                "proposal_envelope": envelope, "maximum_observed_scaled_weight": max_weight,
                "dual_envelope_requested": use_dual_envelope,
                "represented_source_count_tail_upper": tail, "maximum_sampled_source_count": max_count_seen,
                "construction_seconds": construction_seconds, "seconds": perf_counter()-started,
                "arithmetic": "float64 sampling, rendering and concentration bounds; not formal arithmetic certification",
                "confidence_scope": "fixed-budget draws conditional on exact stated sampling/evaluation/model assumptions",
                "full_objective_complete": False}
