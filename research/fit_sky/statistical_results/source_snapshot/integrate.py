"""Validated adaptive integration over continuous, variable-size catalogues.

Every leaf retains its prior mass and likelihood enclosure. A Poisson count tail
has scaled likelihood in [0,1]. No optimizer, shortlist, or sampling convergence
statement substitutes for the remaining evidence mass.
"""

from fractions import Fraction
from itertools import product
from math import isfinite
from time import perf_counter
import hashlib
import json

from flint import arb, ctx

from .arithmetic import Jet, ball, lower, upper
from .scene import Scene


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _box(root, path):
    result = [[arb(lo), arb(hi)] for lo, hi in root["ranges"]]
    for axis, side in path:
        if not isinstance(axis, int) or not 0 <= axis < len(result) or side not in (0, 1):
            raise ValueError("Invalid subdivision path")
        midpoint = (result[axis][0]+result[axis][1])/2
        result[axis][1-side] = midpoint
    return result


def enclose(scene, root, path, method, max_gauss_dimension):
    if method == "auto":
        a = enclose(scene, root, path, "gauss", max_gauss_dimension)
        b = enclose(scene, root, path, "hybrid", max_gauss_dimension)
        chosen = a if a[1]-a[0] <= b[1]-b[0] else b
        return max(a[0], b[0]), min(a[1], b[1]), chosen[2], a[3]+b[3], "intersection_"+chosen[4]
    box = _box(root, path)
    parameters = [lo.union(hi) for lo, hi in box]
    eliminated = None
    evaluate = lambda p: scene.likelihood(root, p)
    if method == "hybrid" and scene.noise == "gaussian":
        for axis, name in enumerate(root["names"]):
            if name.endswith(".flux"):
                try:
                    scene.marginal_flux(root, parameters, axis, box[axis])
                    eliminated = axis
                    evaluate = lambda p: scene.marginal_flux(root, p, eliminated, box[eliminated])
                except (ValueError, ZeroDivisionError):
                    pass
                break
    axes = [i for i in range(len(box)) if i != eliminated]
    # A failed enclosure is [0,1], never an exclusion. It can be subdivided.
    try:
        value = evaluate(parameters)
        if not value.is_finite():
            raise ValueError("Nonfinite likelihood enclosure")
        result = value.intersection(arb(0).union(arb(1)))
        if eliminated is not None:
            result = result.intersection(scene.likelihood(root, parameters))
    except (ValueError, ZeroDivisionError):
        return arb(0), arb(1), None, 1, "unresolved_forward_model"
    if not axes:
        return max(arb(0), result.lower()), min(arb(1), result.upper()), None, 1, "analytic_flux" if eliminated is not None else "point"
    calls, errors = 1, None
    mode = "range"
    if method in ("gauss", "hybrid") and len(axes) <= max_gauss_dimension:
        try:
            errors = [arb(0) for _ in box]
            for axis in axes:
                lo, hi = box[axis]
                jets = [Jet(x, (hi-lo)/2 if i == axis else 0) for i, x in enumerate(parameters)]
                sweep = Jet.cast(evaluate(jets))
                error = sweep.c[4].abs_upper()*arb(4)/45
                if not error.is_finite():
                    raise ValueError("Nonfinite quadrature remainder")
                errors[axis] = error
            total = arb(0)
            for signs in product([-1, 1], repeat=len(axes)):
                point = list(parameters)
                for axis, s in zip(axes, signs):
                    lo, hi = box[axis]
                    point[axis] = (lo+hi)/2+s*(hi-lo)/2/arb(3).sqrt()
                total += evaluate(point)
            total /= 2**len(axes)
            radius = sum(errors, arb(0)).upper()
            enclosure = total+arb(0, radius)
            result = result.intersection(enclosure)
            calls += len(axes)+2**len(axes)
            mode = "analytic_flux_gauss2" if eliminated is not None else "gauss2_directional_fourth_derivatives"
        except NotImplementedError:
            errors = None
        except (ValueError, ZeroDivisionError):
            # Keep the original valid range; do not trust failed quadrature.
            errors = None
    return max(arb(0), result.lower()), min(arb(1), result.upper()), errors, calls, mode


def _summary(scene, roots, leaves, queries):
    lower_terms, upper_terms = [], []
    for cell in leaves.values():
        mass = roots[cell["root"]]["mass"]/2**len(cell["path"])
        lower_terms.append(mass*cell["lo"])
        upper_terms.append(mass*cell["hi"])
    zlo = sum(lower_terms, arb(0)).lower()
    zhi = (sum(upper_terms, arb(0))+scene.tail).upper()
    probabilities = {}
    for query in queries:
        membership = [scene.query(roots[cell["root"]], query, _box(roots[cell["root"]], cell["path"]))
                      for cell in leaves.values()]
        nlo = sum((lo for lo, cell in zip(lower_terms, leaves.values())
                   if scene.query(roots[cell["root"]], query, _box(roots[cell["root"]], cell["path"])) is True), arb(0)).lower()
        nhi = sum((hi for hi, member in zip(upper_terms, membership) if member is not False), arb(0)).upper()
        clo = sum((lo for lo, member in zip(lower_terms, membership) if member is False), arb(0)).lower()
        chi = sum((hi for hi, member in zip(upper_terms, membership) if member is not True), arb(0)).upper()
        # Unknown tail can belong to either event; conservative for all queries.
        nhi, chi = nhi+scene.tail, chi+scene.tail
        low = nlo/(nlo+chi) if nlo+chi > 0 else arb(0)
        high = nhi/(nhi+clo) if nhi+clo > 0 else arb(1)
        probabilities[query] = [max(0., lower(low)), min(1., upper(high))]
    relative = (zhi-zlo)/zlo if zlo > 0 else None
    normalized_lo = zlo*scene.log_normalization.exp()
    normalized_hi = zhi*scene.log_normalization.exp()
    return {
        "scaled_evidence": [max(0., lower(zlo)), max(0., upper(zhi))],
        "log_evidence": [lower(normalized_lo.log()) if normalized_lo > 0 else None,
                         upper(normalized_hi.log()) if normalized_hi > 0 else None],
        "relative_evidence_gap": upper(relative) if relative is not None and isfinite(upper(relative)) else None,
        "queries": probabilities,
        "source_count_tail_prior_upper": max(0., upper(scene.tail)),
        "source_count_tail_posterior_upper": min(1., upper(scene.tail/(scene.tail+zlo))) if scene.tail+zlo > 0 else 1.,
    }


def infer(manifest, *, queries=None, tolerance=.01, max_leaves=4096,
          max_seconds=120., method="gauss", max_gauss_dimension=6, precision=100):
    """Bound evidence and symmetric posterior events for the declared model.

    Limits are soft boundaries between cell evaluations. No claim of globally
    efficient inference is made. precision applies to all model/prior arithmetic.
    """
    if (method not in ("range", "gauss", "hybrid", "auto") or not 0 < tolerance < 1
            or not isinstance(max_leaves, int) or max_leaves < 1
            or not isfinite(max_seconds) or max_seconds <= 0
            or not isinstance(max_gauss_dimension, int) or not 0 <= max_gauss_dimension <= 12
            or not isinstance(precision, int) or precision < 64):
        raise ValueError("Valid tolerances, budgets, quadrature dimension and precision required")
    started = perf_counter()
    with ctx.workprec(precision):
        scene = Scene(manifest)
        roots = scene.roots()
        if not roots:
            raise ValueError("Count prior has no represented configurations")
        queries = queries or [f"has:{t['name']}" for t in scene.types]
        for root in roots:
            for query in queries:
                scene.query(root, query)
        leaves, counter, calls = {}, 0, 0
        history = []

        def add(index, path):
            nonlocal counter, calls
            lo, hi, errors, work, mode = enclose(scene, roots[index], path, method, max_gauss_dimension)
            leaves[counter] = {"root": index, "path": path, "lo": lo, "hi": hi,
                               "errors": errors, "mode": mode}
            calls += work
            counter += 1

        # Root initialization is mandatory; the reported budget includes it.
        for index in range(len(roots)):
            add(index, [])
        while True:
            report = _summary(scene, roots, leaves, queries)
            widths = [b-a for a, b in report["queries"].values()]
            met = (report["relative_evidence_gap"] is not None
                   and report["relative_evidence_gap"] <= tolerance
                   and max(widths, default=0) <= tolerance)
            if met or len(leaves) >= max_leaves or perf_counter()-started >= max_seconds:
                break
            candidates = []
            for key, cell in leaves.items():
                root = roots[cell["root"]]
                if not root["names"] or len(cell["path"]) >= 200:
                    continue
                ambiguous = any(scene.query(root, query, _box(root, cell["path"])) is None for query in queries)
                gap = root["mass"]*(cell["hi"] if ambiguous else cell["hi"]-cell["lo"])/2**len(cell["path"])
                if gap > 0:
                    candidates.append((float(gap.log().upper()), key))
            if not candidates:
                break
            _, key = max(candidates)
            cell = leaves.pop(key)
            root = roots[cell["root"]]
            unresolved_axes = []
            for query in queries:
                if scene.query(root, query, _box(root, cell["path"])) is None:
                    parameter = query.split(":")[2]
                    unresolved_axes.extend(i for i, name in enumerate(root["names"])
                                           if name.endswith("."+parameter))
            if unresolved_axes:
                axis = min(unresolved_axes, key=lambda i: sum(a == i for a, _ in cell["path"]))
            elif cell["errors"] is not None and any(x > 0 for x in cell["errors"]):
                axis = max(range(len(cell["errors"])),
                           key=lambda a: float(cell["errors"][a].abs_upper().log())
                           if cell["errors"][a] > 0 else -float("inf"))
            else:
                # Fair splitting in normalized coordinates, avoiding unit bias.
                depths = [sum(a == i for a, _ in cell["path"])
                          for i in range(len(roots[cell["root"]]["names"]))]
                axis = min(range(len(depths)), key=lambda a: (depths[a], a))
            for side in (0, 1):
                add(cell["root"], cell["path"]+[[axis, side]])
            if len(history) < 20 and (len(leaves) & (len(leaves)-1)) == 0:
                history.append({"leaves": len(leaves), **report})
        serialized = []
        exported = {}
        for cell in leaves.values():
            record = {"root": cell["root"], "path": cell["path"],
                      "likelihood": [max(0., lower(cell["lo"])), min(1., upper(cell["hi"]))],
                      "mode": cell["mode"]}
            serialized.append(record)
            exported[len(exported)] = {"root": cell["root"], "path": cell["path"],
                                       "lo": arb(record["likelihood"][0]), "hi": arb(record["likelihood"][1])}
        report = _summary(scene, roots, exported, queries)
        met = (report["relative_evidence_gap"] is not None and report["relative_evidence_gap"] <= tolerance
               and all(b-a <= tolerance for a, b in report["queries"].values()))
        return {
            "schema_version": 2, "tool": "fit_sky", "manifest": manifest,
            "manifest_sha256": hashlib.sha256(_canonical(manifest)).hexdigest(),
            "settings": {"method": method, "max_gauss_dimension": max_gauss_dimension,
                         "precision": precision, "tolerance": tolerance,
                         "max_leaves": max_leaves, "max_seconds": max_seconds},
            "status": "bounded_target_met" if met else "unresolved",
            "full_objective_complete": False, **report,
            "work": {"leaves": len(leaves), "roots": len(roots), "forward_calls": calls,
                     "seconds": perf_counter()-started,
                     "max_dimension": max(len(r["names"]) for r in roots)},
            "certificate": serialized, "progress": history,
            "scope": "Declared compact source/instrument priors and Gaussian/Poisson image model",
            "arithmetic": "Arb prior, forward likelihood and quadrature enclosures; outward float report bounds",
        }


def verify(report, *, replay=True):
    """Replay every leaf and independently validate its complete binary partition."""
    if report.get("schema_version") != 2 or report.get("tool") != "fit_sky" or report.get("full_objective_complete") is not False:
        raise ValueError("Unsupported report schema or completion claim")
    if hashlib.sha256(_canonical(report["manifest"])).hexdigest() != report["manifest_sha256"]:
        raise ValueError("Manifest hash mismatch")
    settings = report["settings"]
    with ctx.workprec(settings["precision"]):
        scene = Scene(report["manifest"])
        roots = scene.roots()
        trees = [{} for _ in roots]
        leaves = {}
        for index, record in enumerate(report["certificate"]):
            r = record["root"]
            if not isinstance(r, int) or not 0 <= r < len(roots):
                raise ValueError("Unknown prior root")
            node = trees[r]
            path = record["path"]
            _box(roots[r], path)
            for axis, side in path:
                if "leaf" in node or ("axis" in node and node["axis"] != axis):
                    raise ValueError("Overlapping or inconsistent partition")
                node["axis"] = axis
                node = node.setdefault(side, {})
            if node:
                raise ValueError("Duplicate or prefix leaf")
            node["leaf"] = True
            lo, hi = record["likelihood"]
            if not all(isfinite(x) for x in [lo, hi]) or not 0 <= lo <= hi <= 1:
                raise ValueError("Invalid likelihood bound")
            if replay:
                a, b, _, _, _ = enclose(scene, roots[r], path, settings["method"], settings["max_gauss_dimension"])
                if not arb(lo) <= a or not arb(hi) >= b:
                    raise ValueError("Saved likelihood bound fails replay")
            leaves[index] = {"root": r, "path": path, "lo": arb(lo), "hi": arb(hi)}

        def complete(node):
            if "leaf" in node:
                return
            if set(node) != {"axis", 0, 1}:
                raise ValueError("Partition omitted prior mass")
            complete(node[0])
            complete(node[1])

        for tree in trees:
            complete(tree)
        recomputed = _summary(scene, roots, leaves, list(report["queries"]))
        for name in recomputed:
            if report[name] != recomputed[name]:
                raise ValueError("Aggregate evidence or posterior differs from certificate")
        met = (recomputed["relative_evidence_gap"] is not None
               and recomputed["relative_evidence_gap"] <= settings["tolerance"]
               and all(b-a <= settings["tolerance"] for a, b in recomputed["queries"].values()))
        if report["status"] != ("bounded_target_met" if met else "unresolved"):
            raise ValueError("Completion status differs from replay")
        return {"valid": True, "replayed": replay, "leaves": len(leaves),
                "roots": len(roots), "recomputed_outward_summary": recomputed,
                "full_objective_complete": False}
