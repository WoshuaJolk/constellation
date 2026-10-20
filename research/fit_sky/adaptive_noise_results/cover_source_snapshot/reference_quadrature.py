"""Independent tensor quadrature diagnostics for zero/one-source controls.

This is an independent integration rule, not an independent forward renderer:
it uses the separately tested vectorized renderer. Convergence across orders
is a diagnostic, never a missing-mode or numerical-error certificate.
"""

from itertools import product
from time import perf_counter
import numpy as np
from scipy.special import log_ndtr
from flint import ctx

from .scene import Scene
from .statistical import BatchModel


def _flux_average(model, root, points, axis, endpoints):
    """Recover the quadratic log likelihood from three flux evaluations."""
    logs = []
    for f in (0., 1., 2.):
        p = points.copy()
        p[:, axis] = f
        value = model.likelihood(root, p)
        if np.any(value <= 0):
            raise ArithmeticError("Reference quadratic recovery underflow")
        logs.append(np.log(value))
    c, l1, l2 = logs
    a = -(l2-2*l1+c)
    b = l1-c+a/2
    if np.any(a <= 0):
        raise ArithmeticError("Reference flux information lost to rounding")
    lo, hi = endpoints
    left, right = np.sqrt(a)*(lo-b/a), np.sqrt(a)*(hi-b/a)
    # Phi(right)-Phi(left), using the small tail on either side of zero.
    upper_log = np.where(left > 0, log_ndtr(-left), log_ndtr(right))
    lower_log = np.where(left > 0, log_ndtr(-right), log_ndtr(left))
    gap = upper_log+np.log(-np.expm1(lower_log-upper_log))
    return np.exp(c+b*b/(2*a)+.5*np.log(2*np.pi/a)+gap-np.log(hi-lo))


def integrate(manifest, *, order=24, queries=None, batch_size=2048):
    started = perf_counter()
    with ctx.workprec(100):
        scene = Scene(manifest)
        roots = scene.roots()
        if scene.tail != 0 or any(len(r["classes"]) > 1 for r in roots) or scene.noise != "gaussian":
            raise ValueError("Reference requires finite zero/one-source Gaussian scenes")
        queries = queries or ["has:"+t["name"] for t in scene.types]
        model = BatchModel(scene)
        nodes, weights = np.polynomial.legendre.leggauss(order)
        totals = np.zeros(1+len(queries))
        evaluations = 0
        for root in roots:
            fluxes = [i for i, name in enumerate(root["names"]) if name.endswith(".flux")]
            eliminated = fluxes[0] if fluxes else None
            axes = [i for i in range(len(root["names"])) if i != eliminated]
            # Split query discontinuities before applying the smooth rule.
            breaks = [[lo, hi] for lo, hi in root["ranges"]]
            for query in queries:
                if not query.startswith("any:"):
                    continue
                _, classname, parameter, lo, hi = query.split(":")
                for i, cls in enumerate(root["classes"]):
                    key = f"s{i}.{parameter}"
                    if scene.types[cls]["name"] == classname and key in root["names"]:
                        axis = root["names"].index(key)
                        if axis == eliminated:
                            raise ValueError("Flux-event reference requires explicit truncated flux integration")
                        a, b = root["ranges"][axis]
                        breaks[axis] += [v for v in map(float, (lo, hi)) if a < v < b]
            intervals = [list(zip(sorted(set(breaks[i]))[:-1], sorted(set(breaks[i]))[1:])) for i in axes]
            for region in product(*intervals):
                fractions = np.prod([(b-a)/(root["ranges"][i][1]-root["ranges"][i][0])
                                     for i, (a, b) in zip(axes, region)])
                all_indices = np.array(list(product(range(order), repeat=len(axes))), dtype=int).reshape(-1, len(axes)) if axes else np.empty((1, 0), int)
                for start in range(0, len(all_indices), batch_size):
                    index = all_indices[start:start+batch_size]
                    p = np.broadcast_to(np.mean(root["ranges"], axis=1) if root["ranges"] else [],
                                        (len(index), len(root["ranges"]))).copy()
                    w = np.ones(len(index))*float(root["mass"])*fractions
                    for j, (axis, (a, b)) in enumerate(zip(axes, region)):
                        p[:, axis] = (a+b)/2+(b-a)/2*nodes[index[:, j]]
                        w *= weights[index[:, j]]/2
                    likelihood = (_flux_average(model, root, p, eliminated, root["ranges"][eliminated])
                                  if eliminated is not None else model.likelihood(root, p))
                    weighted = w*likelihood
                    totals[0] += weighted.sum()
                    for j, query in enumerate(queries):
                        totals[j+1] += (weighted*model.event(root, p, query)).sum()
                    evaluations += len(p)*(3 if eliminated is not None else 1)
        return {"method": "tensor_gauss_with_quadratic_flux_marginalization",
                "order": order, "scaled_evidence": float(totals[0]),
                "queries": {q: float(v/totals[0]) for q, v in zip(queries, totals[1:])},
                "likelihood_evaluations": evaluations, "seconds": perf_counter()-started,
                "is_certificate": False, "full_objective_complete": False}
