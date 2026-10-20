"""Finite catalogue sum-product inference, retaining source-source interactions.

An exact algebraic reduction in the supplied binary dictionary, evaluated in
float64. This is not a continuous-source completion or an arithmetic certificate.
"""

from dataclasses import dataclass
from itertools import product
from time import perf_counter

import numpy as np
from scipy.special import expit, logsumexp


@dataclass
class Factor:
    variables: tuple
    values: np.ndarray


def _partition(unary, pairs, fixed, max_entries):
    """Variable elimination in the log semiring, choosing the smallest bucket."""
    n = len(unary)
    factors = []
    for i in range(n):
        values = np.array([0., unary[i]])
        if i in fixed:
            if fixed[i] not in (0, 1):
                raise ValueError("Binary conditions required")
            values[1-fixed[i]] = -np.inf
        factors.append(Factor((i,), values))
    for i, j, coupling in pairs:
        values = np.zeros((2, 2))
        values[1, 1] = coupling
        factors.append(Factor((i, j), values))
    peak, operations = 1, 0
    remaining = set(range(n))
    while remaining:
        def scope(v):
            return sorted({a for f in factors if v in f.variables for a in f.variables})
        v = min(remaining, key=lambda a: (len(scope(a)), a))
        variables = scope(v)
        entries = 2**len(variables)
        if entries > max_entries:
            raise ValueError(f"Interaction width requires {entries} entries; limit {max_entries}")
        peak = max(peak, entries)
        bucket = [f for f in factors if v in f.variables]
        factors = [f for f in factors if v not in f.variables]
        total = np.zeros([2]*len(variables))
        for f in bucket:
            shape = [2 if a in f.variables else 1 for a in variables]
            total += f.values.reshape(shape)
        operations += entries*len(bucket)
        reduced = logsumexp(total, axis=variables.index(v))
        factors.append(Factor(tuple(a for a in variables if a != v), reduced))
        remaining.remove(v)
    return float(sum(f.values for f in factors)), peak, operations


def infer(data, templates, probabilities, *, covariance=None, queries=None,
          drop_threshold=0., max_entries=2**20):
    """Integrate binary source existence, including overlapping templates.

    Columns of templates include the candidate's specified flux. covariance is
    the full known pixel covariance (including Gaussian marginalized nuisances
    if desired). The finite dictionary and Bernoulli prior are explicit inputs.
    Returned evidence includes Gaussian normalization and covariance determinant.
    """
    started = perf_counter()
    y, a, p = map(lambda x: np.asarray(x, float), [data, templates, probabilities])
    if (y.ndim != 1 or a.shape != (len(y), len(p)) or p.ndim != 1
            or not all(np.all(np.isfinite(x)) for x in [y, a, p])
            or np.any(p <= 0) or np.any(p >= 1)):
        raise ValueError("Finite pixels/templates and Bernoulli probabilities in (0,1) required")
    if not np.isfinite(drop_threshold) or drop_threshold < 0 or max_entries < 2:
        raise ValueError("Nonnegative dropping threshold and positive factor budget required")
    logdet = 0.
    if covariance is not None:
        c = np.asarray(covariance, float)
        if c.shape != (len(y), len(y)) or not np.allclose(c, c.T, rtol=0, atol=1e-12):
            raise ValueError("Symmetric pixel covariance required")
        chol = np.linalg.cholesky(c)
        logdet = 2*np.log(np.diag(chol)).sum()
        y = np.linalg.solve(chol, y)
        a = np.linalg.solve(chol, a)
    gram, score = a.T@a, a.T@y
    unary = np.log(p)-np.log1p(-p)+score-.5*np.diag(gram)
    pairs, dropped = [], []
    for i in range(len(p)):
        for j in range(i+1, len(p)):
            if abs(gram[i, j]) <= drop_threshold:
                dropped.append(abs(float(gram[i, j])))
            else:
                pairs.append((i, j, -float(gram[i, j])))
    epsilon = float(sum(dropped))
    logz, peak, operations = _partition(unary, pairs, {}, max_entries)
    common = float(np.log1p(-p).sum()-.5*y@y-.5*len(y)*np.log(2*np.pi)-.5*logdet)
    if queries is None:
        queries = {f"source_{i}": {i: 1} for i in range(len(p))}
    posterior = {}
    for name, condition in queries.items():
        if not set(condition) <= set(range(len(p))):
            raise ValueError("Query index outside source dictionary")
        lognum, used, work = _partition(unary, pairs, condition, max_entries)
        q = min(1., float(np.exp(lognum-logz)))
        logodds = np.log(q)-np.log1p(-q) if 0 < q < 1 else (-np.inf if q == 0 else np.inf)
        posterior[name] = {"probability": q,
                           "interaction_bound": [float(expit(logodds-2*epsilon)),
                                                 float(expit(logodds+2*epsilon))]}
        peak, operations = max(peak, used), operations+work
    return {"log_evidence": common+logz,
            "log_evidence_interaction_bound": [common+logz-epsilon, common+logz+epsilon],
            "queries": posterior, "sources": len(p), "retained_edges": len(pairs),
            "omitted_log_likelihood_bound": epsilon, "peak_factor_entries": peak,
            "factor_entry_operations": operations, "seconds": perf_counter()-started,
            "arithmetic": "float64; interaction inequalities are not roundoff-certified",
            "scope": "supplied binary source dictionary with fixed templates",
            "full_objective_complete": False}


def brute_force(data, templates, probabilities):
    """Independent direct-pixel reference for small dictionaries."""
    y, a, p = map(np.asarray, [data, templates, probabilities])
    states = np.array(list(product([0, 1], repeat=len(p))))
    residuals = y[None, :]-states@a.T
    log_weights = -.5*np.sum(residuals**2, axis=1)
    log_weights += states@np.log(p)+(1-states)@np.log1p(-p)
    logz = logsumexp(log_weights)
    weights = np.exp(log_weights-logz)
    return float(logz-.5*len(y)*np.log(2*np.pi)), weights@states


def mix_shared_nuisance(reports, prior_weights):
    """Marginalize a *shared* discrete nuisance after conditional inference."""
    weights = np.asarray(prior_weights, float)
    if (weights.shape != (len(reports),) or np.any(weights <= 0)
            or not np.all(np.isfinite(weights)) or not np.isclose(weights.sum(), 1)):
        raise ValueError("Positive normalized nuisance weights required")
    if any(r["omitted_log_likelihood_bound"] != 0 for r in reports):
        raise ValueError("This mixture requires all conditional interactions retained")
    names = set(reports[0]["queries"])
    if any(set(r["queries"]) != names for r in reports):
        raise ValueError("Identical queries required")
    logweights = np.log(weights)+np.array([r["log_evidence"] for r in reports])
    logz = logsumexp(logweights)
    posterior = np.exp(logweights-logz)
    return {"log_evidence": float(logz), "nuisance_posterior": posterior.tolist(),
            "queries": {name: float(sum(w*r["queries"][name]["probability"]
                                      for w, r in zip(posterior, reports))) for name in names},
            "arithmetic": "float64", "full_objective_complete": False}
