"""Continuous orbital-velocity/count/class/calibration posterior control.

Quadrature nodes integrate the declared continuous prior; they are not a
detected-source catalogue or a shortlist supplied to SequentialFit.
"""

from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.special import logsumexp

from .orbital_experiments import ambiguous_scene
from .reference import render
from .sequential import SequentialFit


QUERIES = ["count=0", "count=1", "count=2", "has:star", "has:orbit_a",
           "any:orbit_a:vy:-0.002:0.006"]


def scene():
    m = ambiguous_scene()
    m["source_types"] = m["source_types"][:2]
    m["source_types"][1]["parameters"]["vy"] = [-.002, .014]
    m["nuisance"].pop("B_dx")
    return m


def reference(m, order):
    """Independent ODE templates and dense joint covariance, all K=0/1/2.

    Split velocity quadrature at the query boundary. Every node contributes
    its prior integration weight; no likelihood-based pruning is performed.
    """
    exposures = m["exposures"]
    data = np.concatenate([np.ravel(e["data"]) for e in exposures])
    linear = m["linear_nuisance"]
    design = np.array(linear["design"])
    factor = cholesky(np.eye(len(data))*.85**2+design@linear["covariance"]@design.T, lower=True)
    norm = -.5*len(data)*np.log(2*np.pi)-np.log(np.diag(factor)).sum()
    nodes, weights = np.polynomial.legendre.leggauss(order)
    # Uniform continuous orbital prior and equal star/orbit class weights.
    velocities = np.r_[.002+.004*nodes, .010+.004*nodes]
    source_weights = np.r_[.5, weights/8, weights/8]
    types = m["source_types"]
    fixed = [{"kind": t["kind"], "parameters": {k: v[0] for k, v in t["parameters"].items()}}
             for t in types]
    sources = [fixed[0]]+[dict(kind="kepler", parameters=dict(fixed[1]["parameters"], vy=v)) for v in velocities]
    catalogues = [c for k in range(3) for c in product(range(len(sources)), repeat=k)]
    count_prior = m["count_prior"]["probabilities"]
    priors = np.array([count_prior[len(c)]*np.prod(source_weights[list(c)]) for c in catalogues])
    events = np.array([[len(c) == k for k in range(3)]+[0 in c, any(j > 0 for j in c),
                        any(0 < j <= order for j in c)] for c in catalogues])
    values = []
    for node, weight in zip(nodes, weights):
        nuisance = {"A_dx": .3*node}
        templates = np.array([np.concatenate([a.ravel() for a in render(exposures, [s], nuisance)]) for s in sources])
        means = np.array([sum((templates[j] for j in c), np.zeros(len(data))) for c in catalogues])
        whitened = solve_triangular(factor, (data-means-design@linear["mean"]).T, lower=True)
        values.append(norm-.5*np.sum(whitened**2, axis=0)+np.log(priors)+np.log(weight/2))
    values = np.array(values)
    log_z = logsumexp(values)
    posterior = np.exp(values-log_z).sum(axis=0)@events
    return dict(order=order, log_evidence=float(log_z), queries=dict(zip(QUERIES, posterior.tolist())),
                scope="Continuous velocity and calibration quadrature with independent ODE rendering; numerical, not certified")


def run(directory):
    directory.mkdir(parents=True, exist_ok=False)
    m = scene()
    (directory/"input.json").write_text(json.dumps(dict(scene=m, queries=QUERIES), indent=2)+"\n")
    refs = [reference(m, order) for order in [8, 12]]
    (directory/"references.json").write_text(json.dumps(refs, indent=2)+"\n")
    runs = []
    for seed in [20261024, 20261025, 20261026]:
        fit = SequentialFit(m, particles=2048, sweeps=16, seed=seed, queries=QUERIES)
        for end in range(1, 5):
            fit.assimilate(end)
            print(json.dumps(dict(seed=seed, **fit.summary())), flush=True)
        fit.save(directory/f"checkpoint-{seed}.npz")
        report = fit.report(include_catalogues=True)
        (directory/f"report-{seed}.json").write_text(json.dumps(report, indent=2)+"\n")
        runs.append(dict(seed=seed, summary=fit.summary(), max_query_error=max(
            abs(v["estimate"]-refs[-1]["queries"][q]) for q, v in fit.summary()["queries"].items())))
    (directory/"summary.json").write_text(json.dumps(dict(references=refs, runs=runs,
        status="approximate_uncertified", full_objective_complete=False), indent=2)+"\n")


if __name__ == "__main__":
    run(Path(sys.argv[1]))
