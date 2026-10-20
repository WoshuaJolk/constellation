"""Independent ambiguous-orbit control; no detected-source input catalogue.

Run: python -m research.fit_sky.orbital_experiments NEW_OUTPUT_DIRECTORY
The finite source-class/count model is declared as the prior, not selected
after inspecting data. Continuous calibration is integrated by quadrature.
"""

from copy import deepcopy
from itertools import product
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.special import logsumexp

from .reference import render
from .sequential import SequentialFit


def ambiguous_scene():
    exposures = []
    for j, time in enumerate([0., 4., 12., 24.]):
        angle = .01720209895*time
        b = j % 2
        exposures.append(dict(id=f"orbit-{j}", instrument="B" if b else "A", time=time,
                              data=np.zeros((5, 7)).tolist(), noise_sigma=.85,
                              psf_sigma=1.05 if b else .8, flux_scale=.85 if b else 1.,
                              affine=[[.9, -.12], [.12, .9]] if b else [[1., 0.], [0., 1.]],
                              offset=[.15, -.1] if b else [0., 0.],
                              observer=[float(np.cos(angle)), float(np.sin(angle)), .00004*b],
                              center=[1., 0., 0.], east=[0., 1., 0.], north=[0., 0., 1.],
                              pixel_scale_arcsec=600., light_time_max=1.))
    source_types = [dict(name="star", kind="star", parameters={"x": [0., 0.],
                            "y": [0., 0.], "flux": [4., 4.]})]
    for name, vy in [("orbit_a", .0024), ("orbit_b", .0032)]:
        values = dict(rx=40., ry=0., rz=0., vx=0., vy=vy, vz=0., flux=4.)
        source_types.append(dict(name=name, kind="kepler", parameters={k: [v, v] for k, v in values.items()}))
    truth = [{"kind": t["kind"], "parameters": {k: v[0] for k, v in t["parameters"].items()}}
             for t in source_types[:2]]
    images = render(exposures, truth, {"A_dx": .12, "B_dx": -.08})
    rng = np.random.default_rng(20261020)
    for e, image in zip(exposures, images):
        e["data"] = (image+rng.normal(0, e["noise_sigma"], image.shape)).tolist()
    design = [[float(e["instrument"] == "A"), float(e["instrument"] == "B")]
              for e in exposures for _ in np.asarray(e["data"]).flat]
    return dict(exposures=exposures, source_types=source_types,
                count_prior={"probabilities": [.15, .5, .35]},
                nuisance={"A_dx": [-.3, .3], "B_dx": [-.3, .3]},
                linear_nuisance=dict(design=design, mean=[0., 0.], covariance=[[.04, .015], [.015, .04]]))


def independent_posterior(manifest, order=12):
    """Full ordered-catalogue enumeration + dense covariance + Gauss rule.

    Only fixed source parameters are supported by this control. Source positions
    come from reference.render's independent DOP853 ODE and Brent light-time
    roots. No Kepler universal solver or sequential likelihood is called.
    Numerical quadrature convergence is checked, not certified.
    """
    if any(lo != hi for t in manifest["source_types"] for lo, hi in t["parameters"].values()):
        raise ValueError("Independent enumeration control requires fixed source parameters")
    exposures = manifest["exposures"]
    data = np.concatenate([np.ravel(e["data"]) for e in exposures])
    sigma = np.concatenate([np.full(np.size(e["data"]), e["noise_sigma"]) for e in exposures])
    linear = manifest["linear_nuisance"]
    design = np.asarray(linear["design"])
    covariance = np.diag(sigma**2)+design@linear["covariance"]@design.T
    factor = cholesky(covariance, lower=True)
    normalization = -.5*len(data)*np.log(2*np.pi)-np.log(np.diag(factor)).sum()
    sources = [{"kind": t["kind"], "parameters": {k: v[0] for k, v in t["parameters"].items()}}
               for t in manifest["source_types"]]
    class_weights = np.array([t.get("weight", 1.) for t in manifest["source_types"]], float)
    class_weights /= class_weights.sum()
    count_weights = np.asarray(manifest["count_prior"]["probabilities"], float)
    count_weights /= count_weights.sum()
    catalogues = [classes for k in range(len(count_weights)) for classes in product(range(len(sources)), repeat=k)]
    names = [t["name"] for t in manifest["source_types"]]
    queries = [f"count={k}" for k in range(len(count_weights))]+["has:"+n for n in names]
    events = np.array([[len(c) == k for k in range(len(count_weights))]+
                       [j in c for j in range(len(names))] for c in catalogues])
    prior = np.array([count_weights[len(c)]*np.prod(class_weights[list(c)]) for c in catalogues])
    nodes, weights = np.polynomial.legendre.leggauss(order)
    axes = sorted(manifest["nuisance"])
    components, nuisance_rows = [], []
    for indices in product(range(order), repeat=len(axes)):
        nuisance = {key: (lo+hi)/2+(hi-lo)*nodes[i]/2 for key, i in zip(axes, indices)
                    for lo, hi in [manifest["nuisance"][key]]}
        templates = [np.concatenate([a.ravel() for a in render(exposures, [s], nuisance)]) for s in sources]
        means = np.array([sum((templates[j] for j in c), np.zeros(len(data))) for c in catalogues])
        residual = data-means-design@linear["mean"]
        whitened = solve_triangular(factor, residual.T, lower=True)
        ll = normalization-.5*np.sum(whitened**2, axis=0)
        components.append(ll+np.log(prior)+sum(np.log(weights[i]/2) for i in indices))
        nuisance_rows.append([nuisance[key] for key in axes])
    components = np.asarray(components)
    log_z = logsumexp(components)
    joint_weights = np.exp(components-log_z)
    posterior = joint_weights.sum(axis=0)@events
    nuisance_mean = joint_weights.sum(axis=1)@np.asarray(nuisance_rows)
    return dict(log_evidence=float(log_z), queries=dict(zip(queries, posterior.tolist())),
                nuisance_mean=dict(zip(axes, nuisance_mean.tolist())),
                order=order, scope="Independent numerical quadrature; not a certificate")


def run(directory):
    directory.mkdir(parents=True, exist_ok=False)
    manifest = ambiguous_scene()
    (directory/"input.json").write_text(json.dumps({"scene": manifest}, indent=2)+"\n")
    references = [independent_posterior(manifest, order) for order in (8, 12)]
    (directory/"references.json").write_text(json.dumps(references, indent=2)+"\n")
    runs = []
    for seed in (20261021, 20261022, 20261023):
        fit = SequentialFit(manifest, particles=1024, sweeps=12, seed=seed,
                            queries=list(references[-1]["queries"]))
        for endpoint in range(1, 5):
            fit.assimilate(endpoint)
            print(json.dumps(dict(seed=seed, **fit.summary())), flush=True)
        fit.save(directory/f"checkpoint-{seed}.npz")
        report = fit.report()
        (directory/f"report-{seed}.json").write_text(json.dumps(report, indent=2)+"\n")
        runs.append(dict(seed=seed, summary=fit.summary(),
                         max_query_error=max(abs(v["estimate"]-references[-1]["queries"][q])
                                             for q, v in fit.summary()["queries"].items())))
    # Quantify bias from a hypothetical post-selection deletion of orbit_b.
    # Renormalizing the remaining class prior defines a different model, not
    # the original posterior (nor conditioning it without adjusting count mass).
    removed = deepcopy(manifest)
    removed["source_types"] = removed["source_types"][:2]
    removed_reference = independent_posterior(removed, 12)
    summary = dict(references=references, runs=runs, shortlist_recomputed_prior=removed_reference,
                   excluded_original_posterior_event=references[-1]["queries"]["has:orbit_b"],
                   status="approximate_uncertified", full_objective_complete=False,
                   note="Synthetic finite orbit-class control; no whole-domain posterior guarantee")
    (directory/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    root = Path(__file__).resolve().parents[2]
    files = [Path(__file__), Path(__file__).with_name("orbital.py"),
             Path(__file__).with_name("sequential.py"), Path(__file__).with_name("reference.py"),
             root/"research/kepler_transform/universal.py", root/"research/detection_limit/interval_orbits.py"]
    (directory/"source_hashes.json").write_text(json.dumps({str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                                         for p in files}, indent=2)+"\n")


if __name__ == "__main__":
    run(Path(sys.argv[1]))
