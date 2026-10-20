"""Fixed FitSky campaign. All failed requests are retained with their domains."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
import shutil
from time import perf_counter

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr

from .graph import infer as graph_infer, brute_force
from .integrate import infer, verify
from .reference import render
from .sampling import sample


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def scalar():
    return {"source_types": [{"name": "star", "kind": "star", "parameters": {
        "x": [0, 0], "y": [0, 0], "flux": [1, 3]}}],
        "exposures": [{"id": "one", "time": 0, "data": [[1.]], "psf_sigma": .6}],
        "count_prior": {"probabilities": [.5, .5]}}


def synthetic(kind):
    rng = np.random.default_rng(20261020)
    if kind == "confused":
        exposures = [{"id": "one", "time": 0, "data": np.zeros((1, 5)).tolist(),
                      "psf_sigma": .7, "noise_sigma": .7}]
        truth = [{"kind": "star", "parameters": {"x": x, "y": 0, "flux": 4.}} for x in [-.4, .4]]
        sources = [{"name": "star", "kind": "star", "parameters": {
            "x": [-1.2, 1.2], "y": [0, 0], "flux": [1, 8]}}]
        count = {"probabilities": [.2, .4, .4]}
        nuisance, true_nuisance = {}, {}
    else:
        exposures = [{"id": f"e{i}", "time": t, "instrument": "a" if i < 2 else "b",
                      "data": np.zeros((3, 3)).tolist(), "psf_sigma": .65, "noise_sigma": .6,
                      "parallax_x": float(np.sin(t)), "parallax_y": float(np.cos(t))}
                     for i, t in enumerate([-1., 0., 1.])]
        truth = [{"kind": "star", "parameters": {"x": .15, "y": 0, "vx": .35, "flux": 8.}}]
        star = {"name": "stationary", "kind": "star", "parameters": {
            "x": [-1, 1], "y": [0, 0], "flux": [2, 12]}}
        mover = deepcopy(star)
        mover["name"] = "moving"
        mover["parameters"]["vx"] = [-.6, .6]
        sources = [star, mover]
        count = {"probabilities": [.5, .5]}
        nuisance = {"b_dx": [-.3, .3]}
        true_nuisance = {"b_dx": .12}
        if kind == "broad":
            for source in sources:
                source["parameters"].update(x=[-2, 2], y=[-2, 2], flux=[1, 12],
                                             parallax=[0, .5], variability=[-.2, .2])
            mover["parameters"].update(vx=[-1, 1], vy=[-1, 1])
            galaxy = {"name": "galaxy", "kind": "galaxy", "parameters": {
                "x": [-2, 2], "y": [-2, 2], "flux": [1, 12],
                "sigma_x": [.3, 1.5], "sigma_y": [.3, 1.5]}}
            sources.append(galaxy)
            truth.extend([{"kind": "star", "parameters": {"x": -1., "y": .5, "flux": 4., "parallax": .2}},
                          {"kind": "galaxy", "parameters": {"x": .8, "y": -.8, "flux": 6.,
                                                            "sigma_x": .8, "sigma_y": .6}}])
            nuisance["b_log_gain"] = [-.1, .1]
            count = {"poisson_mean": 2., "max_count": 3}
    images = render(exposures, truth, true_nuisance)
    for exposure, mean in zip(exposures, images):
        exposure["data"] = (mean+rng.normal(0, exposure["noise_sigma"], mean.shape)).tolist()
    return {"exposures": exposures, "source_types": sources,
            "nuisance": nuisance, "count_prior": count}, {"sources": truth, "nuisance": true_nuisance}


def confusion_reference(manifest, order):
    """Independent tensor quadrature, evaluated as direct pixel sums."""
    z, w = np.polynomial.legendre.leggauss(order)
    x, f = 1.2*z, 4.5+3.5*z
    xx, ff = np.meshgrid(x, f, indexing="ij")
    weights = (w[:, None]*w[None, :]/4).ravel()
    centers = np.arange(-2., 3.)
    px = ndtr((centers[None, :]+.5-xx.ravel()[:, None])/.7)-ndtr((centers[None, :]-.5-xx.ravel()[:, None])/.7)
    templates = ff.ravel()[:, None]*px*(ndtr(.5/.7)-ndtr(-.5/.7))
    y = np.array(manifest["exposures"][0]["data"]).ravel()
    zero = np.exp(-.5*np.sum((y/.7)**2))
    one = weights@np.exp(-.5*np.sum(((y-templates)/.7)**2, axis=1))
    two = 0.
    for template, weight in zip(templates, weights):
        two += weight*(weights@np.exp(-.5*np.sum(((y-template-templates)/.7)**2, axis=1)))
    evidence = np.array([.2*zero, .4*one, .4*two])
    return {"order": order, "scaled_evidence": float(evidence.sum()),
            "count_posterior": (evidence/evidence.sum()).tolist(),
            "scope": "independent numerical quadrature; no rigorous error bound"}


def graph_cases():
    rng = np.random.default_rng(20261020)
    a, y, p = rng.normal(size=(12, 8)), rng.normal(size=12), rng.uniform(.1, .8, 8)
    result = graph_infer(y, a, p)
    z, marginal = brute_force(y, a, p)
    rows = [{"case": "enumeration", "log_evidence_error": abs(z-result["log_evidence"]),
             "max_marginal_error": float(np.max(abs(marginal-np.array(
                 [v["probability"] for v in result["queries"].values()])))), **result}]
    for n in [10, 20, 40, 80]:
        # Each template overlaps only its two neighbors.
        a = np.zeros((n+1, n))
        for i in range(n):
            a[i:i+2, i] = [1., .4]
        rows.append({"case": f"chain_{n}", **graph_infer(rng.normal(size=n+1), a, np.full(n, .2))})
    for n in [8, 12, 16, 24]:
        try:
            rows.append({"case": f"dense_{n}", **graph_infer(rng.normal(size=8), rng.normal(size=(8, n)),
                                                         np.full(n, .2), max_entries=2**16)})
        except ValueError as e:
            rows.append({"case": f"dense_{n}", "status": "failed_factor_budget", "reason": str(e)})
    return rows


def real_scene():
    from astropy.io import fits
    from astropy.wcs import WCS
    base = Path(__file__).parents[1]/"kepler_transform/chart_results/real_inputs"
    metadata = json.loads((base/"observations.json").read_text())
    exposures, hashes, heldout = [], {}, None
    reference_wcs = None
    for i, filename in enumerate(metadata["files"]):
        path = base/filename
        hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        with fits.open(path) as hdus:
            data = hdus["SCI"].data.astype(float)
            sigma = np.sqrt(hdus["VAR"].data.astype(float))
            mask = hdus["MASK"].data != 0
            wcs = WCS(hdus["SCI"].header)
            if reference_wcs is None:
                reference_wcs = wcs
                sky = wcs.all_pix2world([[4, 4], [5, 4], [4, 5]], 0)
            projected = wcs.all_world2pix(sky, 0)
            matrix = np.column_stack([projected[1]-projected[0], projected[2]-projected[0]])
            e = {"id": filename, "instrument": "ztf", "time": metadata["mjd_tdb"][i]-metadata["mjd_tdb"][0],
                 "data": data.tolist(), "noise_sigma": sigma.tolist(), "mask": mask.tolist(),
                 "psf_sigma": metadata["sigma_pixels"][i][0], "flux_scale": metadata["flux_scales"][i],
                 "affine": matrix.tolist(), "offset": (projected[0]-4).tolist()}
            if i < 2:
                exposures.append(e)
            else:
                heldout = e
    types = [{"name": "stationary", "kind": "star", "parameters": {
        "x": [-2, 2], "y": [-2, 2], "flux": [0, 100]}}]
    moving = deepcopy(types[0])
    moving["name"] = "moving"
    moving["parameters"].update(vx=[-30, 30], vy=[-30, 30])
    types.append(moving)
    design = []
    for i, e in enumerate(exposures):
        valid = int(np.count_nonzero(~np.array(e["mask"], bool)))
        design.extend([[1., float(i == 0), float(i == 1)]]*valid)
    return {"source_types": types, "exposures": exposures, "count_prior": {"probabilities": [.5, .5]},
            "linear_nuisance": {"design": design, "mean": [170., 0., 0.],
                               "covariance": np.diag([900., 100., 100.]).tolist()}}, heldout, hashes


def compact(report):
    return {k: report[k] for k in ["status", "queries", "relative_evidence_gap",
                                  "source_count_tail_posterior_upper", "work", "full_objective_complete"]}


def run(output, selected, methods_override=None):
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output/"source_snapshot"
    snapshot.mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copy2(path, snapshot/path.name)
    versions = {"python": platform.python_version(), "numpy": np.__version__}
    summary = {"versions": versions, "full_objective_complete": False, "cases": {}}
    save(output/"summary.json", summary)
    cases = ["scalar", "graph", "confused", "moving", "broad", "real"] if selected == "all" else [selected]
    for case in cases:
        print(f"Starting {case}", flush=True)
        if case == "graph":
            result = graph_cases()
            save(output/"graph.json", result)
            summary["cases"][case] = result
        elif case == "scalar":
            m = scalar()
            response = (ndtr(.5/.6)-ndtr(-.5/.6))**2
            one = quad(lambda f: np.exp(-.5*(1-f*response)**2)/2, 1, 3, epsabs=1e-13)[0]
            zero = np.exp(-.5)
            save(output/"scalar-reference.json", {"scaled_evidence": .5*(one+zero),
                                                 "source_probability": one/(one+zero)})
            for method in ["range", "gauss", "hybrid"]:
                r = infer(m, method=method)
                save(output/f"scalar-{method}.json", r)
                save(output/f"scalar-{method}-verified.json", verify(r))
                summary["cases"][f"scalar-{method}"] = compact(r)
        elif case == "real":
            m, heldout, hashes = real_scene()
            save(output/"real-inputs.json", {"scene": m, "heldout": heldout, "sha256": hashes})
            r = infer(m, method="hybrid", queries=["count=0", "has:stationary", "has:moving"],
                      max_leaves=512, max_seconds=90)
            save(output/"real.json", r)
            save(output/"real-verified.json", verify(r))
            summary["cases"][case] = compact(r)
            # A posterior predictive calculation requires completed posterior
            # accounting. Do not silently turn an unresolved inference into one.
            save(output/"real-heldout.json", {
                "status": "not_certified" if r["status"] == "unresolved" else "inference_bounded",
                "heldout_observation_retained": True,
                "reason": "A calibrated predictive image also requires conditioning the shared background posterior.",
                "full_objective_complete": False})
        else:
            m, truth = synthetic(case)
            save(output/f"{case}-inputs.json", {"scene": m, "truth": truth})
            queries = ["count=0", "count=1", "count=2"] if case == "confused" else ["has:"+t["name"] for t in m["source_types"]]
            if case == "moving":
                queries.append("any:moving:vx:0:0.6")
            methods = methods_override or (["range", "gauss", "hybrid"] if case == "confused" else ["hybrid"])
            for method in methods:
                r = infer(m, method=method, queries=queries, max_leaves=2048, max_seconds=60)
                save(output/f"{case}-{method}.json", r)
                started = perf_counter()
                v = verify(r)
                v["replay_seconds"] = perf_counter()-started
                save(output/f"{case}-{method}-verified.json", v)
                summary["cases"][f"{case}-{method}"] = compact(r)
            if case == "confused":
                save(output/"confused-reference.json", [confusion_reference(m, n) for n in [12, 24, 40]])
        save(output/"summary.json", summary)
        print(json.dumps({"case": case, "results": {k: v for k, v in summary["cases"].items()
                                                   if k.startswith(case)}}, allow_nan=False), flush=True)
    inventory = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(output.rglob("*")) if p.is_file()}
    save(output/"inventory.json", inventory)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--case", choices=["all", "scalar", "graph", "confused", "moving", "broad", "real"], default="all")
    p.add_argument("--methods", nargs="+", choices=["range", "gauss", "hybrid", "auto"])
    args = p.parse_args()
    run(args.output, args.case, args.methods)


if __name__ == "__main__":
    main()
