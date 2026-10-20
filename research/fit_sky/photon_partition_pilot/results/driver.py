"""Exact Poisson-population identity controls, not a general FitSky solver.

Run from the repository root with the research dependencies. Output is written
only to a new directory. Geometry is fixed; one control integrates a continuous
uniform flux prior analytically. No source-count truncation enters the identity.
"""

import argparse
import hashlib
import json
from math import factorial
from pathlib import Path
from time import perf_counter

from flint import arb, ctx
import numpy as np
from scipy.integrate import quad
from scipy.stats import irwinhall, poisson

from research.fit_sky.arithmetic import lower, upper


def interval(x):
    return [lower(x), upper(x)]


def flux_moments(rate, lo, hi, maximum):
    """E[exp(-rate*F) F^k] for uniform F, in ball arithmetic."""
    rate, lo, hi = arb(rate), arb(lo), arb(hi)
    if lo == hi:
        return [(-rate*lo).exp()*lo**k for k in range(maximum+1)]
    if rate == 0:
        return [(hi**(k+1)-lo**(k+1))/((k+1)*(hi-lo)) for k in range(maximum+1)]
    left, right = (-rate*lo).exp(), (-rate*hi).exp()
    integrals = [(left-right)/rate]
    for k in range(1, maximum+1):
        integrals.append((lo**k*left-hi**k*right+k*integrals[-1])/rate)
    return [v/(hi-lo) for v in integrals]


def partitions(moments, background):
    """All set partitions, counted once by anchoring each block at its first bit."""
    n = len(background)
    table = [[arb(1)]]
    for mask in range(1, 1 << n):
        first = mask & -mask
        remainder = mask ^ first
        total = [arb(0) for _ in range(mask.bit_count()+1)]
        subset = remainder
        while True:
            block = subset | first
            rest = mask ^ block
            coefficient = sum((row[block] for row in moments), arb(0))
            for k, value in enumerate(table[rest]):
                total[k+1] += coefficient*value
                if block == first:
                    total[k] += background[first.bit_length()-1]*value
            if subset == 0:
                break
            subset = (subset-1) & remainder
        table.append(total)
    return table[-1]


def infer(counts, source_rate, classes, background, count_queries=range(10)):
    """Complete compound-Poisson catalogue posterior for supplied templates."""
    if any(not isinstance(k, int) or k < 0 for k in counts) or sum(counts) > 12:
        raise ValueError("This exponential control is limited to at most 12 observed photons")
    photons = [i for i, k in enumerate(counts) for _ in range(k)]
    n = len(photons)
    weight_total = sum((arb(c["weight"]) for c in classes), arb(0))
    moments = []
    for cls in classes:
        rates = list(map(arb, cls["response"]))
        if len(rates) != len(counts) or any(not x >= 0 for x in rates):
            raise ValueError("Nonnegative response at every observed pixel required")
        f = flux_moments(sum(rates, arb(0)), *cls["flux"], n)
        row = [arb(source_rate)*arb(cls["weight"])/weight_total*f[0]]
        for mask in range(1, 1 << n):
            product = arb(1)
            for bit, pixel in enumerate(photons):
                if mask & (1 << bit):
                    product *= rates[pixel]
            row.append(arb(source_rate)*arb(cls["weight"])/weight_total*f[mask.bit_count()]*product)
        moments.append(row)
    photon_background = [arb(background[pixel]) for pixel in photons]
    polynomial = partitions(moments, photon_background)
    total = sum(polynomial, arb(0))
    unseen = sum((row[0] for row in moments), arb(0))
    prefactor = (unseen-arb(source_rate)-sum(map(arb, background), arb(0))).exp()
    evidence = prefactor*total/np.prod([factorial(k) for k in counts], dtype=object)
    probabilities = {}
    for k in count_queries:
        p = arb(0)
        for emitting in range(min(k, n)+1):
            p += polynomial[emitting]*(-unseen).exp()*unseen**(k-emitting)/factorial(k-emitting)
        probabilities[str(k)] = interval(p/total)
    presence = {}
    for i, cls in enumerate(classes):
        other = partitions([row for j, row in enumerate(moments) if i != j], photon_background)
        absent = (-moments[i][0]).exp()*sum(other, arb(0))/total
        presence[cls["name"]] = interval(1-absent)
    return {
        "evidence": interval(evidence), "evidence_ball": str(evidence),
        "count_probabilities": probabilities,
        "has_class": presence, "unobserved_source_mean": interval(unseen),
        "emitting_source_probabilities": [interval(v/total) for v in polynomial],
        "source_count_truncated": False, "photons": n,
        "scope": "Exact fixed-geometry controls with analytic flux moments; no general continuous-geometry integration",
        "full_objective_complete": False,
    }


def fixed_reference(counts, source_rate, classes, background, maximum=24):
    """Independent direct source-count sum, with an explicit remaining tail."""
    rates = [np.array(c["response"])*c["flux"][0] for c in classes]
    weights = np.array([c["weight"] for c in classes], float)
    means = source_rate*weights/weights.sum()
    z, presence = 0., np.zeros(len(classes))
    totals = np.zeros(2*maximum+1)
    for first in range(maximum+1):
        for second in range(maximum+1):
            intensity = np.array(background)+first*rates[0]+second*rates[1]
            mass = poisson.pmf(first, means[0])*poisson.pmf(second, means[1])
            term = mass*np.prod(poisson.pmf(counts, intensity))
            z += term
            presence += term*np.array([first > 0, second > 0])
            totals[first+second] += term
    return {"evidence": z, "has_class": (presence/z).tolist(),
            "count_probabilities": (totals[:10]/z).tolist(),
            "omitted_evidence_upper": float(sum(poisson.sf(maximum, means))),
            "scope": "Independent float64 count enumeration, with analytic prior-tail upper bound"}


def continuous_reference(counts, source_rate, response, background, lo, hi):
    """Independent count sum and integration of sums of uniform fluxes."""
    response, background = np.array(response), np.array(background)
    evidence = []
    for count in range(25):
        def likelihood(summed_unit_flux):
            flux = count*lo+(hi-lo)*summed_unit_flux
            return float(np.prod(poisson.pmf(counts, background+flux*response)))
        if count == 0:
            value = likelihood(0.)
        else:
            value = sum(quad(lambda x: irwinhall.pdf(x, count)*likelihood(x),
                             j, j+1, epsabs=1e-18, epsrel=1e-10)[0]
                        for j in range(count))
        evidence.append(poisson.pmf(count, source_rate)*value)
    z = sum(evidence)
    return {"evidence": z, "count_probabilities": (np.array(evidence[:10])/z).tolist(),
            "omitted_evidence_upper": float(poisson.sf(24, source_rate)),
            "scope": "Independent float64 Irwin-Hall quadrature; quadrature error is not certified"}


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    with ctx.workprec(200):
        counts = [1, 0, 1, 2, 0, 1]
        classes = [
            {"name": "a", "weight": .6, "response": [.8, .1, .6, .2, .7, .1], "flux": [1.1, 1.1]},
            {"name": "b", "weight": .4, "response": [.1, .6, .2, .7, .1, .5], "flux": [.6, .6]},
        ]
        background = [.03]*6
        fixed = infer(counts, 2., classes, background)
        reference = fixed_reference(counts, 2., classes, background)
        assert abs(np.mean(fixed["evidence"])-reference["evidence"]) < 1e-12*reference["evidence"]
        for i, cls in enumerate(classes):
            assert abs(np.mean(fixed["has_class"][cls["name"]])-reference["has_class"][i]) < 1e-12
        for k in range(10):
            assert abs(np.mean(fixed["count_probabilities"][str(k)])
                       -reference["count_probabilities"][k]) < 1e-12
        # Thirty scalar exposures, five observed photons, unknown unbounded K
        # and continuous source flux. This is deliberately a simple control.
        counts = [int(i in (0, 6, 12, 18, 24)) for i in range(30)]
        response = [.1+.02*(i % 3) for i in range(30)]
        background = [.005]*30
        classes = [{"name": "star", "weight": 1., "response": response, "flux": [.1, 1.2]}]
        continuous = infer(counts, 2.3, classes, background)
        continuous_ref = continuous_reference(counts, 2.3, response, background, .1, 1.2)
        assert abs(np.mean(continuous["evidence"])-continuous_ref["evidence"]) < 1e-9*continuous_ref["evidence"]
        for k in range(10):
            assert abs(np.mean(continuous["count_probabilities"][str(k)])
                       -continuous_ref["count_probabilities"][k]) < 1e-9
        # No observed photons does not imply that no sources exist.
        empty = infer([0, 0], 2., [{"name": "dark", "weight": 1.,
                                   "response": [0., 0.], "flux": [1., 2.]}], [0., 0.])
        assert empty["evidence"][0] <= 1 <= empty["evidence"][1]
        assert empty["has_class"]["dark"][0] <= 1-np.exp(-2) <= empty["has_class"]["dark"][1]
    result = {"fixed": fixed, "fixed_reference": reference,
              "thirty_exposure_continuous_flux": continuous,
              "continuous_reference": continuous_ref, "unobserved_population": empty,
              "seconds": perf_counter()-started, "all_controls_passed": True,
              "original_broad_scene_solved": False, "full_objective_complete": False}
    (output/"result.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    (output/"driver.py").write_bytes(Path(__file__).read_bytes())
    (output/"inventory.json").write_text(json.dumps({
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()
    }, indent=2)+"\n")
    print(json.dumps({k: result[k] for k in ("seconds", "all_controls_passed",
                                           "original_broad_scene_solved")}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    run(p.parse_args().output)
