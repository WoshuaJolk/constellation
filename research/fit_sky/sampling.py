"""Posterior catalogue draws from the bounded representation.

Conventional floating RNG/proposal probabilities; not an arithmetic certificate
of the random-number generator. Draws are conditional on represented source
counts, with the excluded posterior tail bounded by the inference report.
"""

import numpy as np
from flint import arb, ctx
from scipy.special import logsumexp

from .integrate import _box, enclose, verify
from .arithmetic import upper
from .scene import Scene


def _catalogue(scene, root, parameters):
    values = {**root["fixed"], **dict(zip(root["names"], map(float, parameters)))}
    sources = []
    for i, cls in enumerate(root["classes"]):
        prefix = f"s{i}."
        sources.append({"class": scene.types[cls]["name"], "kind": scene.types[cls]["kind"],
                        "parameters": {k[len(prefix):]: v for k, v in values.items() if k.startswith(prefix)}})
    return {"sources": sources, "nuisance": {k.split(".", 1)[1]: v for k, v in values.items()
                                           if k.startswith("instrument.")}}


def sample(report, count=100, seed=20261020, max_proposals=100000, *,
           noise_floor_certificate=None, conditional_noise_certificate=None,
           fast=False, batch_size=2048):
    if not isinstance(count, int) or count < 1 or max_proposals < count:
        raise ValueError("Positive sample count and sufficient proposal budget required")
    # Replaying is deliberately explicit: caller should verify externally once.
    embedded = report["settings"]["method"] == "noise-floor"
    verify(report, replay=embedded)
    rng = np.random.default_rng(seed)
    with ctx.workprec(report["settings"]["precision"]):
        scene = Scene(report["manifest"])
        roots = scene.roots()
        if noise_floor_certificate is not None:
            from .noise_floor import verify as verify_floor, ceiling
            verify_floor(scene, noise_floor_certificate)
        local = None
        if conditional_noise_certificate is not None:
            from .noise_floor import verify_cover
            local = verify_cover(scene, report, conditional_noise_certificate)
        cells = []
        for index, record in enumerate(report["certificate"]):
            # Quadrature bounds are bounds on *averages*, not pointwise
            # rejection envelopes. Recompute a range bound for every proposal.
            _, hi, _, _, _ = enclose(scene, roots[record["root"]], record["path"], "range", 0)
            if embedded:
                hi = min(hi, arb(record["likelihood"][1]))
            if noise_floor_certificate is not None:
                hi = min(hi, ceiling(noise_floor_certificate, roots[record["root"]]))
            if local is not None:
                hi = min(hi, local[index])
            if hi > 0:
                cells.append({**record, "pointwise_upper": min(1., upper(hi))})
        logs = [float(roots[c["root"]]["mass"].log())-len(c["path"])*np.log(2)
                +np.log(c["pointwise_upper"]) for c in cells]
        weights = np.exp(np.asarray(logs)-logsumexp(logs))
        if np.any(weights <= 0) or not np.all(np.isfinite(weights)):
            raise ArithmeticError("Proposal weight underflow; cannot omit a prior cell")
        if fast:
            from .statistical import BatchModel
            model = BatchModel(scene)
            domains = [np.array([[float(a), float(b)] for a, b in _box(roots[c["root"]], c["path"])]).reshape(-1, 2)
                       for c in cells]
            draws, evaluated = [], 0
            if not isinstance(batch_size, int) or batch_size < 1:
                raise ValueError("Positive integer batch size required")
            for start in range(0, max_proposals, batch_size):
                n = min(batch_size, max_proposals-start)
                choices = rng.choice(len(cells), n, p=weights)
                uniforms = rng.random(n)
                accepted = {}
                for key in np.unique(choices):
                    indices = np.flatnonzero(choices == key)
                    cell, domain = cells[key], domains[key]
                    root = roots[cell["root"]]
                    points = domain[:, 0]+rng.random((len(indices), len(domain)))*(domain[:, 1]-domain[:, 0])
                    probability = model.likelihood(root, points)/cell["pointwise_upper"]
                    if np.any(probability > 1+1e-9):
                        raise ArithmeticError("Rejection envelope violated")
                    for index, point in zip(indices[uniforms[indices] < np.minimum(probability, 1)],
                                            points[uniforms[indices] < np.minimum(probability, 1)]):
                        accepted[int(index)] = _catalogue(scene, root, point)
                evaluated += n
                for index in sorted(accepted):
                    draws.append(accepted[index])
                    if len(draws) == count:
                        return {"catalogues": draws, "proposals": start+index+1, "evaluated_proposals": evaluated,
                                "seed": seed, "complete": True, "conditional_on_represented_source_counts": True,
                                "excluded_posterior_mass_upper": report["source_count_tail_posterior_upper"],
                                "sampling_arithmetic": "float64 RNG, proposal weights and forward-model acceptance"}
            return {"catalogues": draws, "proposals": evaluated, "evaluated_proposals": evaluated,
                    "seed": seed, "complete": False, "conditional_on_represented_source_counts": True,
                    "excluded_posterior_mass_upper": report["source_count_tail_posterior_upper"],
                    "sampling_arithmetic": "float64 RNG, proposal weights and forward-model acceptance"}
        draws, proposed = [], 0
        for proposed in range(1, max_proposals+1):
            cell = cells[rng.choice(len(cells), p=weights)]
            root = roots[cell["root"]]
            intervals = _box(root, cell["path"])
            parameters = [(lo+(hi-lo)*arb(float(u))) for (lo, hi), u in zip(intervals, rng.random(len(intervals)))]
            likelihood = scene.likelihood(root, parameters)
            threshold = arb(float(rng.random()))*arb(cell["pointwise_upper"])
            if not likelihood > threshold:
                if not likelihood < threshold:
                    raise ArithmeticError("Acceptance decision unresolved at selected precision")
                continue
            draws.append(_catalogue(scene, root, parameters))
            if len(draws) == count:
                break
        return {"catalogues": draws, "proposals": proposed, "seed": seed,
                "complete": len(draws) == count,
                "conditional_on_represented_source_counts": True,
                "excluded_posterior_mass_upper": report["source_count_tail_posterior_upper"],
                "sampling_arithmetic": "float64 RNG/proposal weights, Arb acceptance comparisons"}
