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


def sample(report, count=100, seed=20261020, max_proposals=100000):
    if not isinstance(count, int) or count < 1 or max_proposals < count:
        raise ValueError("Positive sample count and sufficient proposal budget required")
    # Replaying is deliberately explicit: caller should verify externally once.
    verify(report, replay=False)
    rng = np.random.default_rng(seed)
    with ctx.workprec(report["settings"]["precision"]):
        scene = Scene(report["manifest"])
        roots = scene.roots()
        cells = []
        for record in report["certificate"]:
            # Quadrature bounds are bounds on *averages*, not pointwise
            # rejection envelopes. Recompute a range bound for every proposal.
            _, hi, _, _, _ = enclose(scene, roots[record["root"]], record["path"], "range", 0)
            if hi > 0:
                cells.append({**record, "pointwise_upper": min(1., upper(hi))})
        logs = [float(roots[c["root"]]["mass"].log())-len(c["path"])*np.log(2)
                +np.log(c["pointwise_upper"]) for c in cells]
        weights = np.exp(np.asarray(logs)-logsumexp(logs))
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
            values = {**root["fixed"], **dict(zip(root["names"], map(float, parameters)))}
            sources = []
            for i, cls in enumerate(root["classes"]):
                prefix = f"s{i}."
                sources.append({"class": scene.types[cls]["name"], "kind": scene.types[cls]["kind"],
                                "parameters": {k[len(prefix):]: v for k, v in values.items() if k.startswith(prefix)}})
            draws.append({"sources": sources, "nuisance": {k.split(".", 1)[1]: v for k, v in values.items()
                                                         if k.startswith("instrument.")}})
            if len(draws) == count:
                break
        return {"catalogues": draws, "proposals": proposed, "seed": seed,
                "complete": len(draws) == count,
                "conditional_on_represented_source_counts": True,
                "excluded_posterior_mass_upper": report["source_count_tail_posterior_upper"],
                "sampling_arithmetic": "float64 RNG/proposal weights, Arb acceptance comparisons"}
