"""Exact and Monte Carlo selection for explicit finite populations and cadence rules.

These models start from supplied per-exposure detection probabilities. They do
not infer those probabilities from catalogue depth or certify orbit linkage.
"""

from pathlib import Path

import numpy as np
from scipy.special import gammaln
from scipy.stats import beta

from .io import _fields, _named_rows, _provenance, read_json
from .nulls import _weights


def validate_selection(manifest):
    _fields(manifest, ["schema_version", "exposures", "objects", "discovery_rules"],
            ["night_boundary_utc_hours", "draws_per_object", "seed", "confidence", "population_size", "observed_count"])
    if manifest["schema_version"] != 1:
        raise ValueError("Only schema_version 1 is supported")
    _named_rows(manifest["exposures"], "exposures")
    _named_rows(manifest["objects"], "objects")
    _named_rows(manifest["discovery_rules"], "discovery_rules")
    if len(manifest["exposures"]) > 4096 or len(manifest["objects"]) > 10000 or len(manifest["discovery_rules"]) > 32:
        raise ValueError("Selection supports up to 4096 exposures, 10000 objects, and 32 rules")
    boundary = manifest.get("night_boundary_utc_hours", 12)
    if not np.isfinite(boundary) or not 0 <= boundary < 24:
        raise ValueError("night_boundary_utc_hours must be in [0,24)")
    surveys, nights, memberships, sources = set(), [], [], set()
    for row in manifest["exposures"]:
        _fields(row, ["id", "observation_id", "surveys", "mjd_utc"])
        if not isinstance(row["observation_id"], str) or not row["observation_id"] or row["observation_id"] in sources:
            raise ValueError("Each independent physical observation_id must be unique; share one row between surveys")
        sources.add(row["observation_id"])
        if not isinstance(row["surveys"], list) or not row["surveys"] or not all(isinstance(s, str) and s for s in row["surveys"]) or len(set(row["surveys"])) != len(row["surveys"]):
            raise ValueError("Exposure surveys must be distinct nonempty strings")
        if not np.isfinite(row["mjd_utc"]) or not 15020 <= row["mjd_utc"] <= 88069:
            raise ValueError("Exposure times must be finite MJD UTC in 1900--2100")
        surveys.update(row["surveys"])
        memberships.append(set(row["surveys"]))
        nights.append(int(np.floor(row["mjd_utc"] - boundary / 24)))
    probabilities = []
    for row in manifest["objects"]:
        _fields(row, ["id", "weight", "detection_probability"])
        p = np.asarray(row["detection_probability"], dtype=float)
        if p.shape != (len(nights),) or not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
            raise ValueError("Each object needs a probability in [0,1] for every exposure")
        probabilities.append(p)
    weights = _weights([row["weight"] for row in manifest["objects"]])
    rules = []
    for row in manifest["discovery_rules"]:
        _fields(row, ["id", "surveys", "min_visits_per_night", "min_nights"], ["max_night_span"])
        selected = row["surveys"]
        if not isinstance(selected, list) or not selected or not set(selected) <= surveys or len(set(selected)) != len(selected):
            raise ValueError("Rule surveys must be distinct known survey names")
        for key in ["min_visits_per_night", "min_nights"]:
            if isinstance(row[key], bool) or not isinstance(row[key], int) or not 1 <= row[key] <= 4096:
                raise ValueError("Rule minimum counts must be integers in [1,4096]")
        span = row.get("max_night_span")
        if span is not None and (isinstance(span, bool) or not isinstance(span, int) or not 0 <= span <= 100000):
            raise ValueError("max_night_span must be a nonnegative integer")
        indices = [i for i, membership in enumerate(memberships) if membership & set(selected)]
        night_groups = [(night, [i for i in indices if nights[i] == night]) for night in sorted({nights[i] for i in indices})]
        rules.append((row, night_groups))
    if "observed_count" in manifest and "population_size" not in manifest:
        raise ValueError("observed_count requires population_size")
    if "population_size" in manifest and (not np.isfinite(manifest["population_size"]) or manifest["population_size"] < 0):
        raise ValueError("population_size must be a finite nonnegative Poisson mean")
    if "observed_count" in manifest and (isinstance(manifest["observed_count"], bool) or not isinstance(manifest["observed_count"], int) or manifest["observed_count"] < 0):
        raise ValueError("observed_count must be a nonnegative integer")
    return np.array(probabilities), weights, rules


def apply_rules(detections, rules):
    """Each row is one shared realization, evaluated by every survey rule."""
    outcomes = []
    for rule, groups in rules:
        qualified = np.column_stack([detections[:, indices].sum(axis=1) >= rule["min_visits_per_night"] for _, indices in groups])
        if "max_night_span" not in rule:
            selected = qualified.sum(axis=1) >= rule["min_nights"]
        else:
            selected = np.zeros(len(detections), dtype=bool)
            nights = np.array([night for night, _ in groups])
            # Windows anchored at every observing night cover every possible
            # qualifying consecutive block, including gaps in the schedule.
            cumulative = np.pad(np.cumsum(qualified, axis=1), ((0,0),(1,0)))
            for left, night in enumerate(nights):
                right = int(np.searchsorted(nights, night + rule["max_night_span"], side="right"))
                selected |= cumulative[:, right] - cumulative[:, left] >= rule["min_nights"]
        outcomes.append(selected)
    matrix = np.column_stack(outcomes)
    return np.column_stack([matrix, matrix.any(axis=1)])


def _binomial_interval(successes, trials, confidence):
    tail = (1 - confidence) / 2
    return [float(beta.ppf(tail, successes, trials-successes+1)) if successes else 0.,
            float(beta.ppf(1-tail, successes+1, trials-successes)) if successes < trials else 1.]


def run_selection(manifest_path, monte_carlo=False):
    path = Path(manifest_path).resolve()
    manifest = read_json(path)
    probabilities, weights, rules = validate_selection(manifest)
    n_objects, n_exposures = probabilities.shape
    ids = [row["id"] for row, _ in rules] + ["union"]
    if "union" in ids[:-1]:
        raise ValueError("Rule ID 'union' is reserved")
    confidence = manifest.get("confidence", .95)
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be in (0,1)")
    if monte_carlo:
        draws = manifest.get("draws_per_object", 10000)
        seed = manifest.get("seed", 2026)
        if isinstance(draws, bool) or not isinstance(draws, int) or not 100 <= draws <= 1000000:
            raise ValueError("draws_per_object must be an integer in [100,1000000]")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32-1:
            raise ValueError("seed must be an integer in [0,2**32-1]")
        if draws*n_exposures*n_objects*len(rules) > 100000000:
            raise ValueError("Experiment exceeds 100 million rule-exposure draws; split explicitly")
        rng = np.random.default_rng(seed)
        estimates, object_reports = [], []
        for obj, p in zip(manifest["objects"], probabilities):
            successes = np.zeros(len(ids), dtype=np.int64)
            for start in range(0, draws, 4096):
                outcomes = apply_rules(rng.random((min(4096, draws-start), n_exposures)) < p, rules)
                successes += outcomes.sum(axis=0)
            estimated = successes / draws
            estimates.append(estimated)
            object_reports.append({"id": obj["id"], "successes": dict(zip(ids, successes.tolist())),
                                   "probabilities": dict(zip(ids, estimated.tolist())),
                                   "union_binomial_interval": _binomial_interval(int(successes[-1]), draws, confidence)})
        # Independent simulations per object: weighted Hoeffding bound. This
        # interval covers Monte Carlo error, not calibration/population error.
        radius = float(np.sqrt(np.log(2/(1-confidence)) * np.sum(weights**2) / (2*draws)))
        algorithm = {"method": "independent Bernoulli exposure draws, shared across survey rules", "seed": seed, "draws_per_object": draws, "confidence": confidence}
    else:
        if any(key in manifest for key in ["draws_per_object", "seed"]):
            raise ValueError("draws_per_object and seed apply to inject_all, not exact map_complete")
        if n_exposures > 20 or 2**n_exposures*n_exposures*n_objects*len(rules) > 50000000:
            raise ValueError("Exact enumeration exceeds its 20-exposure/50-million-operation budget; use inject_all")
        estimates = np.zeros((n_objects, len(ids)))
        probability_mass = np.zeros(n_objects)
        for start in range(0, 2**n_exposures, 4096):
            states = np.arange(start, min(start+4096, 2**n_exposures), dtype=np.uint64)
            detections = ((states[:, None] >> np.arange(n_exposures, dtype=np.uint64)) & 1).astype(bool)
            outcomes = apply_rules(detections, rules)
            for i, p in enumerate(probabilities):
                mass = np.prod(np.where(detections, p, 1-p), axis=1)
                estimates[i] += mass @ outcomes
                probability_mass[i] += mass.sum()
        if not np.allclose(probability_mass, 1, atol=1e-12, rtol=1e-12):
            raise ValueError("Enumeration failed probability-mass normalization")
        estimates = np.clip(estimates, 0, 1)
        object_reports = [{"id": obj["id"], "probabilities": dict(zip(ids, p.tolist()))} for obj, p in zip(manifest["objects"], estimates)]
        radius = 0.
        algorithm = {"method": "exhaustive Bernoulli detection-outcome enumeration", "outcomes_per_object": 2**n_exposures}
    average = weights @ estimates
    low, high = max(0., average[-1]-radius), min(1., average[-1]+radius)
    population = None
    if "population_size" in manifest:
        count = manifest["population_size"]
        mean = float(count * average[-1])
        population = {"poisson_population_mean": count, "expected_selected_count": mean,
                      "expected_count_mc_interval": [count*low, count*high] if monte_carlo else None}
        if "observed_count" in manifest:
            observed = manifest["observed_count"]
            log_likelihood = float(observed*np.log(mean)-mean-gammaln(observed+1)) if mean > 0 else (0. if observed == 0 else None)
            population.update(observed_count=observed, log_count_likelihood=log_likelihood,
                              zero_probability=mean == 0 and observed > 0,
                              likelihood_uses_estimated_efficiency=monte_carlo)
    return {"schema_version": 1, "tool": "inject_all" if monte_carlo else "map_complete", "provenance": _provenance([path]),
            "algorithm": algorithm, "object_weights": weights.tolist(), "objects": object_reports,
            "population_probabilities": dict(zip(ids, average.tolist())),
            "union_mc_interval": [low, high] if monte_carlo else None, "population_count_model": population,
            "scope": "Finite supplied population and cadence rules under conditionally independent per-exposure detections. Shared physical exposures are sampled once across surveys; union probabilities preserve this dependence. Probabilities must already include coverage and calibrated detection selection. Cadence qualification does not prove orbit linkage. Calibration, population, weather and other cross-exposure uncertainties are not simulated. Count likelihood assumes a Poisson population and independent object selection."}
