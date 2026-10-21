"""Budgeted ITF linkage reference, conditional on an explicit finite orbit prior.

No candidate gating: the partition frontier covers every assignment in the
admitted small scene. Bounds concern search truncation, not numerical rounding,
continuous-orbit quadrature, model adequacy or completeness of the ITF archive.
"""
from functools import lru_cache
import heapq
import math
import time

import numpy as np
from scipy.special import logsumexp

from .orbits import C_AU_DAY
from .solve_itf_contract import SCHEMA


def validate(value):
    from jsonschema import Draft202012Validator
    Draft202012Validator(SCHEMA).validate(value)

    def unique(values, label):
        if len(set(values)) != len(values):
            raise ValueError("Duplicate " + label)
    ids = [t["id"] for t in value["tracklets"]]
    unique(ids, "tracklet ID")
    unique([o["id"] for t in value["tracklets"] for o in t["observations"]], "observation ID")
    unique([o["id"] for o in value["orbit_prior"]], "orbit ID")
    for query in value.get("queries", []):
        unique(query, "query member")
        if not set(query) <= set(ids):
            raise ValueError("Query refers to an unknown tracklet")
    numbers = [value["epoch_mjd_tdb"], value["link_activity"], value.get("max_seconds", 180)]
    for t in value["tracklets"]:
        for o in t["observations"]:
            numbers.extend([o["mjd_tdb"], o["ra_deg"], o["dec_deg"], o["sigma_arcsec"],
                            *o["observer_heliocentric_au"]])
    states = np.array([r["state"] for r in value["orbit_prior"]], float)
    weights = np.array([r["weight"] for r in value["orbit_prior"]], float)
    if not (np.all(np.isfinite(numbers)) and np.all(np.isfinite(states))
            and np.all(np.isfinite(weights))):
        raise ValueError("All geometry, budgets and weights must be finite")
    if np.any(np.linalg.norm(states[:, :3], axis=1) < 1e-5):
        raise ValueError("Orbit radius must be at least 1e-5 AU")
    if np.any(np.linalg.norm(states[:, 3:], axis=1) >= C_AU_DAY):
        raise ValueError("Orbit speed must be subluminal")
    return value


def log_likelihood_ratios(value, deadline=math.inf, clock=time.monotonic):
    """One log likelihood ratio per tracklet and supplied orbit.

    Observations follow an isotropic von Mises-Fisher density on the sphere,
    versus independent uniform directions (1/4pi) for the unlinked baseline.
    Sigma is the small-angle standard deviation in each tangent direction.
    This deliberately simple baseline is not a calibrated survey background.
    """
    from research.kepler_transform.universal import apparent
    tracks = value["tracklets"]
    observations = [o for t in tracks for o in t["observations"]]
    times = np.array([o["mjd_tdb"] - value["epoch_mjd_tdb"] for o in observations])
    observers = np.array([o["observer_heliocentric_au"] for o in observations])
    angles = np.deg2rad([[o["ra_deg"], o["dec_deg"]] for o in observations])
    ra, dec = angles.T
    measured = np.column_stack((np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)))
    measured /= np.linalg.norm(measured, axis=1)[:, None]
    kappa = 1 / np.deg2rad(np.array([o["sigma_arcsec"] for o in observations])/3600)**2
    # Normalized directional density divided by 1/(4*pi).
    peak = np.log(2*kappa) - np.log(-np.expm1(-2*kappa))
    states = np.array([r["state"] for r in value["orbit_prior"]], float)
    result = np.empty((len(tracks), len(states)))
    boundaries = np.cumsum([0] + [len(t["observations"]) for t in tracks])
    for start in range(0, len(states), 32):
        if clock() >= deadline:
            return None, start
        stop = min(start+32, len(states))
        emitted, _ = apparent(states[start:stop], times, observers)
        vectors = emitted[..., :3]-observers[None]
        direction = vectors/np.linalg.norm(vectors, axis=2)[..., None]
        # Avoid subtracting nearly equal dot products for sub-arcsecond offsets.
        penalty = .5*np.sum((direction-measured[None])**2, axis=2)*kappa
        log_ratio = peak[None]-penalty
        for j, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            result[j, start:stop] = log_ratio[:, left:right].sum(axis=1)
    if not np.all(np.isfinite(result)):
        raise ArithmeticError("Nonfinite likelihood; no orbit was silently excluded")
    return result, len(states)


def partition_search(log_ratios, log_prior, activity, *, max_nodes=100000,
                     deadline=math.inf, clock=time.monotonic, top_k=10,
                     queries=(), target_unresolved_mass=0):
    """Disjoint restricted-growth partition frontier with analytic mass ceilings.

    Every singleton is unlinked with weight 1. Every block S of size >=2 has
    weight activity * sum_r prior[r] * product_{j in S} ratio[j,r].
    The partition weight is the product of its block weights. Normalization is
    over ALL set partitions, not over a candidate shortlist.
    """
    ratios, prior = np.asarray(log_ratios, float), np.asarray(log_prior, float)
    if (ratios.ndim != 2 or not 1 <= len(ratios) <= 12 or not ratios.shape[1]
            or prior.shape != (ratios.shape[1],)
            or not np.all(np.isfinite(ratios)) or not np.all(np.isfinite(prior))
            or not math.isfinite(activity) or activity <= 0):
        raise ValueError("Finite log likelihoods, log weights and positive activity required")
    prior = prior-logsumexp(prior)
    n = len(ratios)
    if (not isinstance(max_nodes, int) or max_nodes < 0 or top_k < 1
            or not 0 <= target_unresolved_mass < 1):
        raise ValueError("Invalid search budget")
    queries = [frozenset(q) for q in queries]
    if any(len(q) < 2 or not q <= set(range(n)) for q in queries):
        raise ValueError("Invalid query indices")
    log_activity = math.log(activity)

    @lru_cache(None)
    def block_score(block):
        if len(block) == 1:
            return 0.
        return float(log_activity+logsumexp(prior+ratios[list(block)].sum(axis=0)))

    @lru_cache(None)
    def completions(remaining, blocks):
        if remaining == 0:
            return 1
        return blocks*completions(remaining-1, blocks)+completions(remaining-1, blocks+1)

    # A prefix singleton i can later mature into a linked block, costing at most
    # max(1, activity * E[ratio_i]). Each suffix row contributes at most
    # max(1, activity) * max(1, max_r ratio_j). See the derivation in the docs.
    singleton = np.maximum(0, log_activity+logsumexp(prior[None]+ratios, axis=1))
    suffix = np.r_[np.cumsum((max(0., log_activity)+np.maximum(0, ratios.max(axis=1)))[::-1])[::-1], 0.]

    def bound(partition, k):
        weight = sum(block_score(b) for b in partition)
        if k == n:
            # A terminal singleton cannot mature: its exact weight is available.
            return float(weight), float(weight), float(weight)
        ceiling = weight+sum(singleton[b[0]] for b in partition if len(b) == 1)+suffix[k]
        return float(ceiling+math.log(completions(n-k, len(partition)))), float(ceiling), float(weight)

    mass_bound, ceiling, _ = bound((), 0)
    frontier = [(-mass_bound, 0, (), 0, ceiling)]
    serial = 1
    log_known = -math.inf
    events = {}
    query_mass = [-math.inf for _ in queries]
    best = []
    visited = completed = 0
    reason = "exhausted"
    while frontier:
        if visited >= max_nodes:
            reason = "node_budget"
            break
        if clock() >= deadline:
            reason = "time_budget"
            break
        _, _, partition, k, _ = heapq.heappop(frontier)
        visited += 1
        if k == n:
            score = sum(block_score(b) for b in partition)
            completed += 1
            log_known = float(np.logaddexp(log_known, score))
            for block in partition:
                if len(block) >= 2:
                    events[block] = float(np.logaddexp(events.get(block, -math.inf), score))
            for j, query in enumerate(queries):
                if any(query <= set(block) for block in partition):
                    query_mass[j] = float(np.logaddexp(query_mass[j], score))
            heapq.heappush(best, (score, partition))
            if len(best) > top_k:
                heapq.heappop(best)
            if target_unresolved_mass and completed % 32 == 0 and frontier:
                pending = float(logsumexp([-row[0] for row in frontier]))
                if math.exp(pending-np.logaddexp(log_known, pending)) <= target_unresolved_mass:
                    reason = "mass_target"
                    break
            continue
        # All children are inserted before checking budgets again. No subtree
        # disappears merely because it looks implausible or a deadline passes.
        children = [partition+((k,),)]
        children += [partition[:j]+(block+(k,),)+partition[j+1:]
                     for j, block in enumerate(partition)]
        for child in children:
            upper, maximum, _ = bound(child, k+1)
            heapq.heappush(frontier, (-upper, serial, child, k+1, maximum))
            serial += 1

    pending = float(logsumexp([-row[0] for row in frontier])) if frontier else -math.inf
    total_upper = float(np.logaddexp(log_known, pending))
    def upper_probability(log_mass, log_denominator):
        if not math.isfinite(log_mass):
            return 0.
        # Preserve a positive ceiling when a nonempty contribution underflows.
        return min(1., max(float(np.nextafter(0., 1.)), math.exp(min(0., log_mass-log_denominator))))

    unresolved = upper_probability(pending, total_upper) if frontier else 0.

    def interval(log_event):
        lower = math.exp(log_event-total_upper) if math.isfinite(log_event) else 0.
        upper = upper_probability(float(np.logaddexp(log_event, pending)), total_upper)
        return [max(0., min(1., lower)), max(0., min(1., upper))]

    candidates = []
    for block, mass in events.items():
        conditional = prior+ratios[list(block)].sum(axis=0)
        conditional -= logsumexp(conditional)
        candidates.append({"indices": list(block), "probability_interval": interval(mass),
                           "orbit_probabilities_given_block": np.exp(conditional).tolist()})
    candidates.sort(key=lambda row: (-row["probability_interval"][0], row["indices"]))
    best.sort(reverse=True)
    assignments = [{"blocks": [list(b) for b in p], "log_weight": float(score),
                    # An exact complete assignment cannot occur in the frontier.
                    "probability_interval": [
                        math.exp(score-total_upper),
                        upper_probability(score, log_known)]}
                   for score, p in best]
    max_unseen = max((row[4] for row in frontier), default=-math.inf)
    remaining_count = sum(completions(n-row[3], len(row[2])) for row in frontier)
    return {
        "status": "finite_prior_complete" if not frontier else "incomplete",
        "stop_reason": reason,
        "completed_assignments": completed, "remaining_assignments": remaining_count,
        "total_assignments": completions(n, 0), "visited_nodes": visited,
        "frontier_nodes": len(frontier), "unresolved_mass_upper": unresolved,
        "log_unresolved_mass_upper": pending-total_upper if frontier else None,
        "log_partition_function_interval": [
            None if not math.isfinite(log_known) else log_known, total_upper],
        "ranked_candidates": candidates[:top_k],
        "candidate_events_evaluated": len(candidates),
        "candidate_events_not_returned": max(0, len(candidates)-top_k),
        "candidate_ranking": "by lower probability bound; ranks may overlap",
        "unencountered_candidate_probability_upper": unresolved,
        "ranked_assignments": assignments,
        "returned_assignment_order_resolved": bool(best) and max_unseen < best[-1][0],
        "queries": [{"indices": sorted(q), "probability_interval": interval(m)}
                    for q, m in zip(queries, query_mass)],
        "bound_scope": "analytic search truncation bounds evaluated in float64; not directed-rounding certificates",
    }


def run(value):
    started = time.monotonic()
    validate(value)
    deadline = started+value.get("max_seconds", 180.)
    geometry_started = time.monotonic()
    ratios, processed = log_likelihood_ratios(value, deadline)
    geometry_finished = time.monotonic()
    ids = [t["id"] for t in value["tracklets"]]
    if ratios is None:
        report = {"status": "incomplete", "stop_reason": "time_budget_during_likelihoods",
                  "unresolved_mass_upper": 1., "ranked_candidates": [], "ranked_assignments": [],
                  "queries": [{"tracklets": q, "probability_interval": [0., 1.]}
                              for q in value.get("queries", [])]}
    else:
        report = partition_search(
            ratios, np.log([o["weight"] for o in value["orbit_prior"]]), value["link_activity"],
            max_nodes=value.get("max_nodes", 100000), deadline=deadline,
            top_k=value.get("top_k", 10),
            queries=[[ids.index(i) for i in q] for q in value.get("queries", [])],
            target_unresolved_mass=value.get("target_unresolved_mass", 0.))
        for row in report["ranked_candidates"]+report["queries"]:
            row["tracklets"] = [ids[i] for i in row.pop("indices")]
        for row in report["ranked_assignments"]:
            row["linked_groups"] = [[ids[i] for i in b] for b in row["blocks"] if len(b) >= 2]
            row["unlinked"] = [ids[b[0]] for b in row.pop("blocks") if len(b) == 1]
    report.update(
        tool="solve_itf", schema_version=1, snapshot_id=value["snapshot_id"],
        full_objective_complete=False, full_itf_search_complete=False,
        continuous_orbit_coverage_certified=False, excluded_continuous_orbit_mass_upper=None,
        posterior_scope="Conditional finite orbit prior and declared partition potentials only",
        observation_model="Independent isotropic von Mises-Fisher errors; uniform-direction unlinked baseline",
        dynamics="Heliocentric two-body with reception-observer light time",
        orbit_ids=[o["id"] for o in value["orbit_prior"]],
        orbit_states=[o["state"] for o in value["orbit_prior"]],
        epoch_mjd_tdb=value["epoch_mjd_tdb"], frame=value["frame"],
        tracklet_count=len(ids), orbits_processed=processed,
        numerical_bounds_certified=False,
        timings={"validation_seconds": geometry_started-started,
                 "likelihood_seconds": geometry_finished-geometry_started,
                 "search_and_report_seconds": time.monotonic()-geometry_finished,
                 "engine_seconds": time.monotonic()-started,
                 "requested_seconds": value.get("max_seconds", 180.),
                 "scope": "Engine validation, likelihood construction and search; CLI JSON IO is additional",
                 "hard_deadline_guaranteed": False})
    return report
