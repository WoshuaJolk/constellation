"""Exact finite orbit-candidate assignment with a checkable integer certificate.

This solves weighted set packing, not orbit generation. Scores are supplied
integers; no floating-point solver tolerances enter the optimality guarantee.
"""

from pathlib import Path

from .io import _fields, _named_rows, _provenance, read_json


def _problem(value):
    _fields(value, ["detections", "candidates"], ["score_units"])
    detection_ids = _named_rows(value["detections"], "detections")
    if len(detection_ids) > 4096 or not isinstance(value["candidates"], list) or len(value["candidates"]) > 128:
        raise ValueError("Assignment supports at most 4096 detections and 128 candidates")
    if value["candidates"]:
        _named_rows(value["candidates"], "candidates")
    background = {}
    for detection in value["detections"]:
        _fields(detection, ["id", "unassigned_score"])
        score = detection["unassigned_score"]
        if isinstance(score, bool) or not isinstance(score, int):
            raise ValueError("All assignment scores must be exact integers")
        background[detection["id"]] = score
    candidates = []
    for row in value["candidates"]:
        _fields(row, ["id", "detection_ids", "score"])
        ids = row["detection_ids"]
        if not isinstance(ids, list) or not ids or not all(isinstance(i,str) for i in ids) or not set(ids) <= detection_ids or len(set(ids)) != len(ids):
            raise ValueError("Candidate detection_ids must be distinct known detections")
        if isinstance(row["score"], bool) or not isinstance(row["score"], int):
            raise ValueError("All assignment scores must be exact integers")
        benefit = row["score"] - sum(background[i] for i in ids)
        # An optional, disjoint candidate of nonpositive benefit cannot improve
        # the optimum: leave its detections unassigned instead.
        if benefit > 0:
            candidates.append(dict(id=row["id"], detections=set(ids), benefit=benefit))
    return background, candidates


def _clique_partition(remaining, candidates):
    """Partition remaining candidates into pairwise-conflicting cliques.

    At most one candidate per clique may be selected. Summing each clique's
    maximum benefit is an integer upper bound, independently checkable.
    """
    groups = []
    for i in sorted(remaining, key=lambda i: (-candidates[i]["benefit"], candidates[i]["id"])):
        for group in groups:
            if all(candidates[i]["detections"] & candidates[j]["detections"] for j in group):
                group.append(i)
                break
        else:
            groups.append([i])
    return groups


def solve_assignment(problem, node_limit=100000):
    background, candidates = _problem(problem)
    if isinstance(node_limit,bool) or not isinstance(node_limit,int) or not 1 <= node_limit <= 100000:
        raise ValueError("node_limit must be an integer in [1,100000]")
    base = sum(background.values())
    conflicts = [{j for j,c in enumerate(candidates) if candidates[i]["detections"] & c["detections"]} for i in range(len(candidates))]
    # Feasible greedy warm start; exact search is responsible for the guarantee.
    incumbent, incumbent_score, occupied = [], base, set()
    for i in sorted(range(len(candidates)),key=lambda i:(-candidates[i]["benefit"],candidates[i]["id"])):
        if not occupied & candidates[i]["detections"]:
            incumbent.append(i); incumbent_score += candidates[i]["benefit"]
            occupied |= candidates[i]["detections"]
    visited = 0
    def search(remaining, selected, score):
        nonlocal visited, incumbent, incumbent_score
        cliques = _clique_partition(remaining,candidates)
        upper = score + sum(max(candidates[i]["benefit"] for i in group) for group in cliques)
        if score > incumbent_score:
            incumbent, incumbent_score = selected.copy(), score
        if upper <= incumbent_score or not remaining or visited >= node_limit:
            return {"cliques": [[candidates[i]["id"] for i in group] for group in cliques]}
        visited += 1
        pivot = max(remaining,key=lambda i:(candidates[i]["benefit"],len(conflicts[i]&remaining)))
        included = search(remaining-conflicts[pivot],selected+[pivot],score+candidates[pivot]["benefit"])
        excluded = search(remaining-{pivot},selected,score)
        return {"pivot": candidates[pivot]["id"], "include": included, "exclude": excluded}
    tree = search(set(range(len(candidates))),[],base)
    chosen = sorted(candidates[i]["id"] for i in incumbent)
    certificate = {"problem": problem, "selected_candidate_ids": chosen, "tree": tree}
    verified = verify_certificate(certificate)
    selected_rows = [row for row in problem["candidates"] if row["id"] in chosen]
    used = {eid for row in selected_rows for eid in row["detection_ids"]}
    return {"status": "optimal" if verified["gap"] == 0 else "bounded", **verified,
            "selected_candidate_ids": chosen, "unassigned_detection_ids": sorted(set(background)-used),
            "expanded_nodes": visited, "node_limit": node_limit, "certificate": certificate,
            "scope": "Exact integer objective over supplied optional orbit candidates, at most one use of each detection, with unassigned scores. Guarantees concern this finite candidate set and score model only; no physical orbit completeness, false-link probability, or uniqueness guarantee."}


def verify_certificate(certificate):
    """Check feasibility and every bound/branch; do not run the optimizer."""
    _fields(certificate,["problem","selected_candidate_ids","tree"])
    background,candidates = _problem(certificate["problem"])
    lookup = {c["id"]:c for c in candidates}
    selected = certificate["selected_candidate_ids"]
    if not isinstance(selected,list) or not all(isinstance(i,str) for i in selected) or len(set(selected)) != len(selected) or not set(selected) <= lookup.keys():
        raise ValueError("Certificate selected candidates are invalid")
    used=set(); objective=sum(background.values())
    for cid in selected:
        if used & lookup[cid]["detections"]:
            raise ValueError("Certificate assigns a detection more than once")
        used |= lookup[cid]["detections"]
        objective += lookup[cid]["benefit"]
    nodes=0
    def verify(node,remaining,score,depth):
        nonlocal nodes
        nodes += 1
        if nodes > 200257 or depth > 128:
            raise ValueError("Certificate exceeds node/depth limits")
        if "pivot" in node:
            _fields(node,["pivot","include","exclude"])
            pivot=node["pivot"]
            if pivot not in remaining:
                raise ValueError("Certificate branch pivot is not a remaining candidate")
            compatible={cid for cid in remaining if not lookup[cid]["detections"] & lookup[pivot]["detections"]}
            included=verify(node["include"],compatible,score+lookup[pivot]["benefit"],depth+1)
            excluded=verify(node["exclude"],remaining-{pivot},score,depth+1)
            return max(included,excluded)
        _fields(node,["cliques"])
        if not isinstance(node["cliques"],list):
            raise ValueError("Certificate cliques must be a list")
        covered=set(); upper=score
        for clique in node["cliques"]:
            if not isinstance(clique,list) or not clique or not all(isinstance(i,str) for i in clique) or len(set(clique)) != len(clique) or not set(clique) <= remaining or covered & set(clique):
                raise ValueError("Certificate clique partition is invalid")
            for i,cid in enumerate(clique):
                if any(not lookup[cid]["detections"] & lookup[other]["detections"] for other in clique[i+1:]):
                    raise ValueError("Certificate clique contains compatible candidates")
            covered.update(clique)
            upper += max(lookup[cid]["benefit"] for cid in clique)
        if covered != remaining:
            raise ValueError("Certificate omitted remaining candidates")
        return upper
    upper=verify(certificate["tree"],set(lookup),sum(background.values()),0)
    if upper < objective:
        raise ValueError("Certificate bound contradicts feasible objective")
    return {"objective": objective,"upper_bound": upper,"gap": upper-objective,"verified_certificate_nodes": nodes}


def run_match(manifest_path):
    path=Path(manifest_path).resolve()
    manifest=read_json(path)
    _fields(manifest,["schema_version","detections","candidates"],["score_units","node_limit"])
    if manifest["schema_version"] != 1:
        raise ValueError("Only schema_version 1 is supported")
    problem={k:v for k,v in manifest.items() if k not in {"schema_version","node_limit"}}
    return {"schema_version":1,"tool":"match_all","provenance":_provenance([path]),**solve_assignment(problem,manifest.get("node_limit",100000))}


def run_verify_match(report_path):
    path=Path(report_path).resolve()
    report=read_json(path)
    if report.get("tool") != "match_all" or report.get("schema_version") != 1:
        raise ValueError("Supply a match_all report")
    result=verify_certificate(report["certificate"])
    for key in ["objective","upper_bound","gap"]:
        if report.get(key) != result[key]:
            raise ValueError(f"Report {key} differs from its certificate")
    if report.get("selected_candidate_ids") != report["certificate"]["selected_candidate_ids"]:
        raise ValueError("Report selected candidates differ from certificate")
    problem = report["certificate"]["problem"]
    used = {eid for row in problem["candidates"] if row["id"] in report["selected_candidate_ids"] for eid in row["detection_ids"]}
    unmatched = sorted({row["id"] for row in problem["detections"]} - used)
    if report.get("unassigned_detection_ids") != unmatched:
        raise ValueError("Report unassigned detections differ from certificate")
    status="optimal" if result["gap"] == 0 else "bounded"
    if report.get("status") != status:
        raise ValueError("Report status differs from certificate")
    return {"schema_version":1,"tool":"verify_match","provenance":_provenance([path]),"valid":True,"status":status,**result}
