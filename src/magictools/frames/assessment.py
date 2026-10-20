"""Evidence about returned products, separate from successful HTTP execution."""
import hashlib
from pathlib import Path

from .geometry import region_intersects


def evidence():
    return {"frames": 0, "utcIntervals": 0, "footprints": 0, "spacetimeMatches": 0,
            "pointingEnvelopes": 0, "retrievalReferences": 0, "dataLinks": 0,
            "combinedOrUnknownTimeProducts": 0, "productKinds": {}}


def observe(counts, frame):
    counts["frames"] += 1
    kind = frame.get("productKind") or "unspecified"
    counts["productKinds"][kind] = counts["productKinds"].get(kind, 0) + 1
    combined = frame.get("timeSpanUnknown") or kind in {"stack", "cube", "visibility", "observation bundle", "target-pixel time cube"}
    start, end = frame.get("startMjd"), frame.get("endMjd")
    exposure = frame.get("exposureSeconds")
    interval = (start is not None and frame.get("timeScale") == "UTC" and not combined and
                ((end is not None and end > start) or (exposure is not None and exposure > 0)))
    # Parse, rather than treating any nonempty string as a usable footprint.
    footprint = region_intersects(frame.get("footprint"), 0, 0, 0) is not None
    counts["utcIntervals"] += bool(interval)
    counts["footprints"] += footprint
    counts["spacetimeMatches"] += bool(interval and footprint and frame.get("match") in {"footprint intersection", "exposure-time footprint intersection"})
    counts["pointingEnvelopes"] += bool(not footprint and frame.get("fovRadius") is not None)
    counts["retrievalReferences"] += bool(frame.get("url"))
    counts["dataLinks"] += frame.get("linkKind") == "DataLink"
    counts["combinedOrUnknownTimeProducts"] += bool(combined or not interval)


def query_assessment(receipt, counts, stats):
    succeeded = receipt["status"] in {"ok", "partial", "limited"}
    exhausted = stats.get("indexExhausted")
    if exhausted is None and "OK" in stats.get("queryStatuses", []):
        exhausted = not bool(stats.get("incomplete"))
    if not succeeded:
        exhausted = False
    reasons = []
    if exhausted is not True:
        reasons.append("Index exhaustion is unverified or the service truncated/failed the query")
    if receipt["rejectedRows"]:
        reasons.append("Malformed rows were dropped; inspect rejectedRows")
    if counts["utcIntervals"] < counts["frames"]:
        reasons.append("Some products lack a bounded single-exposure UTC interval")
    if counts["footprints"] < counts["frames"]:
        reasons.append("Some products lack a locally usable detector footprint")
    if counts["retrievalReferences"] < counts["frames"]:
        reasons.append("Some products lack a retrieval URL")
    outcome = "evidence-found" if counts["spacetimeMatches"] else "candidates-only" if counts["frames"] else "no-matches-in-queried-index" if succeeded and exhausted and not receipt["rejectedRows"] else "unverified"
    return {"querySucceeded": succeeded, "indexExhausted": exhausted,
            "outcome": outcome, "returnedProductEvidence": counts, "gaps": reasons,
            "entireRequestedRegionCovered": None, "allObservationsCovered": False,
            "detectabilityAssessed": False,
            "meaning": "Evidence-found means at least one returned product has a UTC interval and intersecting footprint; it does not certify continuous coverage or detection."}


def summarize(checks, unqueried=None):
    providers = {}
    for check in checks:
        p = providers.setdefault(check["provider"], {"queries": 0, "queryStates": {}, "outcomes": {}, "gaps": []})
        p["queries"] += 1
        state = check["status"]
        p["queryStates"][state] = p["queryStates"].get(state, 0)+1
        if state == "split":
            continue  # Child queries own the final evidence after subdivision.
        assessment = check.get("coverageAssessment", {})
        outcome = assessment.get("outcome", "unverified")
        p["outcomes"][outcome] = p["outcomes"].get(outcome, 0)+1
        p["gaps"] = sorted(set(p["gaps"]) | set(assessment.get("gaps", ["Legacy receipt has no coverage assessment"])))
    return {"providers": providers, "unqueried": unqueried or [],
            "entireRequestedRegionCovered": None, "allObservationsCovered": False,
            "detectabilityAssessed": False,
            "interpretation": "A completed query is not proof of coverage. Counts describe query windows and can overlap; native frames are deduplicated separately."}


def audit_run(directory, checksums=False):
    from .engine import status
    root = Path(directory).resolve()
    result = status(root)
    output = {"status": result["status"], "coverageAssessment": summarize(result["checks"], result.get("unqueried")),
              "frameCount": len(result["frames"])}
    if checksums:
        failures, checked = [], 0
        for check in result["checks"]:
            for request in check.get("requests", []):
                name, expected = request.get("responseFile"), request.get("sha256")
                if not name or not expected:
                    continue
                path = (root / check["artifactDirectory"] / name).resolve()
                if not path.is_relative_to(root):
                    failures.append({"jobId": check["jobId"], "file": name, "reason": "Artifact path is outside the run directory"})
                    continue
                checked += 1
                try:
                    digest = hashlib.sha256()
                    with path.open("rb") as stream:
                        for chunk in iter(lambda: stream.read(1024*1024), b""):
                            digest.update(chunk)
                    valid = digest.hexdigest() == expected
                except OSError:
                    valid = False
                if not valid:
                    failures.append({"jobId": check["jobId"], "file": name, "reason": "Missing artifact or SHA-256 mismatch"})
        output["artifactIntegrity"] = {"checked": checked, "failures": failures, "verified": checked > 0 and not failures}
    return output
