"""SQLite checkpoints, complete native responses, and streaming query events."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from collections import Counter, deque
from dataclasses import asdict
import csv
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
import uuid

from .adapters import Job, execute, stats
from .assessment import evidence, observe, query_assessment, summarize
from .credentials import Credentials
from .geometry import windows
from .inventory import coverage
from .model import PROVIDERS, LIMITATIONS, iso, mjd
from .parsers import merge_frames
from .profiles import NOIRLAB_INSTRUMENTS, KOA_INSTRUMENTS
from .transport import Transport, Blocked, BudgetExpired, ServiceUnavailable
from .progress import QueryProgress, query_context


def stable(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def job_id(job):
    return hashlib.sha256(stable({k: v for k, v in asdict(job).items() if k != "id"}).encode()).hexdigest()[:24]


def plan(x):
    field_windows = list(windows(x))
    jobs, unavailable = [], []
    for label in sorted({p["label"] for p in x.get("ephemeris", [])}):
        times = [p["mjd"] for p in x["ephemeris"] if p["label"] == label]
        if min(times) > mjd(x["start"]) or max(times) < mjd(x["end"]):
            unavailable.append({"name": "Supplied ephemeris", "hypothesis": label,
                                "reason": f"Track covers only {iso(min(times))} through {iso(max(times))}; requested dates outside this interval are unqueried."})
    for provider in x["providers"]:
        meta = PROVIDERS.get(provider) or {"name": x["services"][provider].get("name", provider), "modes": ["field", "ephemeris"]}
        can_field = "ephemeris" in meta["modes"] and provider != "ssois"
        if can_field and field_windows:
            jobs.extend(Job(provider, w["label"], window=w) for w in field_windows)
        elif provider == "ssois" and x["mode"] == "ephemeris":
            if x.get("experimentalSsoisEphemeris", False):
                jobs.extend(Job(provider, w["label"], window=w) for w in field_windows)
            else:
                from .model import SSOIS_EPHEMERIS_UNAVAILABLE
                unavailable.append({"provider": provider, "name": meta["name"], "mode": "ephemeris",
                                    "reason": SSOIS_EPHEMERIS_UNAVAILABLE})
        elif x["mode"] in meta["modes"] and x["mode"] in {"object", "orbits", "arc"}:
            if x["mode"] == "orbits":
                jobs.extend(Job(provider, o["label"], orbit=o) for o in x["orbits"])
            else:
                jobs.append(Job(provider, x.get("object", "MPC80 arc")))
        else:
            unavailable.append({"provider": provider, "name": meta["name"], "reason": "Supply an observer-specific ephemeris to cross-check this archive along a moving path." if can_field else f"Input mode {x['mode']} is not supported by this service."})
    jobs = [Job(j.provider, j.hypothesis, j.window, j.orbit, {"instrument": instrument})
            if instrument else j
            for j in jobs for instrument in (NOIRLAB_INSTRUMENTS.get(j.provider, [None]) if j.provider != "noirlab" else [None])]
    jobs = [Job(j.provider, j.hypothesis, j.window, j.orbit, {"instrument": inst}) if inst else j
            for j in jobs for inst in (KOA_INSTRUMENTS if j.provider == "koa" else [None])]
    # A custom endpoint is part of the persisted scientific job identity.
    for job in jobs:
        if job.provider in x.get("services", {}):
            job.params["service"] = x["services"][job.provider]
    for job in jobs:
        job.id = job_id(job)
    return jobs, unavailable


def connect(directory):
    db = sqlite3.connect(Path(directory) / "run.sqlite")
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, spec TEXT, status TEXT, receipt TEXT);
        CREATE TABLE IF NOT EXISTS records (job_id TEXT, key TEXT, value TEXT, PRIMARY KEY(job_id,key));
    """)
    return db


def _query(job, x, directory, credentials, transport_factory, progress=None, deadline=None):
    s = stats()
    counts = evidence()
    attempt = uuid.uuid4().hex
    folder = Path(directory) / "queries" / job.id / attempt
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    transport = transport_factory(folder, timeout=x["timeoutSeconds"], credentials=credentials)
    transport.shared_cache = Path(directory) / "metadata-cache"
    transport.deadline = deadline
    if progress is not None:
        transport.on_progress = progress.update
        progress.update(phase="querying archive")
    output = folder / "frames.ndjson.partial"
    receipt = {"jobId": job.id, "attemptId": attempt, "artifactDirectory": str(folder.relative_to(directory)), "provider": job.provider, "hypothesis": job.hypothesis,
               "startedAt": time.time(), "status": "ok", "count": 0, "queryUrl": None}
    try:
        with output.open("w") as stream:
            for f in execute(job, x, transport, s):
                stream.write(stable(f) + "\n")
                receipt["count"] += 1
                observe(counts, f)
                if progress is not None:
                    progress.update(candidateRows=receipt["count"])
                if deadline is not None and time.monotonic() >= deadline:
                    raise BudgetExpired("Fast-pass allowance reached; valid rows retained; resume with --deep")
        if s.get("incomplete"):
            receipt.update(status="limited", message=s["incomplete"])
        elif s.get("partial"):
            receipt.update(status="partial", message=s["partial"])
        elif s["rejected"]:
            receipt.update(status="partial", message="Valid rows retained; malformed rows counted separately")
        else:
            receipt["message"] = "Query completed; completeness is relative to this archive index"
    except BudgetExpired as error:
        receipt.update(status="deferred", message=str(error))
    except ServiceUnavailable as error:
        receipt.update(status="unavailable", message=credentials.redact(str(error)))
    except Blocked as error:
        receipt.update(status="blocked", message=credentials.redact(str(error)))
    except Exception as error:
        receipt.update(status="error", message=credentials.redact(str(error))[:1000])
    # Preserve valid records even if a later page or row parser fails.
    if output.exists():
        output.replace(folder / "frames.ndjson")
    receipt.update(receivedRows=s["received"], rejectedRows=s["rejected"], excludedRows=s["excluded"],
                   warnings=s["warnings"], requests=transport.calls, finishedAt=time.time())
    receipt["canSplitTime"] = s.get("canSplitTime", True)
    receipt["coverageAssessment"] = query_assessment(receipt, counts, s)
    if s.get("service"):
        receipt["service"] = s["service"]
    if s.get("remoteJob"):
        receipt["remoteJob"] = s["remoteJob"]
    if transport.calls:
        receipt["queryUrl"] = transport.calls[0]["url"]
    (folder / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")
    return receipt, folder / "frames.ndjson"


def run(x, directory, *, resume=False, credentials=None, emit=None, transport_factory=Transport, retry_partial=False,
        progress_interval=5, max_seconds=None, query_seconds=10):
    from .model import finite
    progress_interval = finite(progress_interval, "progress interval", 0.01, 3600)
    if max_seconds is not None:
        max_seconds = finite(max_seconds, "fast-pass budget", 0.01, 86400)
    query_seconds = finite(query_seconds, "query allowance", 0.01, 86400)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    # One writer per run directory; concurrent agents should use distinct runs.
    lock = directory / ".writer.lock"
    lock_fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        import fcntl
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (ImportError, BlockingIOError):
        os.close(lock_fd)
        raise ValueError("Run is already active, or advisory locking is unavailable on this platform") from None
    try:
        return _run_locked(x, directory, resume, credentials or Credentials(), emit or (lambda event: None),
                           transport_factory, retry_partial, progress_interval, max_seconds, query_seconds)
    finally:
        os.close(lock_fd)


def _run_locked(x, directory, resume, credentials, emit, transport_factory, retry_partial, progress_interval, max_seconds, query_seconds):
    jobs, unavailable = plan(x)
    signature = hashlib.sha256(stable({k: v for k, v in x.items() if k not in {"timeoutSeconds", "concurrency"}}).encode()).hexdigest()
    db = connect(directory)
    try:
        old = db.execute("SELECT value FROM meta WHERE key='signature'").fetchone()
        if old and (not resume or old[0] != signature):
            raise ValueError("Run directory exists; use --resume with the same scientific input, or choose a new directory")
        db.execute("INSERT OR REPLACE INTO meta VALUES ('signature',?)", (signature,))
        db.execute("INSERT OR REPLACE INTO meta VALUES ('input',?)", (stable(x),))
        db.execute("INSERT OR REPLACE INTO meta VALUES ('unqueried',?)", (stable(unavailable),))
        for job in jobs:
            db.execute("INSERT OR IGNORE INTO jobs VALUES (?,?,?,NULL)", (job.id, stable(asdict(job)), "pending"))
        db.commit()
        successful = {"ok", "split"} | (set() if retry_partial else {"partial"})
        queued = [Job(**json.loads(spec)) for spec, state in db.execute("SELECT spec,status FROM jobs ORDER BY rowid") if state not in successful]
        total = db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        finished = total - len(queued)
        candidate_rows = 0
        started = time.monotonic()
        deadline = started + max_seconds if max_seconds is not None else None
        # Round-robin providers, with at most one active query each. A long
        # ephemeris must not monopolize all workers before other surveys start.
        provider_order = deque(dict.fromkeys(j.provider for j in queued))
        db.execute("INSERT OR REPLACE INTO meta VALUES ('execution',?)", (stable({
            "mode": "fast" if deadline is not None else "deep", "budgetSeconds": max_seconds,
            "queryAllowanceSeconds": query_seconds if deadline is not None else None}),))
        db.commit()
        next_progress = started + progress_interval
        emit({"type": "plan", "queries": len(queued), "totalQueries": total, "reusedQueries": finished,
              "unqueried": unavailable, "runDirectory": str(directory),
              "searchMode": "fast" if deadline is not None else "deep", "budgetSeconds": max_seconds,
              "surveys": survey_results(x, [], unavailable, jobs)})
        with ThreadPoolExecutor(max_workers=x["concurrency"]) as pool:
            active = {}
            while queued or active:
                if deadline is not None and time.monotonic() >= deadline:
                    for job in queued:
                        defer_job(db, job, "deferred", "Fast-pass budget reached before this query; resume with --deep")
                    queued.clear()
                    db.commit()
                while queued and len(active) < x["concurrency"]:
                    busy = {j.provider for j, _ in active.values()}
                    selected = None
                    for _ in range(len(provider_order)):
                        provider = provider_order[0]
                        provider_order.rotate(-1)
                        if provider not in busy:
                            selected = next((i for i, j in enumerate(queued) if j.provider == provider), None)
                            if selected is not None:
                                break
                    if selected is None:
                        break
                    job = queued.pop(selected)
                    db.execute("UPDATE jobs SET status='running' WHERE id=?", (job.id,))
                    db.commit()
                    emit({"type": "query-start", **query_context(job)})
                    progress = QueryProgress()
                    query_deadline = min(deadline, time.monotonic() + query_seconds) if deadline is not None else None
                    active[pool.submit(_query, job, x, directory, credentials, transport_factory, progress, query_deadline)] = (job, progress)
                if not active:
                    break
                done, _ = wait(active, timeout=max(0, next_progress - time.monotonic()), return_when=FIRST_COMPLETED)
                if time.monotonic() >= next_progress:
                    snapshots = [{**query_context(job), **progress.snapshot()}
                                 for job, progress in active.values()]
                    emit({"type": "progress", "elapsedSeconds": round(time.monotonic() - started, 1),
                          "finishedQueries": finished, "totalQueries": total, "queuedQueries": len(queued),
                          "candidateRowsThisRun": candidate_rows + sum(p["candidateRows"] for p in snapshots), "active": snapshots})
                    next_progress = time.monotonic() + progress_interval
                for future in done:
                    job, progress = active.pop(future)
                    receipt, frame_file = future.result()
                    if receipt["status"] == "unavailable":
                        remaining = [j for j in queued if j.provider == job.provider]
                        for skipped in remaining:
                            defer_job(db, skipped, receipt["status"], receipt["message"], job.id)
                        queued = [j for j in queued if j.provider != job.provider]
                        finished += len(remaining)
                    # Failed retries must not erase valid rows from earlier attempts.
                    if receipt["status"] == "ok":
                        db.execute("DELETE FROM records WHERE job_id=?", (job.id,))
                    if frame_file.exists():
                        with frame_file.open() as stream:
                            for line in stream:
                                f = json.loads(line)
                                previous = db.execute("SELECT value FROM records WHERE job_id=? AND key=?", (job.id, f["key"])).fetchone()
                                if previous:
                                    f = merge_frames(json.loads(previous[0]), f)
                                db.execute("INSERT OR REPLACE INTO records VALUES (?,?,?)", (job.id, f["key"], stable(f)))
                    # Split actual service overflow, never arbitrary first-N client rows.
                    if receipt["status"] == "limited" and receipt.get("canSplitTime", True) and job.window and job.window["end"]-job.window["start"] > 1:
                        midpoint = (job.window["start"]+job.window["end"])/2
                        for begin, end in [(job.window["start"], midpoint), (midpoint, job.window["end"])]:
                            child = Job(job.provider, job.hypothesis, {**job.window, "start": begin, "end": end}, job.orbit, job.params)
                            child.id = job_id(child)
                            inserted = db.execute("INSERT OR IGNORE INTO jobs VALUES (?,?,?,NULL)", (child.id, stable(asdict(child)), "pending")).rowcount
                            if inserted:
                                queued.append(child)
                                total += 1
                        receipt["status"] = "split"
                        receipt["message"] += "; automatically subdivided into persisted child queries"
                    db.execute("UPDATE jobs SET status=?,receipt=? WHERE id=?", (receipt["status"], stable(receipt), job.id))
                    db.commit()
                    finished += 1
                    candidate_rows += receipt["count"]
                    emit({"type": "query-complete", **receipt, **query_context(job), "finishedQueries": finished, "totalQueries": total})
        emit({"type": "finalizing"})
        result = result_from_db(db)
        temp = directory / "manifest.json.partial"
        temp.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
        temp.replace(directory / "manifest.json")
        export_csv(result, directory / "frames.csv")
        emit({"type": "complete", "count": len(result["frames"]), "status": result["status"],
              "surveys": result["surveys"], "execution": result["execution"],
              "manifest": str(directory / "manifest.json"), "csv": str(directory / "frames.csv")})
        return result
    finally:
        db.close()


def defer_job(db, job, state, message, cause=None):
    # Keep an earlier attempt's receipt and rows, including its raw artifacts.
    previous = db.execute("SELECT receipt FROM jobs WHERE id=?", (job.id,)).fetchone()[0]
    receipt = json.loads(previous) if previous else {
        **query_context(job), "count": 0, "requests": [], "receivedRows": 0,
        "rejectedRows": 0, "excludedRows": 0}
    receipt.update(status=state, message=message, deferredAt=time.time())
    if "coverageAssessment" in receipt:
        receipt["coverageAssessment"].update(querySucceeded=False, indexExhausted=False)
    if cause:
        receipt["deferredBecauseJob"] = cause
    db.execute("UPDATE jobs SET status=?,receipt=? WHERE id=?", (state, stable(receipt), job.id))


def survey_results(x, checks, unavailable, jobs=None, frames=()):
    """Provider summaries are candidates, not assertions of sky coverage."""
    from .inventory import DATASETS
    result = []
    for provider in x["providers"]:
        own = [c for c in checks if c["provider"] == provider]
        states = Counter(c["status"] for c in own)
        gap = next((u for u in unavailable if u.get("provider") == provider), None)
        pending = sum(j.provider == provider for j in jobs or []) if not own else sum(
            n for s, n in states.items() if s not in {"ok", "split", "partial"})
        matches = [f for f in frames if any(p["provider"] == provider for p in f["provenance"])]
        state = next((s for s in ("unavailable", "blocked", "error", "deferred", "running", "pending", "limited", "partial") if states[s]),
                     "unqueried" if gap else "ok" if own else "pending")
        messages = list(dict.fromkeys(c.get("message", "") for c in own if c["status"] not in {"ok", "split"}))
        if gap:
            messages.append(gap["reason"])
        result.append({"provider": provider,
                       "name": (PROVIDERS.get(provider) or x.get("services", {}).get(provider, {})).get("name", provider),
                       "status": state, "queryStates": dict(states), "remainingQueries": pending,
                       "candidateDatasets": [name for name, providers, _ in DATASETS if provider in providers],
                       "matchedSurveys": sorted({f["survey"] for f in matches if f.get("survey")}),
                       "frameCount": len(matches), "messages": messages,
                       "interpretation": "Candidate datasets describe archive holdings; only returned frames are matches, not proof of detection or complete coverage."})
    return result


def result_from_db(db):
    meta = dict(db.execute("SELECT key,value FROM meta"))
    merged = {}
    for key, value in db.execute("SELECT key,value FROM records ORDER BY key,job_id"):
        row = json.loads(value)
        merged[key] = merge_frames(merged[key], row) if key in merged else row
    checks = []
    for spec, state, receipt in db.execute("SELECT spec,status,receipt FROM jobs ORDER BY rowid"):
        if receipt:
            checks.append({**json.loads(receipt), "status": state})
        else:
            job = json.loads(spec)
            checks.append({"jobId": job["id"], "provider": job["provider"], "hypothesis": job["hypothesis"],
                           "status": state, "count": 0, "message": "Query has not completed"})
    frames = sorted(merged.values(), key=lambda f: (f["startMjd"] if f["startMjd"] is not None else math_inf(), f["key"]))
    x, unavailable = json.loads(meta["input"]), json.loads(meta["unqueried"])
    return {"version": 2, "queriedAt": time.time(), "input": x,
            "execution": json.loads(meta.get("execution", '{}')),
            "surveys": survey_results(x, checks, unavailable, frames=frames),
            "status": "complete" if all(c["status"] in {"ok", "split"} for c in checks) and checks and not json.loads(meta["unqueried"]) else "incomplete",
            "frames": frames, "checks": checks, "datasetCoverage": coverage(checks), "coverageAssessment": summarize(checks, json.loads(meta["unqueried"])),
            "unqueried": json.loads(meta["unqueried"]), "limitations": LIMITATIONS}


def math_inf():
    return float("inf")


def status(directory):
    if not (Path(directory) / "run.sqlite").is_file():
        raise ValueError("No initialized search in this run directory")
    db = connect(directory)
    try:
        if not db.execute("SELECT 1 FROM meta WHERE key='input'").fetchone():
            raise ValueError("No initialized search in this run directory")
        return result_from_db(db)
    finally:
        db.close()


def export_csv(result, path):
    columns = ["key", "survey", "instrument", "id", "startMjd", "endMjd", "timeScale",
               "filter", "exposureSeconds", "productKind", "match", "hypotheses", "url", "linkKind", "providers"]
    path = Path(path)
    temp = path.with_suffix(path.suffix+".partial")
    with temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for frame in result["frames"]:
            row = {k: frame.get(k) for k in columns}
            row["hypotheses"] = ";".join(frame["hypotheses"])
            row["providers"] = ";".join(sorted({p["provider"] for p in frame["provenance"]}))
            for k, v in row.items():
                if isinstance(v, str) and v.lstrip().startswith(("=", "+", "-", "@")):
                    row[k] = "'"+v
            writer.writerow(row)
    temp.replace(path)


def merge_manifests(results):
    frames, checks = {}, []
    for result in results:
        checks.extend(result["checks"])
        for frame in result["frames"]:
            frames[frame["key"]] = merge_frames(frames[frame["key"]], frame) if frame["key"] in frames else frame
    return {"version": 2, "input": [r["input"] for r in results], "status": "complete" if all(r.get("status") == "complete" for r in results) else "incomplete",
            "frames": list(frames.values()), "checks": checks, "limitations": LIMITATIONS,
            "datasetCoverage": coverage(checks), "coverageAssessment": summarize(checks, [u for r in results for u in r.get("unqueried", [])]),
            "unqueried": [u for r in results for u in r.get("unqueried", [])]}
